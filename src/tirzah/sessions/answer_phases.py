"""Split answer pipeline into retrieval and synthesis phases for interpretive PLAN execution.

``retrieve_for_answer`` gathers context only (including deep retrieval chunks).
``synthesize_from_retrieval`` runs the answer adapter and persists the exchange.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pymongo.database import Database

from tirzah.config import AppConfig, RuntimeConfig
from tirzah.domains.registry import (
    clean_domain_id,
    conversation_domain_id_for_session,
)
from tirzah.retrieval.queries import node_identity
from tirzah.sessions import interaction as ix


@dataclass
class AnswerRetrievalPackage:
    query: str
    session_id: str
    focus_node_id: str | None
    selected_node_id: str | None
    retrieval_mode: str
    runtime_config: dict[str, Any]
    process_trace: list[dict[str, Any]] = field(default_factory=list)
    process_run_id: str | None = None
    project_domain_id: str | None = None
    conversation_domain_id: str | None = None
    prompt: dict[str, Any] | None = None
    retrieval_status: str | None = None
    controller_decision: Any = None
    pre_built_answer: dict[str, Any] | None = None
    useful_chunks: list[dict[str, Any]] | None = None
    deep_trace: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "session_id": self.session_id,
            "focus_node_id": self.focus_node_id,
            "selected_node_id": self.selected_node_id,
            "retrieval_mode": self.retrieval_mode,
            "runtime_config": self.runtime_config,
            "process_trace": self.process_trace,
            "process_run_id": self.process_run_id,
            "project_domain_id": self.project_domain_id,
            "conversation_domain_id": self.conversation_domain_id,
            "prompt": self.prompt,
            "retrieval_status": self.retrieval_status,
            "controller_decision": self.controller_decision,
            "pre_built_answer": self.pre_built_answer,
            "useful_chunks": self.useful_chunks,
            "deep_trace": self.deep_trace,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnswerRetrievalPackage":
        return cls(
            query=str(value.get("query") or ""),
            session_id=str(value.get("session_id") or "default"),
            focus_node_id=value.get("focus_node_id"),
            selected_node_id=value.get("selected_node_id"),
            retrieval_mode=str(value.get("retrieval_mode") or "direct"),
            runtime_config=dict(value.get("runtime_config") or {}),
            process_trace=list(value.get("process_trace") or []),
            process_run_id=value.get("process_run_id"),
            project_domain_id=value.get("project_domain_id"),
            conversation_domain_id=value.get("conversation_domain_id"),
            prompt=value.get("prompt"),
            retrieval_status=value.get("retrieval_status"),
            controller_decision=value.get("controller_decision"),
            pre_built_answer=value.get("pre_built_answer"),
            useful_chunks=value.get("useful_chunks"),
            deep_trace=value.get("deep_trace"),
        )


def retrieve_for_answer(
    db: Database,
    config: AppConfig,
    query: str,
    focus_node_id: str | None = None,
    session_id: str = "default",
    answer_adapter_name: str | None = None,
    ollama_model: str | None = None,
    retrieval_mode: str | None = None,
    project_domain_id: str | None = None,
    conversation_domain_id: str | None = None,
    web_research: bool | None = None,
) -> dict[str, Any]:
    """Retrieval phase only — no answer adapter, no exchange persist."""
    runtime_config, process_trace, process_run_id = _begin_answer_request(
        db,
        config,
        query=query,
        focus_node_id=focus_node_id,
        session_id=session_id,
        answer_adapter_name=answer_adapter_name,
        ollama_model=ollama_model,
        retrieval_mode=retrieval_mode,
        web_research=web_research,
    )
    package = AnswerRetrievalPackage(
        query=query,
        session_id=session_id,
        focus_node_id=focus_node_id,
        selected_node_id=focus_node_id,
        retrieval_mode=runtime_config.retrieval_mode,
        runtime_config=runtime_config.model_dump(mode="json"),
        process_trace=process_trace,
        process_run_id=process_run_id,
        project_domain_id=project_domain_id,
        conversation_domain_id=conversation_domain_id,
    )
    if runtime_config.retrieval_mode == "deep":
        return _retrieve_deep(db, config, runtime_config, package)
    if runtime_config.retrieval_mode == "agentic":
        return _retrieve_agentic(db, config, runtime_config, package)
    return _retrieve_direct(db, config, runtime_config, package)


def synthesize_from_retrieval(
    db: Database,
    config: AppConfig,
    package: AnswerRetrievalPackage | dict[str, Any],
) -> dict[str, Any]:
    """Synthesis + persist phase using a retrieval package."""
    if isinstance(package, dict):
        package = AnswerRetrievalPackage.from_dict(package)
    runtime_config = RuntimeConfig.model_validate(package.runtime_config)
    if package.pre_built_answer:
        return _persist_prebuilt_answer(db, config, runtime_config, package)
    if package.useful_chunks is not None:
        return _synthesize_deep_and_persist(db, config, runtime_config, package)
    return _synthesize_and_persist(db, config, runtime_config, package)


def build_retrieval_package_from_context_bundle(
    db: Database,
    config: AppConfig,
    *,
    query: str,
    session_id: str,
    bundle: dict[str, Any],
    focus_node_id: str | None = None,
    answer_adapter_name: str | None = None,
    ollama_model: str | None = None,
    retrieval_mode: str | None = None,
    web_research: bool | None = None,
    project_domain_id: str | None = None,
    conversation_domain_id: str | None = None,
) -> dict[str, Any]:
    """Build an answer envelope from accumulated interpretive tool_results."""
    tool_results = list((bundle or {}).get("tool_results") or [])
    if not tool_results:
        return {"ok": False, "reason": "empty_context_bundle"}
    runtime_config, process_trace, process_run_id = _begin_answer_request(
        db,
        config,
        query=query,
        focus_node_id=focus_node_id,
        session_id=session_id,
        answer_adapter_name=answer_adapter_name,
        ollama_model=ollama_model,
        retrieval_mode=retrieval_mode or "agentic",
        web_research=web_research,
    )
    try:
        prompt = ix.build_agentic_answer_envelope(
            query=query,
            tool_results=tool_results,
            **ix._token_budget_pair(config, runtime_config),
            proposed_controller_decision=ix.final_controller_decision_from_trace(process_trace),
            context_proposal=ix.final_context_proposal_from_trace(process_trace),
        )
        prompt = ix.inject_history_into_prompt(
            prompt,
            ix.render_session_history_block(db, config, session_id=session_id, query=query),
        )
    except Exception as error:
        return {
            "ok": False,
            "reason": "context_bundle_envelope_failed",
            "message": str(error),
            "process_trace": process_trace,
        }
    selected_node_id = focus_node_id or _node_id_from_tool_results(tool_results)
    package = AnswerRetrievalPackage(
        query=query,
        session_id=session_id,
        focus_node_id=focus_node_id,
        selected_node_id=selected_node_id,
        retrieval_mode=runtime_config.retrieval_mode,
        runtime_config=runtime_config.model_dump(mode="json"),
        process_trace=process_trace,
        process_run_id=process_run_id,
        project_domain_id=project_domain_id,
        conversation_domain_id=conversation_domain_id,
        prompt=prompt,
        retrieval_status=(prompt.get("context_metadata") or {}).get("retrieval_status"),
        controller_decision=(prompt.get("context_metadata") or {}).get("controller_decision"),
    )
    package.process_trace.append(
        {
            "step": "context_bundle",
            "input": {"query": query, "tool_result_count": len(tool_results)},
            "output": {
                "ok": True,
                "retrieval_status": package.retrieval_status,
                "tools": [row.get("tool") for row in tool_results],
            },
        }
    )
    return {"ok": True, "package": package.to_dict(), "phase": "context_bundle"}


def synthesize_from_context_bundle(
    db: Database,
    config: AppConfig,
    *,
    query: str,
    session_id: str,
    bundle: dict[str, Any],
    **answer_kwargs: Any,
) -> dict[str, Any]:
    """Synthesize and persist using granular plan context artifacts."""
    built = build_retrieval_package_from_context_bundle(
        db,
        config,
        query=query,
        session_id=session_id,
        bundle=bundle,
        focus_node_id=answer_kwargs.get("focus_node_id"),
        answer_adapter_name=answer_kwargs.get("answer_adapter_name"),
        ollama_model=answer_kwargs.get("ollama_model"),
        retrieval_mode=answer_kwargs.get("retrieval_mode"),
        web_research=answer_kwargs.get("web_research"),
        project_domain_id=answer_kwargs.get("project_domain_id"),
        conversation_domain_id=answer_kwargs.get("conversation_domain_id"),
    )
    if not built.get("ok"):
        return built
    return synthesize_from_retrieval(db, config, built["package"])


def _node_id_from_tool_results(tool_results: list[dict[str, Any]]) -> str | None:
    for result in reversed(tool_results):
        if not result.get("ok"):
            continue
        output = result.get("output") or {}
        if not isinstance(output, dict):
            continue
        if result.get("tool") == "compile_context" and output.get("focus_node_id"):
            return str(output["focus_node_id"])
        if result.get("tool") == "search_nodes":
            matches = output.get("matches") or []
            if matches and matches[0].get("node_id"):
                return str(matches[0]["node_id"])
    return None


def _begin_answer_request(
    db: Database,
    config: AppConfig,
    *,
    query: str,
    focus_node_id: str | None,
    session_id: str,
    answer_adapter_name: str | None,
    ollama_model: str | None,
    retrieval_mode: str | None,
    web_research: bool | None,
) -> tuple[RuntimeConfig, list[dict[str, Any]], str | None]:
    process_trace: list[dict[str, Any]] = [
        {
            "step": "user_prompt",
            "input": {
                "query": query,
                "focus_node_id": focus_node_id,
                "session_id": session_id,
                "requested_adapter": answer_adapter_name,
                "requested_model": ollama_model,
                "retrieval_mode": retrieval_mode or config.runtime.retrieval_mode,
            },
            "output": {"submitted_prompt": query},
        }
    ]
    runtime_config = config.runtime.model_copy()
    if answer_adapter_name:
        runtime_config.answer_adapter = answer_adapter_name
    if ollama_model:
        runtime_config.ollama_model = ollama_model
    if retrieval_mode:
        runtime_config.retrieval_mode = retrieval_mode
    if web_research is not None:
        runtime_config.web_research_enabled = web_research
    if web_research and runtime_config.retrieval_mode != "agentic":
        runtime_config.retrieval_mode = "agentic"
        process_trace[0]["output"]["retrieval_mode_override"] = "agentic_for_web_research"
    if runtime_config.retrieval_mode == "agentic" and not focus_node_id and ix.is_low_intent_query(query):
        runtime_config.retrieval_mode = "direct"
        process_trace[0]["output"]["retrieval_mode_override"] = "direct_no_context_low_intent"
    process_run_id = ix.start_answer_process_run(
        db,
        session_id=session_id,
        retrieval_mode=runtime_config.retrieval_mode,
    )
    return runtime_config, process_trace, process_run_id


def _retrieve_direct(
    db: Database,
    config: AppConfig,
    runtime_config: RuntimeConfig,
    package: AnswerRetrievalPackage,
) -> dict[str, Any]:
    try:
        preparation = ix.prepare_direct_answer_prompt(
            db,
            config=config,
            query=package.query,
            focus_node_id=package.focus_node_id,
            session_id=package.session_id,
            runtime_config=runtime_config,
        )
    except Exception as error:
        return _retrieval_failed(db, package, "retrieval_failed", "retrieval_context", error, config.runtime)
    package.selected_node_id = preparation["selected_node_id"]
    package.prompt = preparation["prompt"]
    package.retrieval_status = preparation["retrieval_status"]
    package.controller_decision = preparation["prompt"]["context_metadata"].get("controller_decision")
    package.process_trace.append(
        {
            "step": "retrieval_context",
            "input": {
                "query": package.query,
                "provided_focus_node_id": package.focus_node_id,
                "selected_node_id": package.selected_node_id,
                "selected_node_source": preparation["selected_node_source"],
                "mode": package.retrieval_mode,
            },
            "output": preparation["retrieval_output"],
        }
    )
    return {"ok": True, "package": package.to_dict(), "phase": "retrieval"}


def _retrieve_agentic(
    db: Database,
    config: AppConfig,
    runtime_config: RuntimeConfig,
    package: AnswerRetrievalPackage,
) -> dict[str, Any]:
    try:
        tool_results = ix.run_memory_agent_loop(
            db=db,
            runtime_config=runtime_config,
            query=package.query,
            focus_node_id=package.focus_node_id,
            session_id=package.session_id,
            max_iterations=config.retrieval.memory_agent_max_iterations,
            process_trace=package.process_trace,
        )
        prompt = ix.build_agentic_answer_envelope(
            query=package.query,
            tool_results=tool_results,
            **ix._token_budget_pair(config, runtime_config),
            proposed_controller_decision=ix.final_controller_decision_from_trace(package.process_trace),
            context_proposal=ix.final_context_proposal_from_trace(package.process_trace),
        )
        prompt = ix.inject_history_into_prompt(
            prompt,
            ix.render_session_history_block(db, config, session_id=package.session_id, query=package.query),
        )
    except Exception as error:
        return _retrieval_failed(
            db, package, "agentic_retrieval_failed", "memory_agent_iteration", error, runtime_config
        )
    package.prompt = prompt
    package.retrieval_status = prompt["context_metadata"]["retrieval_status"]
    package.controller_decision = prompt["context_metadata"].get("controller_decision")
    package.selected_node_id = package.focus_node_id
    return {"ok": True, "package": package.to_dict(), "phase": "retrieval"}


def _retrieve_deep(
    db: Database,
    config: AppConfig,
    runtime_config: RuntimeConfig,
    package: AnswerRetrievalPackage,
) -> dict[str, Any]:
    from tirzah.retrieval.budget import request_budget_plan
    from tirzah.retrieval.deep import (
        make_planner,
        make_scorer,
        make_triager,
        pack_synthesis_chunks,
        run_deep_retrieval,
    )

    identity = ix.first_active_agent_identity(db) if package.session_id else None
    adapter = ix.answer_adapter(runtime_config)
    embedder = (lambda text: ix.build_query_embedding(runtime_config, text)) if runtime_config else None
    try:
        deep_result = run_deep_retrieval(
            db,
            package.query,
            config=config,
            planner=make_planner(adapter),
            triager=make_triager(adapter),
            scorer=make_scorer(adapter) if getattr(config.retrieval, "deep_sufficiency_scoring", False) else None,
            query_embedding=ix.build_query_embedding(runtime_config, package.query),
            identity=identity,
            embedder=embedder,
        )
    except Exception as error:
        return _retrieval_failed(db, package, "deep_retrieval_failed", "deep_retrieval", error, runtime_config)
    useful = deep_result["useful_chunks"]
    # Bound the synthesis context to the per-request model's budget; the
    # history block is added (and re-packed) at synthesis time.
    packed = pack_synthesis_chunks(
        package.query, useful, request_budget_plan(config, runtime=runtime_config)
    )
    kept = packed["kept"]
    used_node_ids = [nid for nid in (node_identity(c) for c in kept) if nid]
    package.useful_chunks = list(kept)
    package.prompt = {
        "prompt_text": "",
        "budget": packed["budget"],
        "context_metadata": {
            "included": [{"node_id": nid} for nid in used_node_ids],
            "evidence_summary": {
                "included_node_count": len(used_node_ids),
                "source_documents": [],
            },
            "skipped": packed["skipped"],
            "skipped_count": len(packed["skipped"]),
        },
    }
    package.retrieval_status = "deep_context" if kept else "deep_no_context"
    package.deep_trace = list(deep_result.get("trace") or [])
    package.selected_node_id = package.focus_node_id
    for entry in package.deep_trace:
        if entry.get("step") == "sufficiency":
            package.process_trace.append(
                {
                    "step": "sufficiency",
                    "input": {},
                    "output": {
                        "context_sufficiency_score": entry.get("context_sufficiency_score"),
                        "recursion": entry.get("recursion"),
                        "remaining_uncertainty_count": len(entry.get("remaining_uncertainty") or []),
                    },
                }
            )
    package.process_trace.append(
        {
            "step": "deep_retrieval",
            "input": {"query": package.query, "session_id": package.session_id, "mode": "deep"},
            "output": {
                "ok": True,
                "useful_count": len(useful),
                "synthesis_kept_count": len(kept),
                "synthesis_budget_skipped_count": len(packed["skipped"]),
                "rounds": deep_result["rounds"],
                "trace": deep_result["trace"],
            },
        }
    )
    return {"ok": True, "package": package.to_dict(), "phase": "retrieval"}


def _synthesize_and_persist(
    db: Database,
    config: AppConfig,
    runtime_config: RuntimeConfig,
    package: AnswerRetrievalPackage,
) -> dict[str, Any]:
    prompt = package.prompt or {}
    evidence_summary = (prompt.get("context_metadata") or {}).get("evidence_summary")
    adapter_step = {
        "step": "answer_adapter",
        "input": {
            "adapter": runtime_config.answer_adapter,
            "model": runtime_config.ollama_model,
            "prompt_text": prompt.get("prompt_text"),
            "controller_decision": package.controller_decision,
            "evidence_summary": evidence_summary,
            "timeout_seconds": runtime_config.ollama_timeout_seconds
            if runtime_config.answer_adapter.startswith("ollama")
            else None,
        },
        "output": {},
    }
    package.process_trace.append(adapter_step)
    try:
        answer = ix.answer_adapter(runtime_config).answer(prompt)
    except Exception as error:
        ix.finish_answer_process_run(
            db,
            package.process_run_id,
            status="blocked",
            current_step_id="answer_adapter_failed",
            exception=ix.answer_exception_payload(
                "answer_adapter_failed",
                "Inspect adapter/model configuration and retry.",
                error,
            ),
        )
        adapter_step["output"] = {"ok": False, "error": str(error)}
        return ix.attach_answer_activity(
            {
                "ok": False,
                "reason": "answer_adapter_failed",
                "message": str(error),
                "adapter": runtime_config.answer_adapter,
                "model": runtime_config.ollama_model,
                "focus_node_id": package.selected_node_id,
                "process_run_id": package.process_run_id,
                "process_trace": package.process_trace,
            }
        )
    from tirzah.adapters.answer import instrumentation_from_answer

    adapter_step["output"] = {
        "ok": True,
        "answer": answer["answer"],
        "used_node_ids": answer["used_node_ids"],
        "adapter": answer["adapter"],
        "model": answer.get("model"),
        **instrumentation_from_answer(answer),
    }
    return _persist_answer(db, config, runtime_config, package, answer)


def _synthesize_deep_and_persist(
    db: Database,
    config: AppConfig,
    runtime_config: RuntimeConfig,
    package: AnswerRetrievalPackage,
) -> dict[str, Any]:
    from tirzah.adapters.answer import instrumentation_from_answer
    from tirzah.retrieval.budget import request_budget_plan
    from tirzah.retrieval.deep import (
        build_synthesis_prompt,
        pack_synthesis_chunks,
        synthesize_answer_result,
    )

    history_block = ix.render_session_history_block(
        db, config, session_id=package.session_id, query=package.query
    )
    # Re-pack with the history block in the prompt so the whole synthesis
    # input, not just the context, stays inside the model's budget.
    packed = pack_synthesis_chunks(
        package.query,
        list(package.useful_chunks or []),
        request_budget_plan(config, runtime=runtime_config),
        history_block,
    )
    useful = packed["kept"]
    used_node_ids = [nid for nid in (node_identity(c) for c in useful) if nid]
    if isinstance(package.prompt, dict) and packed["budget"]:
        package.prompt["budget"] = packed["budget"]
        metadata = package.prompt.setdefault("context_metadata", {})
        metadata["skipped"] = list(metadata.get("skipped") or []) + packed["skipped"]
        metadata["skipped_count"] = len(metadata["skipped"])
    adapter_step = {
        "step": "answer_adapter",
        "input": {
            "adapter": runtime_config.answer_adapter,
            "model": runtime_config.ollama_model,
            "mode": "deep_synthesis",
            # The full synthesis input, so the LLM debugging view (llm_calls)
            # shows deep-mode In→Out like every other call.
            "prompt_text": build_synthesis_prompt(package.query, useful, history_block),
            "useful_count": len(useful),
            "budget": packed["budget"],
            "timeout_seconds": runtime_config.ollama_timeout_seconds
            if runtime_config.answer_adapter.startswith("ollama")
            else None,
        },
        "output": {},
    }
    package.process_trace.append(adapter_step)
    try:
        synthesis = synthesize_answer_result(
            package.query,
            useful,
            ix.answer_adapter(runtime_config),
            history_block=history_block,
        )
    except Exception as error:
        ix.finish_answer_process_run(
            db,
            package.process_run_id,
            status="blocked",
            current_step_id="answer_adapter_failed",
            exception=ix.answer_exception_payload(
                "answer_adapter_failed",
                "Inspect adapter/model configuration and retry.",
                error,
            ),
        )
        adapter_step["output"] = {"ok": False, "error": str(error)}
        return ix.attach_answer_activity(
            {
                "ok": False,
                "reason": "answer_adapter_failed",
                "message": str(error),
                "adapter": runtime_config.answer_adapter,
                "model": runtime_config.ollama_model,
                "focus_node_id": package.selected_node_id,
                "process_run_id": package.process_run_id,
                "process_trace": package.process_trace,
            }
        )
    answer = {
        "answer": str(synthesis.get("answer") or ""),
        "used_node_ids": used_node_ids,
        "adapter": synthesis.get("adapter") or runtime_config.answer_adapter,
        "model": synthesis.get("model") or runtime_config.ollama_model,
        **instrumentation_from_answer(synthesis),
    }
    adapter_step["output"] = {
        "ok": True,
        "answer": answer["answer"],
        "used_node_ids": answer["used_node_ids"],
        "adapter": answer["adapter"],
        "model": answer.get("model"),
        **instrumentation_from_answer(answer),
    }
    return _persist_answer(db, config, runtime_config, package, answer)


def _persist_prebuilt_answer(
    db: Database,
    config: AppConfig,
    runtime_config: RuntimeConfig,
    package: AnswerRetrievalPackage,
) -> dict[str, Any]:
    answer = package.pre_built_answer or {}
    return _persist_answer(db, config, runtime_config, package, answer)


def _persist_answer(
    db: Database,
    config: AppConfig,
    runtime_config: RuntimeConfig,
    package: AnswerRetrievalPackage,
    answer: dict[str, Any],
) -> dict[str, Any]:
    prompt = package.prompt or {}
    try:
        exchange_id = ix.save_exchange(
            db,
            query=package.query,
            answer=answer,
            prompt=prompt,
            focus_node_id=package.selected_node_id,
            session_id=package.session_id,
            process_trace=package.process_trace,
            project_domain_id=package.project_domain_id,
            conversation_domain_id=package.conversation_domain_id,
        )
        ix.schedule_turn_embedding(
            db, config, runtime_config, exchange_id, package.session_id, package.query, answer["answer"]
        )
        ix.schedule_chunking(
            db, config, runtime_config, exchange_id, package.session_id, package.query, answer["answer"]
        )
    except Exception as error:
        ix.finish_answer_process_run(
            db,
            package.process_run_id,
            status="blocked",
            current_step_id="answer_save_failed",
            exception=ix.answer_exception_payload(
                "answer_save_failed",
                "Inspect exchange persistence and retry.",
                error,
            ),
        )
        package.process_trace.append(
            {
                "step": "save_exchange",
                "input": {"session_id": package.session_id, "focus_node_id": package.selected_node_id},
                "output": {"ok": False, "error": str(error), "type": type(error).__name__},
            }
        )
        return ix.attach_answer_activity(
            {
                "ok": False,
                "reason": "answer_save_failed",
                "message": str(error),
                "adapter": runtime_config.answer_adapter,
                "model": runtime_config.ollama_model,
                "focus_node_id": package.selected_node_id,
                "process_run_id": package.process_run_id,
                "process_trace": package.process_trace,
            }
        )
    completed_step = "deep_retrieval" if package.pre_built_answer else "answer_adapter"
    ix.finish_answer_process_run(
        db,
        package.process_run_id,
        status="completed",
        current_step_id="answer_saved",
        completed_step_id=completed_step,
        exchange_id=exchange_id,
    )
    result = {
        "ok": True,
        "exchange_id": exchange_id,
        "session_id": package.session_id,
        "project_domain_id": clean_domain_id(package.project_domain_id),
        "conversation_domain_id": clean_domain_id(
            package.conversation_domain_id,
            fallback=conversation_domain_id_for_session(package.session_id),
        ),
        "focus_node_id": package.selected_node_id,
        "query": package.query,
        "answer": answer["answer"],
        "adapter": answer["adapter"],
        "model": answer.get("model"),
        "used_node_ids": answer["used_node_ids"],
        "budget": prompt.get("budget", {}),
        "semantic": prompt.get("semantic", []),
        "semantic_summary": prompt.get("semantic_summary", ""),
        "semantic_status": prompt.get("semantic_status", "disabled"),
        "semantic_diagnostic": prompt.get("semantic_diagnostic"),
        "retrieval_status": package.retrieval_status,
        "controller_decision": package.controller_decision,
        "process_run_id": package.process_run_id,
        "process_trace": package.process_trace,
        "phase": "synthesis",
    }
    return ix.attach_answer_activity(result)


def _retrieval_failed(
    db: Database,
    package: AnswerRetrievalPackage,
    reason: str,
    step_name: str,
    error: Exception,
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    ix.finish_answer_process_run(
        db,
        package.process_run_id,
        status="blocked",
        current_step_id=reason,
        exception=ix.answer_exception_payload(reason, "Inspect retrieval and retry.", error),
    )
    package.process_trace.append(
        {
            "step": step_name,
            "input": {"query": package.query, "session_id": package.session_id},
            "output": {"ok": False, "error": str(error)},
        }
    )
    return ix.attach_answer_activity(
        {
            "ok": False,
            "reason": reason,
            "message": str(error),
            "adapter": runtime_config.answer_adapter,
            "model": runtime_config.ollama_model,
            "focus_node_id": package.selected_node_id,
            "process_run_id": package.process_run_id,
            "process_trace": package.process_trace,
            "phase": "retrieval",
        }
    )
