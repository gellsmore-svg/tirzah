from __future__ import annotations

import hashlib
import json
import math
import select
import shlex
import subprocess
import struct
from typing import Any
from urllib import error, request

DEFAULT_EMBEDDING_DIMENSIONS = 16
MAX_MOCK_EMBEDDING_DIMENSIONS = 256
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
HTTP_BACKED_EMBEDDING_ADAPTERS = {"ollama_http", "ollama_powershell"}


class EmbeddingAdapterPolicyError(ValueError):
    """The configured embedding adapter is refused by policy (e.g. HTTP-backed),
    as distinct from a transient embedding failure."""


class MockEmbeddingAdapter:
    """Deterministic, dependency-free embedding adapter.

    Produces a bounded unit-norm vector derived from a SHA-256 expansion of the
    source text. The same text always yields the same vector, so stored
    embeddings are reproducible without any external model, GPU, or network
    call. This is the substrate landing point for real local embedding models;
    it intentionally does not change retrieval ranking.
    """

    name = "mock_embedding"
    model = "mock-deterministic-v1"

    def __init__(self, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = bounded_mock_dimensions(dimensions)

    def embed(self, text: str) -> dict[str, Any]:
        normalized = text or ""
        return {
            "adapter": self.name,
            "model": self.model,
            "dimensions": self.dimensions,
            "vector": deterministic_vector(normalized, self.dimensions),
            "source_text_hash": source_text_hash(normalized),
        }


class OllamaHttpEmbeddingAdapter:
    name = "ollama_http_embedding"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model = getattr(config, "embedding_model", None) or DEFAULT_OLLAMA_EMBEDDING_MODEL
        self.dimensions: int | None = None

    def embed(self, text: str) -> dict[str, Any]:
        normalized = text or ""
        vector = l2_normalize(self._embedding_vector(normalized))
        self.dimensions = len(vector)
        return {
            "adapter": self.name,
            "model": self.model,
            "dimensions": len(vector),
            "vector": vector,
            "source_text_hash": source_text_hash(normalized),
        }

    def _embedding_vector(self, text: str) -> list[float]:
        endpoint = f"{str(self.config.ollama_base_url).rstrip('/')}/api/embed"
        payload = json.dumps(
            {
                "model": self.model,
                "input": text,
            }
        ).encode("utf-8")
        req = request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.ollama_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exception:
            raise TimeoutError(
                f"Ollama embedding request timed out after "
                f"{self.config.ollama_timeout_seconds}s while running {self.model}."
            ) from exception
        except error.URLError as exception:
            raise RuntimeError(
                f"Ollama embedding request failed for {self.model} at {endpoint}: "
                f"{exception.reason}"
            ) from exception
        except json.JSONDecodeError as exception:
            raise RuntimeError(
                f"Ollama embedding request returned invalid JSON for {self.model}."
            ) from exception
        vector = ollama_embedding_vector(data)
        if not vector:
            raise RuntimeError(f"Ollama embedding request returned no vector for {self.model}.")
        return vector


class OllamaPowerShellEmbeddingAdapter:
    name = "ollama_powershell_embedding"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model = getattr(config, "embedding_model", None) or DEFAULT_OLLAMA_EMBEDDING_MODEL
        self.dimensions: int | None = None

    def embed(self, text: str) -> dict[str, Any]:
        normalized = text or ""
        vector = l2_normalize(self._embedding_vector(normalized))
        self.dimensions = len(vector)
        return {
            "adapter": self.name,
            "model": self.model,
            "dimensions": len(vector),
            "vector": vector,
            "source_text_hash": source_text_hash(normalized),
        }

    def _embedding_vector(self, text: str) -> list[float]:
        endpoint = f"{str(self.config.ollama_base_url).rstrip('/')}/api/embed"
        payload = json.dumps({"model": self.model, "input": text})
        escaped_endpoint = endpoint.replace("'", "''")
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "$ErrorActionPreference = 'Stop'; "
                "$payload = [Console]::In.ReadToEnd(); "
                f"$response = Invoke-RestMethod -Uri '{escaped_endpoint}' "
                "-Method Post -ContentType 'application/json' "
                "-Body $payload; "
                "$response | "
                "ConvertTo-Json -Depth 5 -Compress"
            ),
        ]
        try:
            completed = subprocess.run(
                command,
                input=payload,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.ollama_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exception:
            raise TimeoutError(
                f"Ollama PowerShell embedding request timed out after "
                f"{self.config.ollama_timeout_seconds}s while running {self.model}."
            ) from exception
        if completed.returncode != 0 or not completed.stdout.strip():
            detail = "\n".join(part for part in [completed.stderr, completed.stdout] if part)
            raise RuntimeError(
                f"Ollama PowerShell embedding request failed for {self.model} "
                f"at {endpoint}: {detail.strip() or completed.returncode}"
            )
        try:
            data = json.loads(completed.stdout.lstrip("\ufeff").strip())
        except json.JSONDecodeError as exception:
            raise RuntimeError(
                f"Ollama PowerShell embedding request returned invalid JSON for {self.model}."
            ) from exception
        vector = ollama_embedding_vector(data)
        if not vector:
            raise RuntimeError(
                f"Ollama PowerShell embedding request returned no vector for {self.model}."
            )
        return vector


class LocalCommandEmbeddingAdapter:
    name = "local_command_profile"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model = getattr(config, "embedding_model", None) or "local-profile-model"
        self.command = local_profile_command(getattr(config, "profile_command", None))
        self.command_mode = getattr(config, "profile_command_mode", None) or "single"
        self.dimensions: int | None = None
        self._worker: subprocess.Popen[str] | None = None

    def embed(self, text: str) -> dict[str, Any]:
        normalized = text or ""
        vector = l2_normalize(self._profile_vector(normalized))
        self.dimensions = len(vector)
        return {
            "adapter": self.name,
            "model": self.model,
            "dimensions": len(vector),
            "vector": vector,
            "source_text_hash": source_text_hash(normalized),
        }

    def _profile_vector(self, text: str) -> list[float]:
        if not self.command:
            raise RuntimeError(
                "Local command profile adapter requires runtime.profile_command. "
                "Configure a local executable that reads JSON from stdin and returns a vector JSON object."
            )
        if self.command_mode == "worker":
            return self._worker_profile_vector(text)
        payload = json.dumps({"model": self.model, "text": text})
        try:
            completed = subprocess.run(
                self.command,
                input=payload,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.ollama_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exception:
            raise TimeoutError(
                f"Local command profile adapter timed out after "
                f"{self.config.ollama_timeout_seconds}s while running {self.model}."
            ) from exception
        if completed.returncode != 0 or not completed.stdout.strip():
            detail = "\n".join(part for part in [completed.stderr, completed.stdout] if part)
            raise RuntimeError(
                f"Local command profile adapter failed for {self.model}: "
                f"{detail.strip() or completed.returncode}"
            )
        try:
            data = json.loads(completed.stdout.lstrip("\ufeff").strip())
        except json.JSONDecodeError as exception:
            raise RuntimeError(
                f"Local command profile adapter returned invalid JSON for {self.model}."
            ) from exception
        vector = local_command_profile_vector(data)
        if not vector:
            raise RuntimeError(
                f"Local command profile adapter returned no usable vector for {self.model}."
            )
        return vector

    def _worker_profile_vector(self, text: str) -> list[float]:
        worker = self._worker_process()
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("Local command profile worker was not opened with stdin/stdout.")
        payload = json.dumps({"model": self.model, "text": text})
        try:
            worker.stdin.write(payload)
            worker.stdin.write("\n")
            worker.stdin.flush()
        except BrokenPipeError as exception:
            self.close()
            raise RuntimeError(
                f"Local command profile worker stopped while running {self.model}."
            ) from exception
        timeout = getattr(self.config, "ollama_timeout_seconds", 180)
        ready, _, _ = select.select([worker.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise TimeoutError(
                f"Local command profile worker timed out after {timeout}s while running {self.model}."
            )
        line = worker.stdout.readline()
        if not line:
            stderr = worker.stderr.read() if worker.stderr else ""
            self.close()
            raise RuntimeError(
                f"Local command profile worker exited while running {self.model}: "
                f"{stderr.strip() or worker.poll()}"
            )
        try:
            data = json.loads(line.lstrip("\ufeff").strip())
        except json.JSONDecodeError as exception:
            raise RuntimeError(
                f"Local command profile worker returned invalid JSON for {self.model}."
            ) from exception
        if isinstance(data.get("error"), dict):
            error_payload = data["error"]
            detail = error_payload.get("message") or error_payload.get("code") or data["error"]
            raise RuntimeError(f"Local command profile worker failed for {self.model}: {detail}")
        vector = local_command_profile_vector(data)
        if not vector:
            raise RuntimeError(
                f"Local command profile worker returned no usable vector for {self.model}."
            )
        return vector

    def _worker_process(self) -> subprocess.Popen[str]:
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        self._worker = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._worker

    def close(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        if worker.poll() is None:
            if worker.stdin:
                try:
                    worker.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=2)

    def __del__(self) -> None:
        self.close()


class HoglahEmbeddingAdapter:
    """Route embeddings through a Hoglah queue daemon (decoupled topology).

    Operator-sanctioned HTTP-backed path: from Tirzah's process this is local
    IPC (submit to a local SQLite queue, await a local result file / localhost
    callback); the separate `hoglah run --real` daemon performs the Ollama HTTP
    call. Returns the same shape as the sibling embedding adapters, L2-normalized
    for ranking parity.
    """

    name = "hoglah"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model = getattr(config, "embedding_model", None) or DEFAULT_OLLAMA_EMBEDDING_MODEL
        self.dimensions: int | None = None
        self._runner: Any | None = None

    def _get_runner(self):
        if self._runner is None:
            from tirzah.adapters.hoglah_runtime import HoglahJobRunner

            self._runner = HoglahJobRunner(self.config)
        return self._runner

    def embed(self, text: str) -> dict[str, Any]:
        normalized = text or ""
        result = self._get_runner().run_embedding(normalized, model=self.model)
        status = result.get("status")
        job_id = result.get("job_id")
        if status != "completed":
            raise RuntimeError(
                result.get("error") or f"Hoglah embedding job {job_id} ended with status {status}."
            )
        vector = l2_normalize(parse_embedding_vector(result.get("embedding")))
        if not vector:
            raise RuntimeError(f"Hoglah embedding job {job_id} returned no usable vector.")
        self.dimensions = len(vector)
        return {
            "adapter": self.name,
            "model": result.get("model") or self.model,
            "dimensions": len(vector),
            "vector": vector,
            "source_text_hash": source_text_hash(normalized),
        }

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None


def embedding_adapter(
    config: Any | None = None,
) -> (
    MockEmbeddingAdapter
    | OllamaHttpEmbeddingAdapter
    | OllamaPowerShellEmbeddingAdapter
    | LocalCommandEmbeddingAdapter
    | HoglahEmbeddingAdapter
):
    if config is None:
        return default_embedding_adapter()
    name = getattr(config, "embedding_adapter", "mock")
    dimensions = getattr(config, "embedding_dimensions", DEFAULT_EMBEDDING_DIMENSIONS)
    if name in HTTP_BACKED_EMBEDDING_ADAPTERS and not getattr(
        config,
        "allow_http_ingestion_adapters",
        False,
    ):
        raise EmbeddingAdapterPolicyError(
            f"Embedding adapter '{name}' is HTTP-backed and is not allowed for "
            "ingestion or retrieval memory operations. Use a local non-HTTP "
            "embedding adapter, or set allow_http_ingestion_adapters only for "
            "temporary diagnostics."
        )
    if name == "mock":
        return MockEmbeddingAdapter(dimensions=dimensions)
    if name == "local_command":
        if not local_profile_command(getattr(config, "profile_command", None)):
            raise ValueError(
                "Local command profile adapter requires runtime.profile_command. "
                "Configure a local executable that reads JSON from stdin and returns a vector JSON object."
            )
        return LocalCommandEmbeddingAdapter(config)
    if name == "ollama_http":
        return OllamaHttpEmbeddingAdapter(config)
    if name == "ollama_powershell":
        return OllamaPowerShellEmbeddingAdapter(config)
    if name == "hoglah":
        # Permitted for memory ops (operator-sanctioned): Tirzah only does local
        # IPC; the Hoglah daemon performs the Ollama HTTP call. Hence "hoglah" is
        # intentionally NOT in HTTP_BACKED_EMBEDDING_ADAPTERS.
        return HoglahEmbeddingAdapter(config)
    raise ValueError(f"Unknown embedding adapter: {name}")


def local_profile_command(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value if str(part)]
    return []


def deterministic_vector(text: str, dimensions: int) -> list[float]:
    encoded = text.encode("utf-8")
    raw: list[float] = []
    counter = 0
    while len(raw) < dimensions:
        digest = hashlib.sha256(encoded + counter.to_bytes(4, "big")).digest()
        for index in range(0, len(digest), 4):
            if len(raw) >= dimensions:
                break
            (value,) = struct.unpack(">I", digest[index : index + 4])
            raw.append(value / 0xFFFFFFFF * 2.0 - 1.0)
        counter += 1
    return l2_normalize(raw[:dimensions])


def l2_normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return [0.0 for _ in vector]
    return [round(value / magnitude, 6) for value in vector]


def ollama_embedding_vector(data: dict[str, Any]) -> list[float]:
    if isinstance(data.get("embedding"), list):
        return parse_embedding_vector(data["embedding"])
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        return parse_embedding_vector(embeddings[0])
    return []


def local_command_profile_vector(data: dict[str, Any]) -> list[float]:
    if isinstance(data.get("vector"), list):
        return parse_embedding_vector(data["vector"])
    return ollama_embedding_vector(data)


def parse_embedding_vector(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return []
    if not vector or not all(math.isfinite(item) for item in vector):
        return []
    return vector


def source_text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def bounded_mock_dimensions(value: Any, default: int = DEFAULT_EMBEDDING_DIMENSIONS) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, MAX_MOCK_EMBEDDING_DIMENSIONS))


_DEFAULT_ADAPTER = MockEmbeddingAdapter()


def default_embedding_adapter() -> MockEmbeddingAdapter:
    return _DEFAULT_ADAPTER
