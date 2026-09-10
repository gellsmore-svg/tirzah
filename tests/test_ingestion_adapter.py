from types import SimpleNamespace

import pytest

from tirzah.adapters.ingestion import ingestion_adapter
from tirzah.config import RuntimeConfig
from tirzah.adapters.llm_ingestion import LlmIngestionAdapter
from tirzah.adapters.mock import MockIngestionAdapter


def test_ingestion_adapter_defaults_to_mock() -> None:
    assert isinstance(ingestion_adapter(), MockIngestionAdapter)


def test_ingestion_adapter_accepts_runtime_mock_selection() -> None:
    config = SimpleNamespace(ingestion_adapter="mock")

    assert isinstance(ingestion_adapter(config), MockIngestionAdapter)


def test_ingestion_adapter_rejects_unknown_adapter() -> None:
    config = SimpleNamespace(ingestion_adapter="llm_ingestion")

    with pytest.raises(ValueError, match="Unknown ingestion adapter: llm_ingestion"):
        ingestion_adapter(config)


def test_ingestion_adapter_accepts_llm_selection() -> None:
    config = RuntimeConfig(ingestion_adapter="llm")

    assert isinstance(ingestion_adapter(config), LlmIngestionAdapter)
