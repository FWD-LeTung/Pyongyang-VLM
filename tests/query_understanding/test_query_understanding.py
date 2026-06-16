"""Tests for query understanding guardrails and parser orchestration."""

from __future__ import annotations

from src.query_understanding.llm_parser import QueryParser
from src.query_understanding.prompts import render_system_prompt
from src.query_understanding.schema import (
    HybridFilterPayload,
    Metadata,
    QueryUnderstandingResponse,
    VectorSearchPayload,
)
from src.query_understanding.validator import QueryValidator


def test_validator_rejects_garbage_inputs() -> None:
    """Garbage inputs should be rejected before any LLM call."""

    validator = QueryValidator()

    assert validator.validate_and_detect("12345") == {
        "is_valid": False,
        "language": "unknown",
        "error_code": "QUERY_ONLY_NUMBERS",
    }
    assert validator.validate_and_detect("!!!!!") == {
        "is_valid": False,
        "language": "unknown",
        "error_code": "QUERY_ONLY_SPECIAL_CHARS",
    }


def test_validator_detects_vietnamese_and_english() -> None:
    """The heuristic router should distinguish common VI and EN descriptions."""

    validator = QueryValidator()

    assert validator.validate_and_detect("tìm người đội nón")["language"] == "vi"
    assert validator.validate_and_detect("A woman in a red dress")["language"] == "en"


def test_schema_defaults_unknown_filters() -> None:
    """Hybrid filters default to unknown when attributes are missing."""

    filters = HybridFilterPayload(upper_color="")

    assert filters.gender == "unknown"
    assert filters.upper_color == "unknown"
    assert filters.accessory == "unknown"


def test_render_system_prompt_escapes_user_xml() -> None:
    """User text should not break XML prompt structure."""

    prompt = render_system_prompt(language="vi", raw_query="tìm người <áo đỏ>")

    assert '<user_input language="vi">' in prompt
    assert "tìm người &lt;áo đỏ&gt;" in prompt


def test_parser_returns_error_schema_for_invalid_query() -> None:
    """Invalid queries should return the graceful error response."""

    parser = QueryParser(client=object())
    response = parser.parse("12345")

    assert response.metadata.status == "error"
    assert response.metadata.error_code == "QUERY_ONLY_NUMBERS"
    assert response.vector_search_payload.normalized_text == ""
    assert response.hybrid_filter_payload.gender == "unknown"
    assert response.generation_source == "none"


def test_parser_uses_fallback_when_primary_fails() -> None:
    """Primary API errors should route to the configured fallback model."""

    class StubParser(QueryParser):
        def __init__(self) -> None:
            super().__init__(client=object())
            self.calls: list[str] = []

        def _parse_with_model(self, model: str, prompt: str) -> QueryUnderstandingResponse:
            self.calls.append(model)
            if model == self.primary_model:
                raise RuntimeError("primary unavailable")
            return QueryUnderstandingResponse(
                metadata=Metadata(
                    original_query="model echo",
                    language_detected="model language",
                    status="success",
                    error_code="stale",
                ),
                vector_search_payload=VectorSearchPayload(
                    normalized_text="The woman is wearing a red dress."
                ),
                hybrid_filter_payload=HybridFilterPayload(
                    gender="female",
                    upper_color="red",
                    upper_type="dress",
                ),
            )

    parser = StubParser()
    response = parser.parse("A woman in a red dress")

    assert parser.calls == [parser.primary_model, QueryParser.FALLBACK_MODEL]
    assert response.metadata.original_query == "A woman in a red dress"
    assert response.metadata.language_detected == "en"
    assert response.metadata.error_code is None
    assert response.hybrid_filter_payload.gender == "female"
    assert response.generation_source == "local_fallback"


def test_parser_uses_google_genai_client_schema() -> None:
    """The cloud route should call google-genai with the Pydantic schema."""

    class FakeModels:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def generate_content(self, **kwargs: object) -> object:
            self.kwargs = kwargs

            class FakeResponse:
                text = """{
                    "metadata": {
                        "original_query": "",
                        "language_detected": "en",
                        "status": "success",
                        "error_code": null
                    },
                    "vector_search_payload": {
                        "normalized_text": "The man is wearing a black shirt."
                    },
                    "hybrid_filter_payload": {
                        "gender": "male",
                        "upper_color": "black",
                        "upper_type": "shirt"
                    }
                }"""

            return FakeResponse()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    client = FakeClient()
    parser = QueryParser(client=client)
    response = parser.parse("A man wearing a black shirt")

    assert client.models.kwargs["model"] == parser.primary_model
    assert client.models.kwargs["contents"]
    assert client.models.kwargs["config"] == {
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": QueryUnderstandingResponse,
    }
    assert response.generation_source == "vertex_gemini"
    assert response.metadata.original_query == "A man wearing a black shirt"
