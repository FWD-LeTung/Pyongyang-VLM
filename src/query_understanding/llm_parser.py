"""LLM orchestration for pedestrian query understanding."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from src.query_understanding.prompts import render_system_prompt
from src.query_understanding.schema import (
    HybridFilterPayload,
    Metadata,
    QueryUnderstandingResponse,
    VectorSearchPayload,
)
from src.query_understanding.validator import QueryValidator
from src.utils.logger import setup_logger
from src.utils.timer import time_it


load_dotenv()
logger = setup_logger(__name__)


class QueryParser:
    """Google Gen AI cloud parser with a local edge-model fallback."""

    PRIMARY_MODEL = "gemini-2.5-flash"
    FALLBACK_MODEL = "local_fallback"
    DEFAULT_LOCAL_MODEL_NAME = "llama3.1:8b-instruct"

    def __init__(
        self,
        *,
        client: object | None = None,
        validator: QueryValidator | None = None,
        initialize_vertex: bool | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        """Initialize parser configuration from environment variables.

        Args:
            client: Optional test seam or externally managed client. Supplying
                this skips eager Vertex initialization unless explicitly
                overridden.
            validator: Optional validator instance.
            initialize_vertex: Backward-compatible flag controlling eager
                Google Gen AI client initialization. Defaults to ``True`` unless
                ``client`` is passed.
            request_timeout: HTTP timeout in seconds for the local fallback.
        """

        load_dotenv()

        self.client = client
        self.validator = validator or QueryValidator()
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID")
        self.gcp_location = os.getenv("GCP_LOCATION", "us-central1")
        self.gemini_model = os.getenv("GEMINI_MODEL", self.PRIMARY_MODEL)
        self.local_model_url = os.getenv(
            "LOCAL_MODEL_URL",
            "http://localhost:11434/v1/chat/completions",
        )
        self.local_model_name = os.getenv(
            "LOCAL_MODEL_NAME",
            self.DEFAULT_LOCAL_MODEL_NAME,
        )
        self.request_timeout = request_timeout
        self.primary_model = self.gemini_model
        self.fallback_model = self.FALLBACK_MODEL
        self._genai_init_error: Exception | None = None

        if initialize_vertex is None:
            initialize_vertex = client is None
        if initialize_vertex:
            self._initialize_genai_client()

    @time_it
    def parse(self, raw_query: str) -> QueryUnderstandingResponse:
        """Validate, normalize, and parse a raw pedestrian search query."""

        query = (raw_query or "").strip()
        validation = self.validator.validate_and_detect(query)
        language = str(validation["language"])

        if not validation["is_valid"]:
            return self._failure_response(
                original_query=query,
                language=language,
                error_code=str(validation["error_code"]),
                status="error",
            )

        prompt = render_system_prompt(language=language, raw_query=query)

        try:
            parsed = self._parse_with_model(self.primary_model, prompt)
            parsed = self._attach_runtime_metadata(
                parsed,
                original_query=query,
                language=language,
                generation_source="vertex_gemini",
            )
            logger.info("Query understanding completed with Google Gen AI on Vertex AI.")
            return parsed
        except Exception:
            logger.warning("Vertex AI failed. Falling back to local model at Edge...")

        try:
            parsed = self._parse_with_model(self.fallback_model, prompt)
            return self._attach_runtime_metadata(
                parsed,
                original_query=query,
                language=language,
                generation_source="local_fallback",
            )
        except Exception:
            logger.error("Both Cloud and Edge LLMs failed.")
            return self._failure_response(
                original_query=query,
                language=language,
                error_code="LLM_API_ERROR",
                status="error",
            )

    def _initialize_genai_client(self) -> None:
        """Initialize Google Gen AI SDK in Vertex AI mode using ADC."""

        if self.client is not None:
            self._genai_init_error = None
            return
        try:
            from google import genai

            self.client = genai.Client(
                vertexai=True,
                project=self.gcp_project_id,
                location=self.gcp_location,
            )
            self._genai_init_error = None
        except Exception as exc:
            self._genai_init_error = exc

    def _parse_with_model(
        self,
        model: str,
        prompt: str,
    ) -> QueryUnderstandingResponse:
        """Route a prompt to the primary cloud model or local fallback."""

        if model == self.primary_model:
            return self._parse_with_genai(prompt)
        return self._parse_with_local(prompt)

    def _parse_with_genai(self, prompt: str) -> QueryUnderstandingResponse:
        """Call Google Gen AI on Vertex AI with schema-constrained JSON output."""

        if self.client is None:
            self._initialize_genai_client()
        if self._genai_init_error is not None or self.client is None:
            raise RuntimeError("Google Gen AI client is not initialized.") from self._genai_init_error

        response = self.client.models.generate_content(
            model=self.primary_model,
            contents=prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "response_schema": QueryUnderstandingResponse,
            },
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, QueryUnderstandingResponse):
            return parsed
        if isinstance(parsed, dict):
            return self._coerce_response(parsed)
        return self._coerce_response(self._extract_genai_text(response))

    def _parse_with_local(self, prompt: str) -> QueryUnderstandingResponse:
        """Call an OpenAI-compatible local endpoint such as Ollama or vLLM."""

        import requests

        payload = {
            "model": self.local_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            self.local_model_url,
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return self._coerce_response(self._extract_local_content(response.json()))

    def _attach_runtime_metadata(
        self,
        response: QueryUnderstandingResponse,
        *,
        original_query: str,
        language: str,
        generation_source: str,
    ) -> QueryUnderstandingResponse:
        """Trust deterministic runtime metadata over model-generated echoes."""

        response.metadata.original_query = original_query
        response.metadata.language_detected = language
        response.generation_source = generation_source

        if response.metadata.status == "success":
            response.metadata.error_code = None
        elif response.metadata.status == "rejected" and not response.metadata.error_code:
            response.metadata.error_code = "OUT_OF_DOMAIN"

        return QueryUnderstandingResponse.model_validate(response.model_dump())

    @staticmethod
    def _failure_response(
        *,
        original_query: str,
        language: str,
        error_code: str,
        status: str,
    ) -> QueryUnderstandingResponse:
        """Build a graceful response for validation or LLM failures."""

        return QueryUnderstandingResponse(
            metadata=Metadata(
                original_query=original_query,
                language_detected=language,
                status=status,
                error_code=error_code,
            ),
            vector_search_payload=VectorSearchPayload(normalized_text=""),
            hybrid_filter_payload=HybridFilterPayload(),
            generation_source="none",
        )

    @classmethod
    def _coerce_response(cls, content: str | dict[str, Any]) -> QueryUnderstandingResponse:
        """Parse JSON content into the typed root response model."""

        if isinstance(content, dict):
            return QueryUnderstandingResponse.model_validate(content)

        text = cls._strip_markdown_fence(content)
        try:
            return QueryUnderstandingResponse.model_validate_json(text)
        except (ValidationError, ValueError):
            json_payload = cls._extract_json_object(text)
            return QueryUnderstandingResponse.model_validate(json_payload)

    @staticmethod
    def _extract_genai_text(response: Any) -> str:
        """Extract text from Google Gen AI response variants."""

        text = getattr(response, "text", None)
        if text:
            return str(text)

        try:
            return str(response.candidates[0].content.parts[0].text)
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Google Gen AI response did not contain text content.") from exc

    @staticmethod
    def _extract_local_content(payload: dict[str, Any]) -> str | dict[str, Any]:
        """Extract assistant content from OpenAI-compatible chat completions."""

        if "choices" in payload:
            message = payload["choices"][0].get("message", {})
            content = message.get("content")
            if content is not None:
                return content

        if "message" in payload and isinstance(payload["message"], dict):
            content = payload["message"].get("content")
            if content is not None:
                return content

        if "response" in payload:
            return payload["response"]

        raise ValueError("Local model response did not contain JSON content.")

    @staticmethod
    def _strip_markdown_fence(content: str) -> str:
        """Remove accidental fenced-code wrappers from local model output."""

        text = content.strip()
        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        """Recover the first complete JSON object from noisy model output."""

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output.")
        return json.loads(text[start : end + 1])
