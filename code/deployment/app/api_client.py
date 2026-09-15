"""HTTP client used by Streamlit to call the routing API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import requests


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 15


class ApiClientError(RuntimeError):
    """A user-facing error raised when the API cannot serve a request."""


@dataclass(frozen=True)
class ToolCandidate:
    tool: str
    score: float


@dataclass(frozen=True)
class Prediction:
    decision: Literal["route", "fallback"]
    tool: str | None
    score: float
    top_k: list[ToolCandidate]
    model_version: str


@dataclass(frozen=True)
class ModelInfo:
    model_version: str
    base_model: str
    dataset_version: str
    fallback_threshold: float
    tool_count: int


class ApiClient:
    """Small client that keeps the Streamlit app independent from model code."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_BASE_URL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def health(self) -> dict[str, object]:
        return self._request("GET", "/health")

    def model_info(self) -> ModelInfo:
        payload = self._request("GET", "/model-info")
        try:
            return ModelInfo(
                model_version=str(payload["model_version"]),
                base_model=str(payload["base_model"]),
                dataset_version=str(payload["dataset_version"]),
                fallback_threshold=float(payload["fallback_threshold"]),
                tool_count=int(payload["tool_count"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ApiClientError("FastAPI returned an invalid model-info response") from error

    def predict(self, text: str, top_k: int) -> Prediction:
        payload = self._request(
            "POST",
            "/predict",
            json_body={"text": text, "top_k": top_k},
        )
        try:
            candidates = payload["top_k"]
            if not isinstance(candidates, list):
                raise TypeError("top_k must be a list")
            return Prediction(
                decision=payload["decision"],
                tool=payload["tool"],
                score=float(payload["score"]),
                top_k=[
                    ToolCandidate(tool=str(item["tool"]), score=float(item["score"]))
                    for item in candidates
                ],
                model_version=str(payload["model_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ApiClientError("FastAPI returned an invalid prediction response") from error

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        try:
            response = requests.request(
                method,
                f"{self._base_url}{path}",
                json=json_body,
                timeout=self._timeout_seconds,
            )
        except requests.RequestException as error:
            raise ApiClientError("FastAPI is unavailable") from error

        try:
            payload = response.json()
        except ValueError as error:
            raise ApiClientError("FastAPI returned an invalid response") from error
        if not isinstance(payload, dict):
            raise ApiClientError("FastAPI returned an invalid response")
        if not response.ok:
            detail = payload.get("detail", "FastAPI returned an error")
            raise ApiClientError(str(detail))
        return payload
