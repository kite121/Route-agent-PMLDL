"""HTTP contracts for the routing API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_TOP_K = 15


class ApiSchema(BaseModel):
    """Reject fields that are not part of the public API contract."""

    model_config = ConfigDict(extra="forbid")


class PredictRequest(ApiSchema):
    """A user request and the number of alternative tools to return."""

    text: str = Field(description="Non-empty user request")
    top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized


class ToolCandidate(ApiSchema):
    """One candidate tool ranked by cosine similarity."""

    tool: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)


class PredictResponse(ApiSchema):
    """The route/fallback decision produced by the packaged model."""

    decision: Literal["route", "fallback"]
    tool: str | None = None
    score: float = Field(ge=-1.0, le=1.0)
    top_k: list[ToolCandidate] = Field(min_length=1)
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self) -> "PredictResponse":
        if self.decision == "route":
            if self.tool is None:
                raise ValueError("route decision requires a selected tool")
            if self.top_k[0].tool != self.tool:
                raise ValueError("selected tool must be the first top_k candidate")
        elif self.tool is not None:
            raise ValueError("fallback decision must not select a tool")
        return self


class HealthResponse(ApiSchema):
    """Liveness and successful model-loading state."""

    status: Literal["ok"]
    model_loaded: bool
    model_version: str = Field(min_length=1)


class ModelInfoResponse(ApiSchema):
    """Non-sensitive metadata of the model currently served by the API."""

    model_version: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    fallback_threshold: float = Field(ge=-1.0, le=1.0)
    tool_count: int = Field(ge=1)


class ErrorResponse(ApiSchema):
    """The common error body returned by the API."""

    detail: str = Field(min_length=1)
