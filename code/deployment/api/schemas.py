"""HTTP contracts for the routing API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_TOP_K = 15


class ApiSchema(BaseModel):
    """Reject fields that are not part of the public API contract."""

    model_config = ConfigDict(extra="forbid")


class PredictRequest(ApiSchema):
    """A request, displayed-candidate count, and optional eligible tool subset."""

    text: str = Field(description="Non-empty user request")
    top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)
    allowed_tools: list[str] | None = Field(default=None, max_length=MAX_TOP_K)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
        return normalized

    @field_validator("allowed_tools")
    @classmethod
    def normalize_allowed_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("allowed_tools must contain at least one tool")
        normalized = [tool.strip() for tool in value]
        if any(not tool for tool in normalized):
            raise ValueError("allowed_tools must not contain blank names")
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_tools must not contain duplicates")
        return normalized


class ToolCandidate(ApiSchema):
    """One candidate tool ranked by cosine similarity."""

    tool: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)


class ToolDefinitionResponse(ApiSchema):
    """One tool stored in the registry of the currently served model."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    arguments: list[str]


class ToolsResponse(ApiSchema):
    """The immutable tool registry associated with the served model version."""

    model_version: str = Field(min_length=1)
    tools: list[ToolDefinitionResponse] = Field(min_length=1)


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
