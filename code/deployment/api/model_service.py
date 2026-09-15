"""Adapter between FastAPI and the packaged routing model."""

from __future__ import annotations

import os
from pathlib import Path

from models.inference import Router, load_router

from deployment.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ToolCandidate,
)


class ModelService:
    """A single loaded packaged model used by all API requests."""

    def __init__(self, router: Router) -> None:
        self._router = router

    @classmethod
    def load(cls) -> "ModelService":
        """Load and verify the packaged model once at API startup.

        Docker Compose sets ``MODEL_DIR`` to the read-only model volume and
        ``MODEL_CACHE_DIR`` to a writable extraction cache. Without them, the
        local ``models/current`` default remains convenient for development.
        """

        model_dir = os.environ.get("MODEL_DIR")
        cache_dir = os.environ.get("MODEL_CACHE_DIR")
        if model_dir:
            router = load_router(
                Path(model_dir),
                cache_dir=Path(cache_dir) if cache_dir else None,
            )
        else:
            router = load_router(cache_dir=Path(cache_dir)) if cache_dir else load_router()
        return cls(router=router)

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Route one API-validated user request."""

        result = self._router.route(request.text, top_k=request.top_k)
        return PredictResponse(
            decision=result.decision,
            tool=result.tool,
            score=result.score,
            top_k=[
                ToolCandidate(tool=item.tool, score=item.score)
                for item in result.top_k
            ],
            model_version=result.model_version,
        )

    def health(self) -> HealthResponse:
        """Return the liveness state of a successfully loaded service."""

        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=self._router.package.model_version,
        )

    def model_info(self) -> ModelInfoResponse:
        """Return non-sensitive metadata of the packaged model in service."""

        info = self._router.model_info()
        return ModelInfoResponse(
            model_version=info["model_version"],
            base_model=info["base_model"],
            dataset_version=info["dataset_version"],
            fallback_threshold=info["fallback_threshold"],
            tool_count=info["tool_count"],
        )
