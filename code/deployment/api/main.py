"""FastAPI service for the packaged semantic tool router."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from deployment.api.model_service import ModelService
from deployment.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


logger = logging.getLogger(__name__)
MODEL_UNAVAILABLE_DETAIL = "Packaged model is unavailable"
MODEL_UNAVAILABLE_RESPONSES = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the packaged model once without preventing diagnostic endpoints."""

    app.state.model_service = None
    try:
        app.state.model_service = ModelService.load()
    except Exception:
        logger.exception("Could not load the packaged routing model")
    yield
    app.state.model_service = None


app = FastAPI(
    title="Route Agent API",
    version="1.0.0",
    description="HTTP API for the packaged semantic tool router.",
    lifespan=lifespan,
)


def _model_service(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MODEL_UNAVAILABLE_DETAIL,
        )
    return service


@app.get(
    "/health",
    response_model=HealthResponse,
    responses=MODEL_UNAVAILABLE_RESPONSES,
)
def health(request: Request) -> HealthResponse:
    """Return success only when the packaged model is loaded."""

    return _model_service(request).health()


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses=MODEL_UNAVAILABLE_RESPONSES,
)
def model_info(request: Request) -> ModelInfoResponse:
    """Return the version and immutable metadata of the served model."""

    return _model_service(request).model_info()


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses=MODEL_UNAVAILABLE_RESPONSES,
)
def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    """Route one validated user request through the packaged model."""

    return _model_service(request).predict(payload)
