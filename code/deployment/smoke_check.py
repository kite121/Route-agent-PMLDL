"""Verify the Docker-deployed API serves the model package mounted by Compose.

Run after ``docker compose -f code/deployment/docker-compose.yml up --build -d``:
    .venv/bin/python code/deployment/smoke_check.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "current"
REQUEST_TIMEOUT_SECONDS = 5


def _read_expected_model_version(model_dir: Path) -> str:
    manifest_path = model_dir / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read model manifest: {manifest_path}") from error
    version = manifest.get("model_version") if isinstance(manifest, dict) else None
    if not isinstance(version, str) or not version:
        raise RuntimeError("Model manifest has no model_version")
    return version


def _request_json(
    api_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"{api_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode()
    except HTTPError as error:
        raise RuntimeError(f"{method} {path} returned HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"{method} {path} could not reach the API") from error
    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{method} {path} returned invalid JSON") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"{method} {path} returned a non-object JSON value")
    return result


def _wait_for_health(api_url: str, expected_version: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = _request_json(api_url, "/health")
            if health.get("status") != "ok" or health.get("model_loaded") is not True:
                raise RuntimeError("API health response does not confirm a loaded model")
            if health.get("model_version") != expected_version:
                raise RuntimeError("API health response has an unexpected model version")
            return
        except RuntimeError as error:
            last_error = error
            time.sleep(2)
    raise RuntimeError(
        f"API did not become healthy within {timeout_seconds} seconds: {last_error}"
    )


def _check_prediction(
    api_url: str,
    *,
    text: str,
    expected_decision: str,
    expected_tool: str | None,
    expected_version: str,
) -> None:
    prediction = _request_json(
        api_url,
        "/predict",
        method="POST",
        payload={"text": text, "top_k": 3},
    )
    if prediction.get("decision") != expected_decision:
        raise RuntimeError(f"Unexpected decision for {text!r}: {prediction.get('decision')!r}")
    if prediction.get("tool") != expected_tool:
        raise RuntimeError(f"Unexpected tool for {text!r}: {prediction.get('tool')!r}")
    if prediction.get("model_version") != expected_version:
        raise RuntimeError("Prediction was served by an unexpected model version")
    candidates = prediction.get("top_k")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Prediction has no top_k candidates")


def _check_tool_registry(api_url: str, expected_version: str) -> None:
    registry = _request_json(api_url, "/tools")
    if registry.get("model_version") != expected_version:
        raise RuntimeError("Tools response has an unexpected model version")
    tools = registry.get("tools")
    if not isinstance(tools, list) or not tools:
        raise RuntimeError("Tools response has no registered tools")
    alarm = next((tool for tool in tools if tool.get("name") == "alarm_set"), None)
    if not isinstance(alarm, dict) or not isinstance(alarm.get("description"), str):
        raise RuntimeError("Tools response has an invalid alarm_set definition")


def _check_allowed_tool_filter(api_url: str) -> None:
    prediction = _request_json(
        api_url,
        "/predict",
        method="POST",
        payload={
            "text": "Set an alarm for seven tomorrow morning",
            "top_k": 3,
            "allowed_tools": ["weather_query"],
        },
    )
    candidates = prediction.get("top_k")
    if not isinstance(candidates, list) or [item.get("tool") for item in candidates] != [
        "weather_query"
    ]:
        raise RuntimeError("Prediction did not restrict candidates to allowed_tools")
    if prediction.get("tool") not in {None, "weather_query"}:
        raise RuntimeError("Prediction selected a tool outside allowed_tools")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check the Docker-deployed router API")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--startup-timeout-seconds", type=int, default=90)
    arguments = parser.parse_args()

    if arguments.startup_timeout_seconds <= 0:
        parser.error("--startup-timeout-seconds must be positive")

    expected_version = _read_expected_model_version(arguments.model_dir)
    _wait_for_health(
        arguments.api_url,
        expected_version,
        arguments.startup_timeout_seconds,
    )
    info = _request_json(arguments.api_url, "/model-info")
    if info.get("model_version") != expected_version:
        raise RuntimeError("Model-info response has an unexpected model version")
    _check_tool_registry(arguments.api_url, expected_version)
    _check_prediction(
        arguments.api_url,
        text="Set an alarm for seven tomorrow morning",
        expected_decision="route",
        expected_tool="alarm_set",
        expected_version=expected_version,
    )
    _check_prediction(
        arguments.api_url,
        text="Tell me a funny joke about cats",
        expected_decision="fallback",
        expected_tool=None,
        expected_version=expected_version,
    )
    _check_allowed_tool_filter(arguments.api_url)
    print(f"Smoke check passed for model {expected_version} at {arguments.api_url}")


if __name__ == "__main__":
    main()
