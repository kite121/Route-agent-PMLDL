"""Accept or reject the latest evaluated routing encoder for deployment.

The decision is based only on the fixed test metrics already written by
``evaluate.py``. A rejected model remains available in ClearML, but is not
eligible for packaging or deployment.

Run from the repository root after a completed evaluation:
    .venv/bin/python code/models/quality_gate.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"
RUNS_DIR = PROJECT_ROOT / "models" / "runs"

CLEARML_PROJECT_NAME = "PMLDL/Route-agent"
ACCEPTED_TAG = "deployment-candidate"
REJECTED_TAG = "rejected-by-quality-gate"


@dataclass(frozen=True)
class QualityGateSettings:
    minimum_accuracy_at_1: float


@dataclass(frozen=True)
class EvaluationRun:
    dataset_version: str
    training_task_id: str
    evaluation_task_id: str
    evaluation_path: Path
    evaluation_sha256: str
    model_reference: str
    base_test_macro_f1: float
    fine_tuned_test_accuracy_at_1: float
    fine_tuned_test_macro_f1: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_project(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _load_settings() -> QualityGateSettings:
    try:
        config = yaml.safe_load(TRAINING_CONFIG_PATH.read_text())
        minimum_accuracy_at_1 = config["quality_gate"]["minimum_accuracy_at_1"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError(
            "training.yaml must define quality_gate.minimum_accuracy_at_1"
        ) from error
    if (
        isinstance(minimum_accuracy_at_1, bool)
        or not isinstance(minimum_accuracy_at_1, (int, float))
        or not 0.0 <= float(minimum_accuracy_at_1) <= 1.0
    ):
        raise ValueError(
            "quality_gate.minimum_accuracy_at_1 must be a number from 0 to 1"
        )
    return QualityGateSettings(
        minimum_accuracy_at_1=float(minimum_accuracy_at_1),
    )


def _finite_metric(
    payload: object,
    *,
    path: str,
) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ValueError(f"{path} must be a numeric metric")
    metric = float(payload)
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise ValueError(f"{path} must be a finite metric from 0 to 1")
    return metric


def _load_latest_evaluation() -> EvaluationRun:
    candidates: list[tuple[str, Path, dict[str, object]]] = []
    for evaluation_path in RUNS_DIR.glob("*/*/evaluation.json"):
        try:
            payload = json.loads(evaluation_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        created_at = payload.get("created_at")
        if isinstance(created_at, str):
            candidates.append((created_at, evaluation_path, payload))
    if not candidates:
        raise FileNotFoundError(
            "no evaluation report found; run code/models/evaluate.py first"
        )

    _, evaluation_path, payload = max(candidates, key=lambda candidate: candidate[0])
    if payload.get("schema_version") != 1:
        raise ValueError("evaluation report has an unsupported schema version")
    dataset_version = payload.get("dataset_version")
    training_task_id = payload.get("training_task_id")
    evaluation_task_id = payload.get("clearml_task_id")
    models = payload.get("models")
    if not all(
        isinstance(value, str) and value
        for value in (dataset_version, training_task_id, evaluation_task_id)
    ) or not isinstance(models, dict):
        raise ValueError("evaluation report is missing required provenance")

    base = models.get("base")
    fine_tuned = models.get("fine_tuned")
    if not isinstance(base, dict) or not isinstance(fine_tuned, dict):
        raise ValueError("evaluation report must contain base and fine_tuned results")
    base_test = base.get("test")
    fine_tuned_test = fine_tuned.get("test")
    model_reference = fine_tuned.get("model_reference")
    if (
        not isinstance(base_test, dict)
        or not isinstance(fine_tuned_test, dict)
        or not isinstance(model_reference, str)
        or not model_reference
    ):
        raise ValueError("evaluation report has incomplete test results")

    return EvaluationRun(
        dataset_version=dataset_version,
        training_task_id=training_task_id,
        evaluation_task_id=evaluation_task_id,
        evaluation_path=evaluation_path,
        evaluation_sha256=_sha256(evaluation_path),
        model_reference=model_reference,
        base_test_macro_f1=_finite_metric(
            base_test.get("macro_f1"),
            path="models.base.test.macro_f1",
        ),
        fine_tuned_test_accuracy_at_1=_finite_metric(
            fine_tuned_test.get("accuracy_at_1"),
            path="models.fine_tuned.test.accuracy_at_1",
        ),
        fine_tuned_test_macro_f1=_finite_metric(
            fine_tuned_test.get("macro_f1"),
            path="models.fine_tuned.test.macro_f1",
        ),
    )


def _decision(
    evaluation: EvaluationRun,
    settings: QualityGateSettings,
) -> tuple[bool, dict[str, dict[str, float | bool]]]:
    accuracy_passed = (
        evaluation.fine_tuned_test_accuracy_at_1 >= settings.minimum_accuracy_at_1
    )
    macro_f1_passed = (
        evaluation.fine_tuned_test_macro_f1 >= evaluation.base_test_macro_f1
    )
    checks: dict[str, dict[str, float | bool]] = {
        "minimum_accuracy_at_1": {
            "observed": evaluation.fine_tuned_test_accuracy_at_1,
            "minimum": settings.minimum_accuracy_at_1,
            "passed": accuracy_passed,
        },
        "macro_f1_not_below_base": {
            "fine_tuned": evaluation.fine_tuned_test_macro_f1,
            "base": evaluation.base_test_macro_f1,
            "passed": macro_f1_passed,
        },
    }
    return accuracy_passed and macro_f1_passed, checks


def _configure_clearml() -> None:
    load_dotenv(ENV_PATH, override=False)
    required = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"set {', '.join(missing)} in .env before quality gate")
    os.environ["CLEARML_LOG_ENVIRONMENT"] = ""


def _start_clearml_task(
    *,
    evaluation: EvaluationRun,
    settings: QualityGateSettings,
    accepted: bool,
    checks: dict[str, dict[str, float | bool]],
):
    _configure_clearml()
    from clearml import Task

    outcome = "accepted" if accepted else "rejected"
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name=f"quality-gate-{evaluation.dataset_version}",
        task_type=Task.TaskTypes.testing,
        tags=[evaluation.dataset_version, "quality-gate", outcome],
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
        auto_resource_monitoring=False,
    )
    task.connect(
        {
            "source": {
                "evaluation_task_id": evaluation.evaluation_task_id,
                "evaluation_sha256": evaluation.evaluation_sha256,
                "training_task_id": evaluation.training_task_id,
                "model_reference": evaluation.model_reference,
            },
            "quality_gate": {
                "minimum_accuracy_at_1": settings.minimum_accuracy_at_1,
                "checks": checks,
                "accepted": accepted,
            },
        },
        name="configuration",
    )
    return task


def _tag_training_task(training_task_id: str, accepted: bool) -> None:
    from clearml import Task

    training_task = Task.get_task(task_id=training_task_id)
    training_task.add_tags([ACCEPTED_TAG if accepted else REJECTED_TAG])


def main() -> None:
    settings = _load_settings()
    evaluation = _load_latest_evaluation()
    accepted, checks = _decision(evaluation, settings)
    task = None

    try:
        task = _start_clearml_task(
            evaluation=evaluation,
            settings=settings,
            accepted=accepted,
            checks=checks,
        )
        report = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "clearml_task_id": task.id,
            "dataset_version": evaluation.dataset_version,
            "decision": "accepted" if accepted else "rejected",
            "training_task_id": evaluation.training_task_id,
            "evaluation_task_id": evaluation.evaluation_task_id,
            "evaluation_sha256": evaluation.evaluation_sha256,
            "model_reference": evaluation.model_reference,
            "checks": checks,
        }
        report_path = evaluation.evaluation_path.parent / "quality_gate.json"
        _write_json(report_path, report)
        if not task.upload_artifact(
            name="quality_gate_report",
            artifact_object=str(report_path),
            wait_on_upload=True,
        ):
            raise RuntimeError("ClearML did not upload the quality gate report")
        _tag_training_task(evaluation.training_task_id, accepted)
    except Exception as error:
        if task is not None:
            task.mark_failed(status_reason=f"{type(error).__name__}: {error}")
        raise
    else:
        task.close()

    outcome = "ACCEPTED" if accepted else "REJECTED"
    print(f"Quality gate {outcome}: {_relative_to_project(report_path)}")


if __name__ == "__main__":
    main()
