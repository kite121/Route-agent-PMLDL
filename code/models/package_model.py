"""Package only a quality-gate-approved routing encoder for deployment.

The resulting archive contains the encoder, the exact tool registry and the
model manifest. It is published locally under ``models/current`` and uploaded
to ClearML as a deployment artifact.

Run from the repository root after an accepted quality gate:
    .venv/bin/python code/models/package_model.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
TOOLS_CONFIG_PATH = PROJECT_ROOT / "configs" / "tools.yaml"
RUNS_DIR = PROJECT_ROOT / "models" / "runs"
CURRENT_MODEL_DIR = PROJECT_ROOT / "models" / "current"

CLEARML_PROJECT_NAME = "PMLDL/Route-agent"
ARCHIVE_NAME = "agent_router.tar.gz"
MANIFEST_NAME = "model_manifest.json"
REQUIRED_ARCHIVE_PATHS = {"encoder", "tools.yaml", MANIFEST_NAME}


@dataclass(frozen=True)
class AcceptedRun:
    dataset_version: str
    training_task_id: str
    evaluation_task_id: str
    quality_gate_task_id: str
    run_dir: Path
    encoder_path: Path
    model_name: str
    dataset_sha256: str
    source_train_sha256: str
    training_pairs_sha256: str
    tools_sha256: str
    git_commit: str
    evaluation_sha256: str
    test_metrics: dict[str, float]
    threshold: float
    quality_checks: dict[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_descriptions_sha256(path: Path) -> str:
    """Hash the exact tool descriptions used as encoder passages in training."""

    try:
        config = yaml.safe_load(path.read_text())
        tools = config["tools"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError("tools.yaml is missing or invalid") from error
    if not isinstance(tools, dict) or not tools:
        raise ValueError("tools.yaml has no tools")
    descriptions: dict[str, str] = {}
    for name, specification in tools.items():
        description = specification.get("description") if isinstance(specification, dict) else None
        if not isinstance(name, str) or not isinstance(description, str) or not description.strip():
            raise ValueError("tools.yaml has an invalid tool description")
        descriptions[name] = description.strip()
    canonical = json.dumps(
        dict(sorted(descriptions.items())),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _relative_to_project(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _required_string(payload: dict[str, object], key: str, *, description: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} is missing {key}")
    return value


def _required_metrics(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError("evaluation report is missing fine-tuned test metrics")
    required = ("accuracy_at_1", "recall_at_k", "mrr_at_k", "macro_f1")
    metrics: dict[str, float] = {}
    for name in required:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"evaluation metric {name} is invalid")
        metric = float(value)
        if not 0.0 <= metric <= 1.0:
            raise ValueError(f"evaluation metric {name} must be from 0 to 1")
        metrics[name] = metric
    return metrics


def _latest_gate_path() -> Path:
    candidates: list[tuple[str, Path]] = []
    for path in RUNS_DIR.glob("*/*/quality_gate.json"):
        payload = _read_json(path, description="quality gate report")
        created_at = payload.get("created_at")
        if isinstance(created_at, str) and created_at:
            candidates.append((created_at, path))
    if not candidates:
        raise FileNotFoundError(
            "no quality gate report found; run code/models/quality_gate.py first"
        )
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _load_accepted_run() -> AcceptedRun:
    gate_path = _latest_gate_path()
    gate = _read_json(gate_path, description="quality gate report")
    if gate.get("schema_version") != 1:
        raise ValueError("quality gate report has an unsupported schema version")
    if gate.get("decision") != "accepted":
        raise ValueError(
            "the latest quality gate rejected its model; refusing to package it"
        )

    dataset_version = _required_string(gate, "dataset_version", description="gate")
    training_task_id = _required_string(gate, "training_task_id", description="gate")
    evaluation_task_id = _required_string(gate, "evaluation_task_id", description="gate")
    quality_gate_task_id = _required_string(gate, "clearml_task_id", description="gate")
    evaluation_sha256 = _required_string(gate, "evaluation_sha256", description="gate")
    checks = gate.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("quality gate report is missing its checks")

    run_dir = gate_path.parent
    training_path = run_dir / "training_run.json"
    evaluation_path = run_dir / "evaluation.json"
    encoder_path = run_dir / "encoder"
    training = _read_json(training_path, description="training run")
    evaluation = _read_json(evaluation_path, description="evaluation report")
    if not encoder_path.is_dir():
        raise FileNotFoundError(f"approved encoder is missing: {encoder_path}")
    if training.get("status") != "completed":
        raise ValueError("approved training run did not complete")
    if training.get("clearml_task_id") != training_task_id:
        raise ValueError("gate and training run refer to different ClearML tasks")
    if evaluation.get("clearml_task_id") != evaluation_task_id:
        raise ValueError("gate and evaluation report refer to different ClearML tasks")
    if evaluation.get("training_task_id") != training_task_id:
        raise ValueError("evaluation report refers to a different training task")
    if _sha256(evaluation_path) != evaluation_sha256:
        raise ValueError("evaluation report checksum does not match the quality gate")
    models = evaluation.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("fine_tuned"), dict):
        raise ValueError("evaluation report is missing fine-tuned model results")
    fine_tuned = models["fine_tuned"]
    if fine_tuned.get("model_reference") != str(encoder_path):
        raise ValueError("evaluation report did not evaluate the approved local encoder")
    if evaluation.get("dataset_version") != dataset_version:
        raise ValueError("gate and evaluation report use different dataset versions")

    threshold = fine_tuned.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("evaluation report has an invalid fallback threshold")
    return AcceptedRun(
        dataset_version=dataset_version,
        training_task_id=training_task_id,
        evaluation_task_id=evaluation_task_id,
        quality_gate_task_id=quality_gate_task_id,
        run_dir=run_dir,
        encoder_path=encoder_path,
        model_name=_required_string(training, "model_name", description="training run"),
        dataset_sha256=_required_string(
            training, "dataset_sha256", description="training run"
        ),
        source_train_sha256=_required_string(
            training, "source_train_sha256", description="training run"
        ),
        training_pairs_sha256=_required_string(
            training, "training_pairs_sha256", description="training run"
        ),
        tools_sha256=_required_string(training, "tools_sha256", description="training run"),
        git_commit=_required_string(training, "git_commit", description="training run"),
        evaluation_sha256=evaluation_sha256,
        test_metrics=_required_metrics(fine_tuned.get("test")),
        threshold=float(threshold),
        quality_checks=checks,
    )


def _model_manifest(
    accepted_run: AcceptedRun,
    *,
    package_task_id: str,
) -> dict[str, object]:
    model_version = f"{accepted_run.dataset_version}-{accepted_run.training_task_id[:8]}"
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "model_version": model_version,
        "base_model": accepted_run.model_name,
        "artifact": {
            "filename": ARCHIVE_NAME,
            "encoder_path": "encoder",
            "tool_registry_path": "tools.yaml",
        },
        "routing": {"fallback_threshold": accepted_run.threshold},
        "lineage": {
            "dataset_version": accepted_run.dataset_version,
            "dataset_sha256": accepted_run.dataset_sha256,
            "source_train_sha256": accepted_run.source_train_sha256,
            "training_pairs_sha256": accepted_run.training_pairs_sha256,
            "tools_sha256": accepted_run.tools_sha256,
            "tools_config_sha256": _sha256(TOOLS_CONFIG_PATH),
            "git_commit": accepted_run.git_commit,
            "training_task_id": accepted_run.training_task_id,
            "evaluation_task_id": accepted_run.evaluation_task_id,
            "quality_gate_task_id": accepted_run.quality_gate_task_id,
            "package_task_id": package_task_id,
            "evaluation_sha256": accepted_run.evaluation_sha256,
        },
        "test_metrics": accepted_run.test_metrics,
        "quality_gate": {
            "decision": "accepted",
            "checks": accepted_run.quality_checks,
        },
    }


def _create_archive(
    *,
    accepted_run: AcceptedRun,
    manifest: dict[str, object],
    staging_dir: Path,
) -> tuple[Path, Path]:
    tools_sha256 = _tool_descriptions_sha256(TOOLS_CONFIG_PATH)
    if tools_sha256 != accepted_run.tools_sha256:
        raise ValueError("current tools.yaml checksum differs from the trained registry")

    manifest_path = staging_dir / MANIFEST_NAME
    archive_path = staging_dir / ARCHIVE_NAME
    _write_json(manifest_path, manifest)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(accepted_run.encoder_path, arcname="encoder", recursive=True)
        archive.add(TOOLS_CONFIG_PATH, arcname="tools.yaml", recursive=False)
        archive.add(manifest_path, arcname=MANIFEST_NAME, recursive=False)
    _verify_archive(archive_path)
    return archive_path, manifest_path


def _verify_archive(archive_path: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        names = {member.name.rstrip("/") for member in archive.getmembers()}
    missing = [path for path in REQUIRED_ARCHIVE_PATHS if path not in names]
    if missing:
        raise ValueError(f"package archive is missing: {', '.join(sorted(missing))}")
    if not any(name.startswith("encoder/") for name in names):
        raise ValueError("package archive has an empty encoder directory")


def _publish_current(archive_path: Path, manifest_path: Path) -> tuple[Path, Path]:
    CURRENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target_archive = CURRENT_MODEL_DIR / ARCHIVE_NAME
    target_manifest = CURRENT_MODEL_DIR / MANIFEST_NAME
    archive_temporary = CURRENT_MODEL_DIR / f".{ARCHIVE_NAME}.tmp"
    manifest_temporary = CURRENT_MODEL_DIR / f".{MANIFEST_NAME}.tmp"
    shutil.copyfile(archive_path, archive_temporary)
    shutil.copyfile(manifest_path, manifest_temporary)
    os.replace(archive_temporary, target_archive)
    os.replace(manifest_temporary, target_manifest)
    return target_archive, target_manifest


def _configure_clearml() -> None:
    load_dotenv(ENV_PATH, override=False)
    required = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"set {', '.join(missing)} in .env before packaging")
    os.environ["CLEARML_LOG_ENVIRONMENT"] = ""


def _start_clearml_task(accepted_run: AcceptedRun, model_version: str):
    _configure_clearml()
    from clearml import Task

    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name=f"package-{model_version}",
        task_type=Task.TaskTypes.data_processing,
        tags=[accepted_run.dataset_version, "packaging", "deployment-candidate"],
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
        auto_resource_monitoring=False,
    )
    task.connect(
        {
            "source": {
                "training_task_id": accepted_run.training_task_id,
                "evaluation_task_id": accepted_run.evaluation_task_id,
                "quality_gate_task_id": accepted_run.quality_gate_task_id,
                "evaluation_sha256": accepted_run.evaluation_sha256,
            },
            "model": {
                "model_version": model_version,
                "base_model": accepted_run.model_name,
                "dataset_version": accepted_run.dataset_version,
            },
        },
        name="configuration",
    )
    return task


def main() -> None:
    accepted_run = _load_accepted_run()
    model_version = f"{accepted_run.dataset_version}-{accepted_run.training_task_id[:8]}"
    task = None

    try:
        task = _start_clearml_task(accepted_run, model_version)
        manifest = _model_manifest(accepted_run, package_task_id=task.id)
        with tempfile.TemporaryDirectory(dir=RUNS_DIR.parent) as temporary_dir:
            archive_path, manifest_path = _create_archive(
                accepted_run=accepted_run,
                manifest=manifest,
                staging_dir=Path(temporary_dir),
            )
            target_archive, target_manifest = _publish_current(
                archive_path,
                manifest_path,
            )
        if not task.upload_artifact(
            name="agent_router",
            artifact_object=str(target_archive),
            wait_on_upload=True,
        ):
            raise RuntimeError("ClearML did not upload the packaged model")
        if not task.upload_artifact(
            name="model_manifest",
            artifact_object=str(target_manifest),
            wait_on_upload=True,
        ):
            raise RuntimeError("ClearML did not upload the model manifest")
    except Exception as error:
        if task is not None:
            task.mark_failed(status_reason=f"{type(error).__name__}: {error}")
        raise
    else:
        task.close()

    print(f"Packaged {model_version}: {_relative_to_project(target_archive)}")
    print(f"Manifest: {_relative_to_project(target_manifest)}")


if __name__ == "__main__":
    main()
