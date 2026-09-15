"""Scheduled batch retraining and deployment pipeline for Route Agent.

Each DAG run processes at most one immutable MASSIVE batch. Tasks exchange no
dataset rows or model artifacts through XCom; those remain versioned files in
the repository workspace and are referenced by manifests.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG, TaskGroup
from airflow.sdk.exceptions import AirflowException, AirflowSkipException


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_EXECUTABLE = PROJECT_ROOT / ".venv" / "bin" / "python"
COMPOSE_FILE = PROJECT_ROOT / "code" / "deployment" / "docker-compose.yml"
PIPELINE_STATE_PATH = PROJECT_ROOT / "data" / "manifests" / "pipeline_state.json"
MODEL_RUNS_DIR = PROJECT_ROOT / "models" / "runs"

DEFAULT_ARGS = {
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _run_project_script(script_path: str) -> None:
    """Run an existing project command without passing artifacts through XCom."""

    if not PYTHON_EXECUTABLE.is_file():
        raise AirflowException(f"Project virtual environment is missing: {PYTHON_EXECUTABLE}")
    script = PROJECT_ROOT / script_path
    if not script.is_file():
        raise AirflowException(f"Project script is missing: {script}")
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [str(PYTHON_EXECUTABLE), str(script)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def _require_pending_batch() -> None:
    """Skip a scheduled run cleanly when no immutable input batch remains."""

    try:
        state = json.loads(PIPELINE_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AirflowException("Pipeline state is missing or invalid") from error
    next_batch_id = state.get("next_batch_id") if isinstance(state, dict) else None
    if next_batch_id is None:
        raise AirflowSkipException("No pending batch is available for retraining")
    if not isinstance(next_batch_id, str) or not next_batch_id:
        raise AirflowException("Pipeline state has an invalid next_batch_id")
    print(f"Processing immutable batch {next_batch_id}")


def _require_accepted_quality_gate() -> None:
    """Block packaging and deployment unless this run's gate accepted the model."""

    candidates: list[tuple[str, Path, dict[str, object]]] = []
    for report_path in MODEL_RUNS_DIR.glob("*/*/quality_gate.json"):
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise AirflowException(f"Invalid quality-gate report: {report_path}") from error
        created_at = report.get("created_at") if isinstance(report, dict) else None
        if isinstance(created_at, str) and created_at:
            candidates.append((created_at, report_path, report))
    if not candidates:
        raise AirflowException("No quality-gate report was produced")
    _, report_path, report = max(candidates, key=lambda candidate: candidate[0])
    if report.get("decision") != "accepted":
        raise AirflowException(f"Quality gate rejected the model: {report_path}")


def _deploy_current_model() -> None:
    """Restart services so the API reloads the newly packaged model volume."""

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "up",
            "--build",
            "--force-recreate",
            "-d",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _run_deployment_smoke_check() -> None:
    """Verify that the restarted API serves the current package manifest."""

    _run_project_script("code/deployment/smoke_check.py")


with DAG(
    dag_id="agent_router_pipeline",
    description="One-batch route-agent retraining, quality gate, and deployment.",
    start_date=datetime(2026, 9, 15, tzinfo=timezone.utc),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["pmldl", "route-agent", "batch", "deployment"],
) as dag:
    with TaskGroup(group_id="data_engineering") as data_engineering:
        download_data = PythonOperator(
            task_id="download_data",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/datasets/download_data.py"},
        )
        create_batches = PythonOperator(
            task_id="create_batches",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/datasets/create_batches.py"},
        )
        require_pending_batch = PythonOperator(
            task_id="require_pending_batch",
            python_callable=_require_pending_batch,
        )
        validate_data = PythonOperator(
            task_id="validate_data",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/datasets/validate_data.py"},
        )
        prepare_data = PythonOperator(
            task_id="prepare_data",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/datasets/prepare_data.py"},
        )
        download_data >> create_batches >> require_pending_batch >> validate_data >> prepare_data

    with TaskGroup(group_id="model_engineering") as model_engineering:
        build_training_pairs = PythonOperator(
            task_id="build_training_pairs",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/models/build_training_pairs.py"},
        )
        train_model = PythonOperator(
            task_id="train_model",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/models/train.py"},
        )
        evaluate_model = PythonOperator(
            task_id="evaluate_model",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/models/evaluate.py"},
        )
        quality_gate = PythonOperator(
            task_id="quality_gate",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/models/quality_gate.py"},
        )
        require_accepted_quality_gate = PythonOperator(
            task_id="require_accepted_quality_gate",
            python_callable=_require_accepted_quality_gate,
        )
        build_training_pairs >> train_model >> evaluate_model >> quality_gate >> require_accepted_quality_gate

    with TaskGroup(group_id="deployment") as deployment:
        package_model = PythonOperator(
            task_id="package_model",
            python_callable=_run_project_script,
            op_kwargs={"script_path": "code/models/package_model.py"},
        )
        deploy_current_model = PythonOperator(
            task_id="deploy_current_model",
            python_callable=_deploy_current_model,
        )
        smoke_check = PythonOperator(
            task_id="smoke_check",
            python_callable=_run_deployment_smoke_check,
        )
        package_model >> deploy_current_model >> smoke_check

    data_engineering >> model_engineering >> deployment
