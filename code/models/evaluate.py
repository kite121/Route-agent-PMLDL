"""Evaluate base and fine-tuned routing encoders on fixed dataset splits.

The fallback threshold is selected exclusively on validation by Macro F1. The
selected threshold is then applied unchanged to the held-out test split.

Run from the repository root after a completed `train.py` run:
    .venv/bin/python code/models/evaluate.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATASETS_DIR = PROJECT_ROOT / "code" / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from schemas import DatasetManifest, DatasetSplit, PipelineState


ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"
TOOLS_CONFIG_PATH = PROJECT_ROOT / "configs" / "tools.yaml"
PIPELINE_STATE_PATH = PROJECT_ROOT / "data" / "manifests" / "pipeline_state.json"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
RUNS_DIR = PROJECT_ROOT / "models" / "runs"

CLEARML_PROJECT_NAME = "PMLDL/Route-agent"
EVALUATION_BATCH_SIZE = 128
EVALUATION_SAMPLE_SEED = 42
PROCESSED_COLUMNS = [
    "sample_id",
    "text",
    "locale",
    "source_intent",
    "label",
    "split",
    "batch_id",
]


@dataclass(frozen=True)
class EvaluationSettings:
    base_model_name: str
    top_k: int
    samples_per_split: int


@dataclass(frozen=True)
class EncoderRun:
    name: str
    model_reference: str
    clearml_task_id: str | None


@dataclass(frozen=True)
class SplitScores:
    labels: np.ndarray
    top_indices: np.ndarray
    top_scores: np.ndarray
    top_k_indices: np.ndarray
    elapsed_seconds: float


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


def _load_settings() -> EvaluationSettings:
    try:
        config = yaml.safe_load(TRAINING_CONFIG_PATH.read_text())
        model_name = config["model"]["name"]
        top_k = config["evaluation"]["top_k"]
        samples_per_split = config["evaluation"]["samples_per_split"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError(
            "training.yaml must define model.name and evaluation settings"
        ) from error
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model.name must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("evaluation.top_k must be a positive integer")
    if (
        isinstance(samples_per_split, bool)
        or not isinstance(samples_per_split, int)
        or samples_per_split <= 0
    ):
        raise ValueError("evaluation.samples_per_split must be a positive integer")
    return EvaluationSettings(
        base_model_name=model_name.strip(),
        top_k=top_k,
        samples_per_split=samples_per_split,
    )


def _dataset_manifest_path(dataset_version: str) -> Path:
    return MANIFESTS_DIR / f"dataset_{dataset_version}.json"


def _load_tool_registry() -> tuple[list[str], dict[str, str], str]:
    try:
        config = yaml.safe_load(TOOLS_CONFIG_PATH.read_text())
        tools = config["tools"]
        fallback_label = config["fallback"]["name"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError(
            "configs/tools.yaml must define tools and fallback.name"
        ) from error
    if not isinstance(tools, dict) or not tools:
        raise ValueError("tools.yaml must define a non-empty tools mapping")
    if not isinstance(fallback_label, str) or not fallback_label.strip():
        raise ValueError("fallback.name must be a non-empty string")

    descriptions: dict[str, str] = {}
    for name, specification in tools.items():
        if not isinstance(name, str) or not isinstance(specification, dict):
            raise ValueError("tool registry has an invalid entry")
        description = specification.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"tool {name} must have a non-empty description")
        descriptions[name] = description.strip()
    if fallback_label in descriptions:
        raise ValueError("fallback.name must not duplicate a tool name")
    tool_names = sorted(descriptions)
    return tool_names, {name: descriptions[name] for name in tool_names}, fallback_label


def _load_dataset_manifest() -> DatasetManifest:
    try:
        state = PipelineState.model_validate_json(PIPELINE_STATE_PATH.read_text())
    except (OSError, ValidationError) as error:
        raise ValueError("pipeline_state.json is missing or invalid") from error
    if state.current_dataset_version is None:
        raise ValueError("run code/datasets/prepare_data.py before evaluation")
    try:
        manifest = DatasetManifest.model_validate_json(
            _dataset_manifest_path(state.current_dataset_version).read_text()
        )
    except (OSError, ValidationError) as error:
        raise ValueError("current dataset manifest is missing or invalid") from error
    if manifest.dataset_version != state.current_dataset_version:
        raise ValueError("dataset manifest version does not match pipeline state")
    return manifest


def _read_split(
    manifest: DatasetManifest,
    split: DatasetSplit,
    valid_labels: set[str],
) -> pa.Table:
    metadata = manifest.splits[split]
    path = PROJECT_ROOT / metadata.path
    if not path.is_file():
        raise FileNotFoundError(f"processed {split.value} split is missing: {metadata.path}")
    if _sha256(path) != metadata.sha256:
        raise ValueError(f"processed {split.value} checksum does not match manifest")
    table = pq.read_table(path)
    if table.column_names != PROCESSED_COLUMNS:
        raise ValueError(f"processed {split.value} has an unexpected schema")
    if table.num_rows != metadata.sample_count:
        raise ValueError(f"processed {split.value} count does not match manifest")
    if any(table[column].null_count for column in PROCESSED_COLUMNS if column != "batch_id"):
        raise ValueError(f"processed {split.value} contains null values")
    if not pc.all(pc.equal(table["split"], split.value)).as_py():
        raise ValueError(f"processed {split.value} contains wrong split rows")
    if pc.count_distinct(table["sample_id"]).as_py() != table.num_rows:
        raise ValueError(f"processed {split.value} contains duplicate sample IDs")
    labels = set(table["label"].to_pylist())
    if not labels.issubset(valid_labels):
        raise ValueError(f"processed {split.value} contains labels outside the registry")
    return table


def _sample_split(
    table: pa.Table,
    *,
    sample_count: int,
    seed: int,
) -> pa.Table:
    """Select a reproducible evaluation subset without mixing data splits."""
    if table.num_rows <= sample_count:
        return table
    indices = np.random.default_rng(seed).choice(
        table.num_rows,
        size=sample_count,
        replace=False,
    )
    return table.take(pa.array(np.sort(indices), type=pa.int64()))


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _latest_completed_encoder(dataset_version: str) -> EncoderRun:
    candidates: list[tuple[str, Path, dict[str, object]]] = []
    for metadata_path in (RUNS_DIR / dataset_version).glob("*/training_run.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        model_path = metadata_path.parent / "encoder"
        if metadata.get("status") == "completed" and model_path.is_dir():
            created_at = metadata.get("created_at")
            task_id = metadata.get("clearml_task_id")
            if isinstance(created_at, str) and isinstance(task_id, str):
                candidates.append((created_at, model_path, metadata))
    if not candidates:
        raise FileNotFoundError(
            "no completed local encoder found; wait for train.py to finish"
        )
    _, model_path, metadata = max(candidates, key=lambda candidate: candidate[0])
    task_id = metadata["clearml_task_id"]
    return EncoderRun(
        name="fine_tuned",
        model_reference=str(model_path),
        clearml_task_id=task_id if isinstance(task_id, str) else None,
    )


def _score_split(
    model: SentenceTransformer,
    table: pa.Table,
    tool_texts: list[str],
    tool_names: list[str],
    fallback_label: str,
    top_k: int,
    device: str,
) -> SplitScores:
    if top_k > len(tool_names):
        raise ValueError("evaluation.top_k cannot exceed the number of tools")
    label_to_index = {name: index for index, name in enumerate(tool_names)}
    fallback_index = len(tool_names)
    texts = table["text"].to_pylist()
    labels = np.array(
        [label_to_index.get(label, fallback_index) for label in table["label"].to_pylist()],
        dtype=np.int16,
    )
    if any(label != fallback_label and label not in label_to_index for label in table["label"].to_pylist()):
        raise ValueError("evaluation split has an unknown label")

    model.eval()
    tool_embeddings = model.encode(
        tool_texts,
        batch_size=len(tool_texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=device,
    )
    top_indices_parts: list[np.ndarray] = []
    top_scores_parts: list[np.ndarray] = []
    top_k_parts: list[np.ndarray] = []
    started_at = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(texts), EVALUATION_BATCH_SIZE):
            query_texts = [f"query: {text}" for text in texts[offset : offset + EVALUATION_BATCH_SIZE]]
            query_embeddings = model.encode(
                query_texts,
                batch_size=EVALUATION_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=device,
            )
            scores = query_embeddings @ tool_embeddings.T
            top_order = np.argsort(-scores, axis=1)[:, :top_k]
            top_indices_parts.append(top_order[:, 0].astype(np.int16))
            top_scores_parts.append(scores[np.arange(scores.shape[0]), top_order[:, 0]])
            top_k_parts.append(top_order.astype(np.int16))
    return SplitScores(
        labels=labels,
        top_indices=np.concatenate(top_indices_parts),
        top_scores=np.concatenate(top_scores_parts),
        top_k_indices=np.concatenate(top_k_parts),
        elapsed_seconds=time.perf_counter() - started_at,
    )


def _macro_f1_from_confusion(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1_values = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(f1_values.mean())


def _calibrate_threshold(scores: SplitScores, tool_count: int) -> tuple[float, float]:
    """Find the exact validation threshold maximizing final-label Macro F1."""

    fallback_index = tool_count
    confusion = np.zeros((tool_count + 1, tool_count + 1), dtype=np.int64)
    np.add.at(confusion, (scores.labels, scores.top_indices), 1)
    best_threshold = float(np.nextafter(scores.top_scores.min(), -np.inf))
    best_macro_f1 = _macro_f1_from_confusion(confusion)

    sorted_indices = np.argsort(scores.top_scores, kind="stable")
    sorted_scores = scores.top_scores[sorted_indices]
    position = 0
    while position < len(sorted_indices):
        next_position = position + 1
        while (
            next_position < len(sorted_indices)
            and sorted_scores[next_position] == sorted_scores[position]
        ):
            next_position += 1
        changed = sorted_indices[position:next_position]
        actual = scores.labels[changed]
        previous = scores.top_indices[changed]
        np.add.at(confusion, (actual, previous), -1)
        np.add.at(confusion, (actual, np.full(len(changed), fallback_index)), 1)
        threshold = float(np.nextafter(sorted_scores[position], np.inf))
        macro_f1 = _macro_f1_from_confusion(confusion)
        if macro_f1 > best_macro_f1:
            best_threshold = threshold
            best_macro_f1 = macro_f1
        position = next_position
    return best_threshold, best_macro_f1


def _final_predictions(scores: SplitScores, threshold: float, tool_count: int) -> np.ndarray:
    fallback_index = tool_count
    return np.where(
        scores.top_scores < threshold,
        fallback_index,
        scores.top_indices,
    ).astype(np.int16)


def _metrics(
    scores: SplitScores,
    threshold: float,
    tool_count: int,
) -> dict[str, float]:
    fallback_index = tool_count
    predictions = _final_predictions(scores, threshold, tool_count)
    all_labels = np.arange(tool_count + 1)
    routed = scores.labels != fallback_index
    if not routed.any():
        raise ValueError("evaluation split has no routed tool examples")

    routed_top_k = scores.top_k_indices[routed]
    routed_targets = scores.labels[routed]
    ranks = np.argmax(routed_top_k == routed_targets[:, None], axis=1) + 1
    found_in_top_k = np.any(routed_top_k == routed_targets[:, None], axis=1)
    reciprocal_rank = np.where(found_in_top_k, 1.0 / ranks, 0.0)
    fallback_precision, fallback_recall, fallback_f1, _ = precision_recall_fscore_support(
        scores.labels,
        predictions,
        labels=[fallback_index],
        average=None,
        zero_division=0,
    )
    average_latency_ms = 1000 * scores.elapsed_seconds / len(scores.labels)
    return {
        "accuracy_at_1": float(accuracy_score(scores.labels, predictions)),
        "recall_at_k": float(found_in_top_k.mean()),
        "mrr_at_k": float(reciprocal_rank.mean()),
        "macro_f1": float(
            f1_score(
                scores.labels,
                predictions,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "fallback_precision": float(fallback_precision[0]),
        "fallback_recall": float(fallback_recall[0]),
        "fallback_f1": float(fallback_f1[0]),
        "average_inference_ms": float(average_latency_ms),
    }


def _configure_clearml() -> None:
    load_dotenv(ENV_PATH, override=False)
    required = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"set {', '.join(missing)} in .env before evaluation")
    os.environ["CLEARML_LOG_ENVIRONMENT"] = ""


def _start_clearml_task(
    *,
    dataset_manifest: DatasetManifest,
    fine_tuned_run: EncoderRun,
    device: str,
    samples_per_split: int,
) -> object:
    _configure_clearml()
    from clearml import Task

    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name=f"evaluate-{dataset_manifest.dataset_version}",
        task_type=Task.TaskTypes.testing,
        tags=[dataset_manifest.dataset_version, "evaluation", "fixed-splits"],
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
        auto_resource_monitoring=False,
    )
    task.connect(
        {
            "data": {
                "dataset_version": dataset_manifest.dataset_version,
                "dataset_sha256": dataset_manifest.dataset_sha256,
                "validation_sha256": dataset_manifest.splits[DatasetSplit.VALIDATION].sha256,
                "test_sha256": dataset_manifest.splits[DatasetSplit.TEST].sha256,
            },
            "fine_tuned": {
                "training_task_id": fine_tuned_run.clearml_task_id,
                "model_path": fine_tuned_run.model_reference,
            },
            "runtime": {"device": device},
            "evaluation": {
                "strategy": "deterministic_uniform_without_replacement",
                "samples_per_split": samples_per_split,
                "sampling_seed": EVALUATION_SAMPLE_SEED,
            },
        },
        name="configuration",
    )
    return task


def _report_metrics(task: object, split: str, model_name: str, metrics: dict[str, float]) -> None:
    logger = task.get_logger()
    for metric_name, value in metrics.items():
        logger.report_scalar(
            title=split,
            series=f"{model_name}/{metric_name}",
            value=value,
            iteration=0,
        )


def _evaluate_encoder(
    run: EncoderRun,
    validation: pa.Table,
    test: pa.Table,
    tool_names: list[str],
    tool_descriptions: dict[str, str],
    fallback_label: str,
    top_k: int,
    device: str,
) -> tuple[float, dict[str, float], dict[str, float]]:
    print(f"Loading {run.name} encoder: {run.model_reference}", flush=True)
    model = SentenceTransformer(run.model_reference, device=device)
    tool_texts = [f"passage: {tool_descriptions[name]}" for name in tool_names]
    validation_scores = _score_split(
        model,
        validation,
        tool_texts,
        tool_names,
        fallback_label,
        top_k,
        device,
    )
    threshold, validation_macro_f1 = _calibrate_threshold(
        validation_scores,
        len(tool_names),
    )
    validation_metrics = _metrics(validation_scores, threshold, len(tool_names))
    if abs(validation_metrics["macro_f1"] - validation_macro_f1) > 1e-12:
        raise RuntimeError("calibrated validation Macro F1 is inconsistent")
    test_scores = _score_split(
        model,
        test,
        tool_texts,
        tool_names,
        fallback_label,
        top_k,
        device,
    )
    test_metrics = _metrics(test_scores, threshold, len(tool_names))
    return threshold, validation_metrics, test_metrics


def main() -> None:
    settings = _load_settings()
    tool_names, tool_descriptions, fallback_label = _load_tool_registry()
    if settings.top_k > len(tool_names):
        raise ValueError("evaluation.top_k exceeds the configured number of tools")
    dataset_manifest = _load_dataset_manifest()
    valid_labels = {*tool_names, fallback_label}
    validation = _read_split(
        dataset_manifest,
        DatasetSplit.VALIDATION,
        valid_labels,
    )
    test = _read_split(dataset_manifest, DatasetSplit.TEST, valid_labels)
    validation = _sample_split(
        validation,
        sample_count=settings.samples_per_split,
        seed=EVALUATION_SAMPLE_SEED,
    )
    test = _sample_split(
        test,
        sample_count=settings.samples_per_split,
        seed=EVALUATION_SAMPLE_SEED + 1,
    )
    fine_tuned_run = _latest_completed_encoder(dataset_manifest.dataset_version)
    device = _select_device()
    task = None

    try:
        task = _start_clearml_task(
            dataset_manifest=dataset_manifest,
            fine_tuned_run=fine_tuned_run,
            device=device,
            samples_per_split=settings.samples_per_split,
        )
        base_run = EncoderRun(
            name="base",
            model_reference=settings.base_model_name,
            clearml_task_id=None,
        )
        results: dict[str, object] = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "clearml_task_id": task.id,
            "dataset_version": dataset_manifest.dataset_version,
            "dataset_sha256": dataset_manifest.dataset_sha256,
            "training_task_id": fine_tuned_run.clearml_task_id,
            "top_k": settings.top_k,
            "evaluation_sample": {
                "strategy": "deterministic_uniform_without_replacement",
                "samples_per_split": settings.samples_per_split,
                "sampling_seed": EVALUATION_SAMPLE_SEED,
                "validation_rows": validation.num_rows,
                "test_rows": test.num_rows,
            },
            "tool_names": tool_names,
            "models": {},
        }
        for run in (base_run, fine_tuned_run):
            threshold, validation_metrics, test_metrics = _evaluate_encoder(
                run,
                validation,
                test,
                tool_names,
                tool_descriptions,
                fallback_label,
                settings.top_k,
                device,
            )
            results["models"][run.name] = {
                "model_reference": run.model_reference,
                "threshold": threshold,
                "validation": validation_metrics,
                "test": test_metrics,
            }
            _report_metrics(task, "validation", run.name, validation_metrics)
            _report_metrics(task, "test", run.name, test_metrics)
            task.get_logger().report_scalar(
                title="calibration",
                series=f"{run.name}/fallback_threshold",
                value=threshold,
                iteration=0,
            )
            print(
                f"{run.name}: validation Macro F1={validation_metrics['macro_f1']:.4f}, "
                f"test Macro F1={test_metrics['macro_f1']:.4f}",
                flush=True,
            )

        report_path = (
            Path(fine_tuned_run.model_reference).parent / "evaluation.json"
        )
        _write_json(report_path, results)
        if not task.upload_artifact(
            name="evaluation_report",
            artifact_object=str(report_path),
            wait_on_upload=True,
        ):
            raise RuntimeError("ClearML did not upload the evaluation report")
        print(f"Saved evaluation report to {_relative_to_project(report_path)}")
    except Exception as error:
        if task is not None:
            task.mark_failed(status_reason=f"{type(error).__name__}: {error}")
        raise
    else:
        task.close()


if __name__ == "__main__":
    main()
