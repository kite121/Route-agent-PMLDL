"""Fine-tune the routing bi-encoder and log an auditable ClearML experiment.

The script consumes the versioned pair artifact from `build_training_pairs.py`.
Every optimization batch is balanced: half positive and half negative pairs.

Run from the repository root:
    .venv/bin/python code/models/train.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from transformers import get_linear_schedule_with_warmup


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATASETS_DIR = PROJECT_ROOT / "code" / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from schemas import (
    DatasetManifest,
    DatasetSplit,
    PairKind,
    PipelineState,
    TrainingPairsManifest,
)


ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"
PIPELINE_STATE_PATH = PROJECT_ROOT / "data" / "manifests" / "pipeline_state.json"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
MODELS_DIR = PROJECT_ROOT / "models" / "runs"

CLEARML_PROJECT_NAME = "PMLDL/Route-agent"
LOSS_NAME = "OnlineContrastiveLoss"
LOSS_MARGIN = 0.5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

PAIR_COLUMNS = [
    "pair_id",
    "sample_id",
    "query_text",
    "source_label",
    "candidate_tool",
    "tool_text",
    "label",
    "pair_kind",
]


@dataclass(frozen=True)
class TrainingSettings:
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int


@dataclass(frozen=True)
class TrainingPair:
    query_text: str
    tool_text: str
    label: int


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


def _load_training_settings() -> TrainingSettings:
    try:
        config = yaml.safe_load(TRAINING_CONFIG_PATH.read_text())
        model_name = config["model"]["name"]
        training = config["training"]
        epochs = training["epochs"]
        batch_size = training["batch_size"]
        learning_rate = training["learning_rate"]
        seed = training["seed"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError(
            "training.yaml must define model.name and the training settings"
        ) from error

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model.name must be a non-empty string")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("training.epochs must be a positive integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 2
        or batch_size % 2
    ):
        raise ValueError("training.batch_size must be an even integer of at least 2")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or learning_rate <= 0
    ):
        raise ValueError("training.learning_rate must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training.seed must be an integer")
    return TrainingSettings(
        model_name=model_name.strip(),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        seed=seed,
    )


def _dataset_manifest_path(dataset_version: str) -> Path:
    return MANIFESTS_DIR / f"dataset_{dataset_version}.json"


def _pairs_manifest_path(dataset_version: str) -> Path:
    return MANIFESTS_DIR / f"training_pairs_{dataset_version}.json"


def _load_training_data() -> tuple[DatasetManifest, TrainingPairsManifest, list[TrainingPair]]:
    try:
        state = PipelineState.model_validate_json(PIPELINE_STATE_PATH.read_text())
    except (OSError, ValidationError) as error:
        raise ValueError("pipeline_state.json is missing or invalid") from error
    if state.current_dataset_version is None:
        raise ValueError("run code/datasets/prepare_data.py before training")

    dataset_version = state.current_dataset_version
    try:
        dataset_manifest = DatasetManifest.model_validate_json(
            _dataset_manifest_path(dataset_version).read_text()
        )
        pairs_manifest = TrainingPairsManifest.model_validate_json(
            _pairs_manifest_path(dataset_version).read_text()
        )
    except (OSError, ValidationError) as error:
        raise ValueError(
            "run code/models/build_training_pairs.py before training"
        ) from error

    train_metadata = dataset_manifest.splits[DatasetSplit.TRAIN]
    if (
        pairs_manifest.dataset_version != dataset_version
        or pairs_manifest.dataset_sha256 != dataset_manifest.dataset_sha256
        or pairs_manifest.source_train_sha256 != train_metadata.sha256
    ):
        raise ValueError("training pair lineage does not match the current dataset")

    pairs_path = PROJECT_ROOT / pairs_manifest.pairs.path
    if not pairs_path.is_file():
        raise FileNotFoundError(f"training pairs are missing: {pairs_manifest.pairs.path}")
    if _sha256(pairs_path) != pairs_manifest.pairs.sha256:
        raise ValueError("training pairs checksum does not match its manifest")

    table = pq.read_table(pairs_path)
    if table.column_names != PAIR_COLUMNS:
        raise ValueError("training pairs have an unexpected schema")
    if table.num_rows != pairs_manifest.pairs.sample_count:
        raise ValueError("training pair count does not match its manifest")
    if any(table[column].null_count for column in PAIR_COLUMNS):
        raise ValueError("training pairs contain null values")

    pairs: list[TrainingPair] = []
    observed_kinds: Counter[PairKind] = Counter()
    for row in table.to_pylist():
        label = row["label"]
        if label not in {0, 1}:
            raise ValueError("training pair labels must be 0 or 1")
        try:
            pair_kind = PairKind(row["pair_kind"])
        except ValueError as error:
            raise ValueError("training pair has an unknown pair kind") from error
        if (pair_kind is PairKind.TOOL_POSITIVE) != bool(label):
            raise ValueError("training pair kind does not match its label")
        pairs.append(
            TrainingPair(
                query_text=row["query_text"],
                tool_text=row["tool_text"],
                label=int(label),
            )
        )
        observed_kinds[pair_kind] += 1

    if dict(observed_kinds) != pairs_manifest.pair_counts:
        raise ValueError("training pair kinds do not match their manifest")
    if not any(pair.label == 1 for pair in pairs):
        raise ValueError("training pairs contain no positive examples")
    if not any(pair.label == 0 for pair in pairs):
        raise ValueError("training pairs contain no negative examples")
    return dataset_manifest, pairs_manifest, pairs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _repeat_to_length(
    indices: list[int],
    target_length: int,
    rng: random.Random,
) -> list[int]:
    result: list[int] = []
    while len(result) < target_length:
        cycle = list(indices)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:target_length]


def _balanced_batches(
    pairs: list[TrainingPair],
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    """Return batches with an equal number of positive and negative pairs."""

    positive_indices = [index for index, pair in enumerate(pairs) if pair.label == 1]
    negative_indices = [index for index, pair in enumerate(pairs) if pair.label == 0]
    if not positive_indices or not negative_indices:
        raise ValueError("balanced training needs both positive and negative pairs")

    per_class = batch_size // 2
    sampled_per_class = math.ceil(
        max(len(positive_indices), len(negative_indices)) / per_class
    ) * per_class
    rng = random.Random(seed + epoch)
    positives = _repeat_to_length(positive_indices, sampled_per_class, rng)
    negatives = _repeat_to_length(negative_indices, sampled_per_class, rng)
    batches: list[list[int]] = []
    for offset in range(0, sampled_per_class, per_class):
        batch = positives[offset : offset + per_class] + negatives[
            offset : offset + per_class
        ]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def _move_to_device(features: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in features.items()
    }


def _batch_features(
    model: SentenceTransformer,
    pairs: list[TrainingPair],
    batch_indices: list[int],
    device: str,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    batch = [pairs[index] for index in batch_indices]
    query_features = _move_to_device(
        model.tokenize([pair.query_text for pair in batch]),
        device,
    )
    tool_features = _move_to_device(
        model.tokenize([pair.tool_text for pair in batch]),
        device,
    )
    labels = torch.tensor(
        [pair.label for pair in batch],
        dtype=torch.float32,
        device=device,
    )
    return [query_features, tool_features], labels


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _configure_clearml() -> None:
    load_dotenv(ENV_PATH, override=False)
    required = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"set {names} in .env before running train.py")

    # Do not copy the process environment, including credentials, to Task metadata.
    os.environ["CLEARML_LOG_ENVIRONMENT"] = ""


def _start_clearml_task(
    *,
    settings: TrainingSettings,
    dataset_manifest: DatasetManifest,
    pairs_manifest: TrainingPairsManifest,
    device: str,
    git_commit: str,
    airflow_run_id: str | None,
):
    _configure_clearml()
    from clearml import Task

    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name=f"train-{dataset_manifest.dataset_version}",
        task_type=Task.TaskTypes.training,
        tags=[dataset_manifest.dataset_version, "bi-encoder", "online-contrastive"],
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
        auto_resource_monitoring=False,
    )
    task.connect(
        {
            "model": {"name": settings.model_name},
            "training": asdict(settings),
            "loss": {"name": LOSS_NAME, "margin": LOSS_MARGIN},
            "optimizer": {
                "name": "AdamW",
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
            },
            "data": {
                "dataset_version": dataset_manifest.dataset_version,
                "dataset_sha256": dataset_manifest.dataset_sha256,
                "source_train_sha256": pairs_manifest.source_train_sha256,
                "training_pairs_sha256": pairs_manifest.pairs.sha256,
                "training_pair_count": pairs_manifest.pairs.sample_count,
                "tools_sha256": pairs_manifest.tools_sha256,
            },
            "runtime": {
                "device": device,
                "git_commit": git_commit,
                "airflow_run_id": airflow_run_id,
            },
        },
        name="configuration",
    )
    return task


def _run_metadata(
    *,
    task_id: str,
    settings: TrainingSettings,
    dataset_manifest: DatasetManifest,
    pairs_manifest: TrainingPairsManifest,
    device: str,
    git_commit: str,
    airflow_run_id: str | None,
    status: str,
    epoch_losses: list[float] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "status": status,
        "clearml_task_id": task_id,
        "model_name": settings.model_name,
        "loss": {"name": LOSS_NAME, "margin": LOSS_MARGIN},
        "training": asdict(settings),
        "optimizer": {"name": "AdamW", "weight_decay": WEIGHT_DECAY},
        "device": device,
        "git_commit": git_commit,
        "airflow_run_id": airflow_run_id,
        "dataset_version": dataset_manifest.dataset_version,
        "dataset_sha256": dataset_manifest.dataset_sha256,
        "source_train_sha256": pairs_manifest.source_train_sha256,
        "training_pairs_sha256": pairs_manifest.pairs.sha256,
        "tools_sha256": pairs_manifest.tools_sha256,
        "epoch_losses": epoch_losses or [],
    }


def _train(
    *,
    model: SentenceTransformer,
    pairs: list[TrainingPair],
    settings: TrainingSettings,
    device: str,
    task,
) -> list[float]:
    train_loss = losses.OnlineContrastiveLoss(model=model, margin=LOSS_MARGIN)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    first_epoch_batches = _balanced_batches(
        pairs,
        settings.batch_size,
        settings.seed,
        epoch=0,
    )
    total_steps = len(first_epoch_batches) * settings.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    logger = task.get_logger()
    epoch_losses: list[float] = []
    global_step = 0

    for epoch in range(settings.epochs):
        model.train()
        batches = _balanced_batches(
            pairs,
            settings.batch_size,
            settings.seed,
            epoch,
        )
        batch_losses: list[float] = []
        for batch_indices in batches:
            sentence_features, labels = _batch_features(
                model,
                pairs,
                batch_indices,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            value = train_loss(sentence_features, labels)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            loss_value = float(value.detach().cpu().item())
            batch_losses.append(loss_value)
            logger.report_scalar(
                title="training",
                series="batch_loss",
                value=loss_value,
                iteration=global_step,
            )
            logger.report_scalar(
                title="training",
                series="learning_rate",
                value=float(scheduler.get_last_lr()[0]),
                iteration=global_step,
            )
            global_step += 1

        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)
        logger.report_scalar(
            title="training",
            series="epoch_loss",
            value=epoch_loss,
            iteration=epoch + 1,
        )
        logger.report_scalar(
            title="training",
            series="balanced_pairs_per_epoch",
            value=float(len(batches) * settings.batch_size),
            iteration=epoch + 1,
        )
        print(
            f"Epoch {epoch + 1}/{settings.epochs}: "
            f"loss={epoch_loss:.6f}, batches={len(batches)}",
            flush=True,
        )
    return epoch_losses


def main() -> None:
    settings = _load_training_settings()
    dataset_manifest, pairs_manifest, pairs = _load_training_data()
    _set_seed(settings.seed)
    device = _select_device()
    git_commit = _git_commit()
    airflow_run_id = os.environ.get("AIRFLOW_CTX_DAG_RUN_ID")
    task = None
    metadata_path: Path | None = None

    try:
        task = _start_clearml_task(
            settings=settings,
            dataset_manifest=dataset_manifest,
            pairs_manifest=pairs_manifest,
            device=device,
            git_commit=git_commit,
            airflow_run_id=airflow_run_id,
        )
        run_dir = MODELS_DIR / dataset_manifest.dataset_version / task.id
        model_dir = run_dir / "encoder"
        metadata_path = run_dir / "training_run.json"
        _write_json(
            metadata_path,
            _run_metadata(
                task_id=task.id,
                settings=settings,
                dataset_manifest=dataset_manifest,
                pairs_manifest=pairs_manifest,
                device=device,
                git_commit=git_commit,
                airflow_run_id=airflow_run_id,
                status="running",
            ),
        )

        print(f"ClearML Task: {task.id}", flush=True)
        print(f"Loading {settings.model_name} on {device}...", flush=True)
        model = SentenceTransformer(settings.model_name, device=device)
        epoch_losses = _train(
            model=model,
            pairs=pairs,
            settings=settings,
            device=device,
            task=task,
        )
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir), create_model_card=False)
        completed_metadata = _run_metadata(
            task_id=task.id,
            settings=settings,
            dataset_manifest=dataset_manifest,
            pairs_manifest=pairs_manifest,
            device=device,
            git_commit=git_commit,
            airflow_run_id=airflow_run_id,
            status="completed",
            epoch_losses=epoch_losses,
        )
        _write_json(metadata_path, completed_metadata)

        if not task.upload_artifact(
            name="routing_encoder",
            artifact_object=str(model_dir),
            metadata={
                "dataset_version": dataset_manifest.dataset_version,
                "dataset_sha256": dataset_manifest.dataset_sha256,
                "training_pairs_sha256": pairs_manifest.pairs.sha256,
            },
            wait_on_upload=True,
        ):
            raise RuntimeError("ClearML did not upload the routing encoder artifact")
        if not task.upload_artifact(
            name="training_run",
            artifact_object=str(metadata_path),
            wait_on_upload=True,
        ):
            raise RuntimeError("ClearML did not upload the training metadata artifact")
        if not task.upload_artifact(
            name="training_pairs_manifest",
            artifact_object=str(_pairs_manifest_path(dataset_manifest.dataset_version)),
            wait_on_upload=True,
        ):
            raise RuntimeError("ClearML did not upload the training-pairs manifest")

        logger = task.get_logger()
        logger.report_scalar(
            title="training",
            series="final_loss",
            value=epoch_losses[-1],
            iteration=settings.epochs,
        )
        print(f"Saved encoder to {_relative_to_project(model_dir)}")
        print(f"Final training loss: {epoch_losses[-1]:.6f}")
    except Exception as error:
        if task is not None:
            if metadata_path is not None:
                _write_json(
                    metadata_path,
                    _run_metadata(
                        task_id=task.id,
                        settings=settings,
                        dataset_manifest=dataset_manifest,
                        pairs_manifest=pairs_manifest,
                        device=device,
                        git_commit=git_commit,
                        airflow_run_id=airflow_run_id,
                        status="failed",
                    ),
                )
            task.mark_failed(status_reason=f"{type(error).__name__}: {error}")
        raise
    else:
        task.close()


if __name__ == "__main__":
    main()
