"""Create deterministic immutable training batches from the raw MASSIVE split.

Run from the repository root after `download_data.py`:
    .venv/bin/python code/datasets/create_batches.py
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from pydantic import ValidationError


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from schemas import (
    BatchMetadata,
    BatchStatus,
    BatchesManifest,
    DatasetSplit,
    PipelineState,
    SourceManifest,
)


TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"
BATCHES_DIR = PROJECT_ROOT / "data" / "raw" / "batches"
BATCHES_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "batches_manifest.json"
PIPELINE_STATE_PATH = PROJECT_ROOT / "data" / "manifests" / "pipeline_state.json"
EXPECTED_COLUMNS = ["sample_id", "text", "locale", "intent", "split"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_project(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _load_batch_settings() -> tuple[int, int]:
    try:
        config = yaml.safe_load(TRAINING_CONFIG_PATH.read_text())
        batch_size = config["data"]["samples_per_batch"]
        seed = config["training"]["seed"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError(
            "configs/training.yaml must define data.samples_per_batch "
            "and training.seed"
        ) from error

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("data.samples_per_batch must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training.seed must be an integer")
    return batch_size, seed


def _read_source_manifest() -> SourceManifest:
    if not (PROJECT_ROOT / "data" / "manifests" / "source_manifest.json").is_file():
        raise FileNotFoundError("run code/datasets/download_data.py before batching")

    try:
        manifest = SourceManifest.model_validate_json(
            (PROJECT_ROOT / "data" / "manifests" / "source_manifest.json").read_text()
        )
    except (OSError, ValidationError) as error:
        raise ValueError("source_manifest.json is missing or invalid") from error

    return manifest


def _load_train_table(source_manifest: SourceManifest) -> pa.Table:
    metadata = source_manifest.splits[DatasetSplit.TRAIN]
    train_path = PROJECT_ROOT / metadata.path
    if not train_path.is_file():
        raise FileNotFoundError(f"source train file is missing: {metadata.path}")
    if _sha256(train_path) != metadata.sha256:
        raise ValueError("source train file SHA256 does not match source_manifest.json")

    table = pq.read_table(train_path)
    if table.column_names != EXPECTED_COLUMNS:
        raise ValueError("source train file has an unexpected schema")
    if table.num_rows != metadata.sample_count:
        raise ValueError("source train row count does not match source_manifest.json")
    if any(table[column].null_count for column in EXPECTED_COLUMNS):
        raise ValueError("source train file contains null values")
    if not pc.all(pc.equal(table["split"], DatasetSplit.TRAIN.value)).as_py():
        raise ValueError("source train file contains non-train rows")
    if pc.count_distinct(table["sample_id"]).as_py() != table.num_rows:
        raise ValueError("source train file contains duplicate sample_id values")
    return table


def _write_parquet(table: pa.Table, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(target_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json_manifest(path: Path, manifest: BatchesManifest | PipelineState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    temporary_path.replace(path)


def _read_valid_batch_artifacts(
    source_manifest: SourceManifest,
    batch_size: int,
    train_row_count: int,
) -> tuple[BatchesManifest, PipelineState] | None:
    if not BATCHES_MANIFEST_PATH.is_file() or not PIPELINE_STATE_PATH.is_file():
        return None

    try:
        batches_manifest = BatchesManifest.model_validate_json(
            BATCHES_MANIFEST_PATH.read_text()
        )
        pipeline_state = PipelineState.model_validate_json(
            PIPELINE_STATE_PATH.read_text()
        )
    except (OSError, ValidationError):
        return None

    source_train = source_manifest.splits[DatasetSplit.TRAIN]
    expected_batch_count = math.ceil(train_row_count / batch_size)
    if (
        batches_manifest.source_revision != source_manifest.revision
        or batches_manifest.source_train_sha256 != source_train.sha256
        or batches_manifest.batch_size != batch_size
        or len(batches_manifest.batches) != expected_batch_count
    ):
        return None

    known_batch_ids = {batch.batch_id for batch in batches_manifest.batches}
    if not set(pipeline_state.processed_batch_ids).issubset(known_batch_ids):
        return None
    if (
        pipeline_state.next_batch_id is not None
        and pipeline_state.next_batch_id not in known_batch_ids
    ):
        return None

    for batch in batches_manifest.batches:
        path = PROJECT_ROOT / batch.path
        if not path.is_file() or _sha256(path) != batch.sha256:
            return None
        if pq.ParquetFile(path).metadata.num_rows != batch.sample_count:
            return None

    return batches_manifest, pipeline_state


def _ensure_empty_batch_directory() -> None:
    existing_batches = sorted(BATCHES_DIR.glob("batch_*.parquet"))
    if existing_batches:
        raise RuntimeError(
            "batch files exist without a valid manifest; restore the manifest "
            "or remove the generated batches before recreating them"
        )


def _create_batches(
    train_table: pa.Table,
    source_manifest: SourceManifest,
    batch_size: int,
    seed: int,
) -> BatchesManifest:
    _ensure_empty_batch_directory()
    shuffled_indices = pa.array(
        np.random.default_rng(seed).permutation(train_table.num_rows),
        type=pa.int64(),
    )
    shuffled_train = train_table.take(shuffled_indices)
    batches: list[BatchMetadata] = []

    for index, offset in enumerate(
        range(0, shuffled_train.num_rows, batch_size),
        start=1,
    ):
        batch_id = f"batch_{index:05d}"
        batch_table = shuffled_train.slice(offset, batch_size)
        batch_path = BATCHES_DIR / f"{batch_id}.parquet"
        _write_parquet(batch_table, batch_path)
        batches.append(
            BatchMetadata(
                batch_id=batch_id,
                path=_relative_to_project(batch_path),
                sha256=_sha256(batch_path),
                sample_count=batch_table.num_rows,
                status=BatchStatus.PENDING,
            )
        )

    source_train = source_manifest.splits[DatasetSplit.TRAIN]
    return BatchesManifest(
        source_revision=source_manifest.revision,
        source_train_sha256=source_train.sha256,
        batch_size=batch_size,
        batches=batches,
    )


def _print_summary(manifest: BatchesManifest, *, reused: bool) -> None:
    action = "Reused" if reused else "Created"
    first_batch = manifest.batches[0]
    last_batch = manifest.batches[-1]
    print(
        f"{action} {len(manifest.batches)} batches of up to "
        f"{manifest.batch_size} samples"
    )
    print(f"  first: {first_batch.batch_id} ({first_batch.sample_count} rows)")
    print(f"  last: {last_batch.batch_id} ({last_batch.sample_count} rows)")


def main() -> None:
    batch_size, seed = _load_batch_settings()
    source_manifest = _read_source_manifest()
    train_table = _load_train_table(source_manifest)

    existing = _read_valid_batch_artifacts(
        source_manifest,
        batch_size,
        train_table.num_rows,
    )
    if existing is not None:
        batches_manifest, _ = existing
        _print_summary(batches_manifest, reused=True)
        return

    batches_manifest = _create_batches(
        train_table,
        source_manifest,
        batch_size,
        seed,
    )
    _write_json_manifest(BATCHES_MANIFEST_PATH, batches_manifest)
    _write_json_manifest(
        PIPELINE_STATE_PATH,
        PipelineState(next_batch_id=batches_manifest.batches[0].batch_id),
    )
    _print_summary(batches_manifest, reused=False)


if __name__ == "__main__":
    main()
