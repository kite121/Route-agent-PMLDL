"""Create a versioned training dataset from the next validated batch.

The MASSIVE validation and test splits are cleaned once and remain fixed across
dataset versions. Each later invocation only appends one validated training
batch and creates the next immutable manifest.

Run from the repository root after `validate_data.py`:
    .venv/bin/python code/datasets/prepare_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

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
    DataFileMetadata,
    DataPreparationReport,
    DatasetManifest,
    DatasetSplit,
    PipelineState,
    PreparationSplitStats,
    ProcessedSample,
    SourceManifest,
    ValidationReport,
)
from validate_data import clean_raw_table


TOOLS_CONFIG_PATH = PROJECT_ROOT / "configs" / "tools.yaml"
SOURCE_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "source_manifest.json"
BATCHES_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "batches_manifest.json"
PIPELINE_STATE_PATH = PROJECT_ROOT / "data" / "manifests" / "pipeline_state.json"
VALIDATION_REPORTS_DIR = PROJECT_ROOT / "data" / "manifests" / "validation_reports"
PREPARATION_REPORTS_DIR = PROJECT_ROOT / "data" / "manifests" / "processing_reports"
DATASET_MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
VALIDATED_BATCHES_DIR = PROJECT_ROOT / "data" / "processed" / "validated"
FIXED_SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "fixed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_COLUMNS = ["sample_id", "text", "locale", "intent", "split"]
PROCESSED_COLUMNS = [
    "sample_id",
    "text",
    "locale",
    "source_intent",
    "label",
    "split",
    "batch_id",
]
PROCESSED_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("locale", pa.string()),
        pa.field("source_intent", pa.string()),
        pa.field("label", pa.string()),
        pa.field("split", pa.string()),
        pa.field("batch_id", pa.string()),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_project(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _write_json(
    path: Path,
    payload: (
        BatchesManifest
        | PipelineState
        | DatasetManifest
        | DataPreparationReport
    ),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(payload.model_dump_json(indent=2) + "\n")
    temporary_path.replace(path)


def _write_parquet(table: pa.Table, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(target_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_source_manifest() -> SourceManifest:
    try:
        return SourceManifest.model_validate_json(SOURCE_MANIFEST_PATH.read_text())
    except (OSError, ValidationError) as error:
        raise ValueError("run code/datasets/download_data.py before preparation") from error


def _load_pipeline_manifests() -> tuple[BatchesManifest, PipelineState]:
    try:
        batches_manifest = BatchesManifest.model_validate_json(
            BATCHES_MANIFEST_PATH.read_text()
        )
        pipeline_state = PipelineState.model_validate_json(
            PIPELINE_STATE_PATH.read_text()
        )
    except (OSError, ValidationError) as error:
        raise ValueError(
            "run code/datasets/create_batches.py before preparation"
        ) from error
    return batches_manifest, pipeline_state


def _load_label_settings() -> tuple[set[str], str]:
    try:
        config = yaml.safe_load(TOOLS_CONFIG_PATH.read_text())
        tools = config["tools"]
        fallback_label = config["fallback"]["name"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError(
            "configs/tools.yaml must define tools and fallback.name"
        ) from error

    if not isinstance(tools, dict) or not tools:
        raise ValueError("configs/tools.yaml must define a non-empty tools mapping")
    if not all(isinstance(name, str) and name.strip() for name in tools):
        raise ValueError("tool names must be non-empty strings")
    if not isinstance(fallback_label, str) or not fallback_label.strip():
        raise ValueError("configs/tools.yaml fallback.name must be a non-empty string")
    if fallback_label in tools:
        raise ValueError("fallback.name must not duplicate a registered tool name")
    return set(tools), fallback_label


def _select_next_batch(
    batches_manifest: BatchesManifest,
    pipeline_state: PipelineState,
) -> BatchMetadata | None:
    if pipeline_state.next_batch_id is None:
        return None
    for batch in batches_manifest.batches:
        if batch.batch_id == pipeline_state.next_batch_id:
            return batch
    raise ValueError(
        f"next_batch_id {pipeline_state.next_batch_id} is absent from batches_manifest"
    )


def _validate_table(
    table: pa.Table,
    expected_columns: list[str],
    expected_split: DatasetSplit,
    *,
    allow_null_batch_id: bool,
) -> None:
    if table.column_names != expected_columns:
        raise ValueError(f"unexpected columns for {expected_split.value} split")
    required_columns = [column for column in expected_columns if column != "batch_id"]
    if any(table[column].null_count for column in required_columns):
        raise ValueError(f"{expected_split.value} split contains null values")
    if "batch_id" in expected_columns:
        null_batch_ids = table["batch_id"].null_count
        if allow_null_batch_id and null_batch_ids != table.num_rows:
            raise ValueError(
                f"{expected_split.value} evaluation split must not have batch IDs"
            )
        if not allow_null_batch_id and null_batch_ids:
            raise ValueError(f"{expected_split.value} train split has null batch IDs")
    if not pc.all(pc.equal(table["split"], expected_split.value)).as_py():
        raise ValueError(f"file contains non-{expected_split.value} rows")
    if pc.count_distinct(table["sample_id"]).as_py() != table.num_rows:
        raise ValueError(f"{expected_split.value} split contains duplicate sample IDs")


def _read_source_split(
    source_manifest: SourceManifest,
    split: DatasetSplit,
) -> pa.Table:
    metadata = source_manifest.splits[split]
    path = PROJECT_ROOT / metadata.path
    if not path.is_file():
        raise FileNotFoundError(f"source {split.value} file is missing: {metadata.path}")
    if _sha256(path) != metadata.sha256:
        raise ValueError(f"source {split.value} checksum does not match manifest")

    table = pq.read_table(path)
    if table.num_rows != metadata.sample_count:
        raise ValueError(f"source {split.value} row count does not match manifest")
    _validate_table(
        table,
        RAW_COLUMNS,
        split,
        allow_null_batch_id=True,
    )
    return table


def _read_validated_batch(batch: BatchMetadata) -> tuple[pa.Table, ValidationReport]:
    data_path = VALIDATED_BATCHES_DIR / f"{batch.batch_id}.parquet"
    report_path = VALIDATION_REPORTS_DIR / f"{batch.batch_id}.json"
    try:
        report = ValidationReport.model_validate_json(report_path.read_text())
    except (OSError, ValidationError) as error:
        raise ValueError(f"validated batch report is missing: {batch.batch_id}") from error
    if report.batch_id != batch.batch_id:
        raise ValueError(f"validation report does not belong to {batch.batch_id}")
    if report.input_count != batch.sample_count:
        raise ValueError(f"validation report has wrong input count: {batch.batch_id}")
    if not data_path.is_file():
        raise FileNotFoundError(f"validated batch is missing: {batch.batch_id}")

    table = pq.read_table(data_path)
    if table.num_rows != report.valid_count:
        raise ValueError(f"validated batch row count does not match report: {batch.batch_id}")
    _validate_table(
        table,
        RAW_COLUMNS,
        DatasetSplit.TRAIN,
        allow_null_batch_id=True,
    )
    return table, report


def _read_processed_split(
    metadata: DataFileMetadata,
    split: DatasetSplit,
) -> pa.Table:
    path = PROJECT_ROOT / metadata.path
    if not path.is_file():
        raise FileNotFoundError(f"processed {split.value} file is missing: {metadata.path}")
    if _sha256(path) != metadata.sha256:
        raise ValueError(f"processed {split.value} checksum does not match manifest")

    table = pq.read_table(path)
    if table.num_rows != metadata.sample_count:
        raise ValueError(f"processed {split.value} row count does not match manifest")
    _validate_table(
        table,
        PROCESSED_COLUMNS,
        split,
        allow_null_batch_id=split is not DatasetSplit.TRAIN,
    )
    return table


def _transform_rows(
    table: pa.Table,
    split: DatasetSplit,
    registered_tools: set[str],
    fallback_label: str,
    *,
    batch_id: str | None,
) -> tuple[pa.Table, int, int]:
    rows: list[dict[str, str | None]] = []
    registered_tool_count = 0
    fallback_count = 0

    for raw in table.to_pylist():
        source_intent = raw["intent"]
        label = source_intent if source_intent in registered_tools else fallback_label
        sample = ProcessedSample(
            sample_id=raw["sample_id"],
            text=raw["text"],
            locale=raw["locale"],
            source_intent=source_intent,
            label=label,
            split=split,
            batch_id=batch_id,
        )
        rows.append(sample.model_dump(mode="json"))
        if label == fallback_label:
            fallback_count += 1
        else:
            registered_tool_count += 1

    return (
        pa.Table.from_pylist(rows, schema=PROCESSED_SCHEMA),
        registered_tool_count,
        fallback_count,
    )


def _make_split_stats(
    *,
    input_count: int,
    output_count: int,
    registered_tool_count: int,
    fallback_count: int,
    rejected_by_reason: dict[str, int],
) -> PreparationSplitStats:
    return PreparationSplitStats(
        input_count=input_count,
        output_count=output_count,
        dropped_count=input_count - output_count,
        registered_tool_count=registered_tool_count,
        fallback_count=fallback_count,
        rejected_by_reason=rejected_by_reason,
    )


def _combine_split_stats(
    previous: PreparationSplitStats,
    current: PreparationSplitStats,
) -> PreparationSplitStats:
    rejected_by_reason: dict[str, int] = dict(previous.rejected_by_reason)
    for reason, count in current.rejected_by_reason.items():
        rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + count
    return _make_split_stats(
        input_count=previous.input_count + current.input_count,
        output_count=previous.output_count + current.output_count,
        registered_tool_count=(
            previous.registered_tool_count + current.registered_tool_count
        ),
        fallback_count=previous.fallback_count + current.fallback_count,
        rejected_by_reason=rejected_by_reason,
    )


def _metadata_for(path: Path, sample_count: int) -> DataFileMetadata:
    return DataFileMetadata(
        path=_relative_to_project(path),
        sha256=_sha256(path),
        sample_count=sample_count,
    )


def _dataset_manifest_path(dataset_version: str) -> Path:
    return DATASET_MANIFESTS_DIR / f"dataset_{dataset_version}.json"


def _preparation_report_path(dataset_version: str) -> Path:
    return PREPARATION_REPORTS_DIR / f"{dataset_version}.json"


def _load_current_dataset(
    source_manifest: SourceManifest,
    pipeline_state: PipelineState,
) -> tuple[DatasetManifest | None, pa.Table | None, DataPreparationReport | None]:
    if pipeline_state.current_dataset_version is None:
        if pipeline_state.processed_batch_ids:
            raise ValueError("state has processed batches but no current dataset version")
        return None, None, None

    manifest_path = _dataset_manifest_path(pipeline_state.current_dataset_version)
    report_path = _preparation_report_path(pipeline_state.current_dataset_version)
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text())
        report = DataPreparationReport.model_validate_json(report_path.read_text())
    except (OSError, ValidationError) as error:
        raise ValueError("current dataset manifest or preparation report is invalid") from error

    if manifest.dataset_version != pipeline_state.current_dataset_version:
        raise ValueError("current dataset manifest version does not match pipeline state")
    if manifest.source_dataset != source_manifest.dataset_name:
        raise ValueError("current dataset uses a different source dataset")
    if manifest.source_config != source_manifest.dataset_config:
        raise ValueError("current dataset uses a different source configuration")
    if manifest.source_revision != source_manifest.revision:
        raise ValueError("current dataset uses a different source revision")
    if manifest.included_batch_ids != pipeline_state.processed_batch_ids:
        raise ValueError("current dataset batches do not match pipeline state")
    if report.dataset_version != manifest.dataset_version:
        raise ValueError("preparation report does not match current dataset version")

    train_table = _read_processed_split(
        manifest.splits[DatasetSplit.TRAIN],
        DatasetSplit.TRAIN,
    )
    batch_ids = set(train_table["batch_id"].to_pylist())
    if batch_ids != set(manifest.included_batch_ids):
        raise ValueError("current train split does not match its batch lineage")
    return manifest, train_table, report


def _build_fixed_splits(
    source_manifest: SourceManifest,
    registered_tools: set[str],
    fallback_label: str,
) -> tuple[
    dict[DatasetSplit, pa.Table],
    dict[DatasetSplit, DataFileMetadata],
    dict[DatasetSplit, PreparationSplitStats],
]:
    tables: dict[DatasetSplit, pa.Table] = {}
    metadata: dict[DatasetSplit, DataFileMetadata] = {}
    stats: dict[DatasetSplit, PreparationSplitStats] = {}

    for split in (DatasetSplit.VALIDATION, DatasetSplit.TEST):
        raw_table = _read_source_split(source_manifest, split)
        cleaned_table, rejected_by_reason, clean_registered_count = clean_raw_table(
            raw_table,
            registered_tools,
        )
        processed_table, registered_count, fallback_count = _transform_rows(
            cleaned_table,
            split,
            registered_tools,
            fallback_label,
            batch_id=None,
        )
        if clean_registered_count != registered_count:
            raise RuntimeError("registered-tool count changed during fixed split mapping")
        if processed_table.num_rows == 0:
            raise ValueError(f"fixed {split.value} split has no valid rows")

        target_path = FIXED_SPLITS_DIR / f"{split.value}.parquet"
        _write_parquet(processed_table, target_path)
        tables[split] = processed_table
        metadata[split] = _metadata_for(target_path, processed_table.num_rows)
        stats[split] = _make_split_stats(
            input_count=raw_table.num_rows,
            output_count=processed_table.num_rows,
            registered_tool_count=registered_count,
            fallback_count=fallback_count,
            rejected_by_reason=rejected_by_reason,
        )
    return tables, metadata, stats


def _reuse_fixed_splits(
    manifest: DatasetManifest,
    report: DataPreparationReport,
) -> tuple[
    dict[DatasetSplit, pa.Table],
    dict[DatasetSplit, DataFileMetadata],
    dict[DatasetSplit, PreparationSplitStats],
]:
    tables: dict[DatasetSplit, pa.Table] = {}
    metadata: dict[DatasetSplit, DataFileMetadata] = {}
    stats: dict[DatasetSplit, PreparationSplitStats] = {}
    for split in (DatasetSplit.VALIDATION, DatasetSplit.TEST):
        expected_path = _relative_to_project(FIXED_SPLITS_DIR / f"{split.value}.parquet")
        split_metadata = manifest.splits[split]
        if split_metadata.path != expected_path:
            raise ValueError(f"current {split.value} split is not the fixed split")
        tables[split] = _read_processed_split(split_metadata, split)
        metadata[split] = split_metadata
        stats[split] = report.splits[split]
    return tables, metadata, stats


def _assert_disjoint_splits(tables: dict[DatasetSplit, pa.Table]) -> None:
    sample_ids = {
        split: set(table["sample_id"].to_pylist()) for split, table in tables.items()
    }
    pairs = (
        (DatasetSplit.TRAIN, DatasetSplit.VALIDATION),
        (DatasetSplit.TRAIN, DatasetSplit.TEST),
        (DatasetSplit.VALIDATION, DatasetSplit.TEST),
    )
    for left, right in pairs:
        if sample_ids[left] & sample_ids[right]:
            raise ValueError(f"{left.value} and {right.value} splits overlap")


def _dataset_sha256(
    source_manifest: SourceManifest,
    batch_ids: list[str],
    split_metadata: dict[DatasetSplit, DataFileMetadata],
) -> str:
    payload = {
        "source_dataset": source_manifest.dataset_name,
        "source_config": source_manifest.dataset_config,
        "source_revision": source_manifest.revision,
        "included_batch_ids": batch_ids,
        "splits": {
            split.value: {
                "sha256": split_metadata[split].sha256,
                "sample_count": split_metadata[split].sample_count,
            }
            for split in DatasetSplit
        },
    }
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_payload.encode()).hexdigest()


def _next_dataset_version(processed_batch_ids: list[str]) -> str:
    number = len(processed_batch_ids) + 1
    if number > 9999:
        raise ValueError("dataset version limit reached")
    return f"v{number:04d}"


def _next_pipeline_state(
    batches_manifest: BatchesManifest,
    pipeline_state: PipelineState,
    batch: BatchMetadata,
    dataset_version: str,
) -> PipelineState:
    included_batch_ids = [*pipeline_state.processed_batch_ids, batch.batch_id]
    batch_index = next(
        index
        for index, candidate in enumerate(batches_manifest.batches)
        if candidate.batch_id == batch.batch_id
    )
    next_batch_id = (
        batches_manifest.batches[batch_index + 1].batch_id
        if batch_index + 1 < len(batches_manifest.batches)
        else None
    )
    return PipelineState(
        current_dataset_version=dataset_version,
        processed_batch_ids=included_batch_ids,
        next_batch_id=next_batch_id,
    )


def _update_batch_status(
    manifest: BatchesManifest,
    batch_id: str,
    status: BatchStatus,
) -> BatchesManifest:
    return manifest.model_copy(
        update={
            "batches": [
                batch.model_copy(update={"status": status})
                if batch.batch_id == batch_id
                else batch
                for batch in manifest.batches
            ]
        }
    )


def _load_completed_version(
    source_manifest: SourceManifest,
    pipeline_state: PipelineState,
    batch: BatchMetadata,
) -> DatasetManifest | None:
    dataset_version = _next_dataset_version(pipeline_state.processed_batch_ids)
    manifest_path = _dataset_manifest_path(dataset_version)
    report_path = _preparation_report_path(dataset_version)
    if not manifest_path.is_file() and not report_path.is_file():
        return None
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text())
        report = DataPreparationReport.model_validate_json(report_path.read_text())
    except (OSError, ValidationError) as error:
        raise ValueError("incomplete prepared dataset artifacts require inspection") from error

    expected_batches = [*pipeline_state.processed_batch_ids, batch.batch_id]
    if (
        manifest.dataset_version != dataset_version
        or manifest.source_revision != source_manifest.revision
        or manifest.included_batch_ids != expected_batches
        or report.dataset_version != dataset_version
        or report.processed_batch_id != batch.batch_id
    ):
        raise ValueError("existing prepared dataset does not match current batch")
    for split in DatasetSplit:
        _read_processed_split(manifest.splits[split], split)
    return manifest


def _finalize_state(
    batches_manifest: BatchesManifest,
    pipeline_state: PipelineState,
    batch: BatchMetadata,
    dataset_version: str,
) -> PipelineState:
    processed_manifest = _update_batch_status(
        batches_manifest,
        batch.batch_id,
        BatchStatus.PROCESSED,
    )
    _write_json(BATCHES_MANIFEST_PATH, processed_manifest)
    next_state = _next_pipeline_state(
        batches_manifest,
        pipeline_state,
        batch,
        dataset_version,
    )
    _write_json(PIPELINE_STATE_PATH, next_state)
    return next_state


def _print_summary(
    dataset_manifest: DatasetManifest,
    report: DataPreparationReport,
    *,
    reused: bool,
) -> None:
    action = "Reused" if reused else "Prepared"
    train_stats = report.splits[DatasetSplit.TRAIN]
    print(
        f"{action} {dataset_manifest.dataset_version} from "
        f"{report.processed_batch_id}"
    )
    for split in DatasetSplit:
        metadata = dataset_manifest.splits[split]
        print(
            f"  {split.value}: {metadata.sample_count} rows, "
            f"sha256={metadata.sha256}, path={metadata.path}"
        )
    print(
        f"  training labels: {train_stats.registered_tool_count} registered, "
        f"{train_stats.fallback_count} fallback"
    )


def main() -> None:
    source_manifest = _load_source_manifest()
    batches_manifest, pipeline_state = _load_pipeline_manifests()
    registered_tools, fallback_label = _load_label_settings()
    batch = _select_next_batch(batches_manifest, pipeline_state)

    if batch is None:
        print("No new data: every available batch has already been processed")
        return

    if batch.status is BatchStatus.PROCESSED:
        completed_manifest = _load_completed_version(
            source_manifest,
            pipeline_state,
            batch,
        )
        if completed_manifest is None:
            raise ValueError(f"{batch.batch_id} is processed but has no dataset manifest")
        next_state = _finalize_state(
            batches_manifest,
            pipeline_state,
            batch,
            completed_manifest.dataset_version,
        )
        report = DataPreparationReport.model_validate_json(
            _preparation_report_path(completed_manifest.dataset_version).read_text()
        )
        _print_summary(completed_manifest, report, reused=True)
        print(f"  resumed state; next batch: {next_state.next_batch_id or 'none'}")
        return

    if batch.status is not BatchStatus.VALIDATED:
        print(
            f"No validated batch available: {batch.batch_id} is "
            f"{batch.status.value}"
        )
        return

    completed_manifest = _load_completed_version(
        source_manifest,
        pipeline_state,
        batch,
    )
    if completed_manifest is not None:
        next_state = _finalize_state(
            batches_manifest,
            pipeline_state,
            batch,
            completed_manifest.dataset_version,
        )
        report = DataPreparationReport.model_validate_json(
            _preparation_report_path(completed_manifest.dataset_version).read_text()
        )
        _print_summary(completed_manifest, report, reused=True)
        print(f"  resumed state; next batch: {next_state.next_batch_id or 'none'}")
        return

    current_manifest, current_train, current_report = _load_current_dataset(
        source_manifest,
        pipeline_state,
    )
    validated_table, validation_report = _read_validated_batch(batch)
    new_train, registered_count, fallback_count = _transform_rows(
        validated_table,
        DatasetSplit.TRAIN,
        registered_tools,
        fallback_label,
        batch_id=batch.batch_id,
    )
    if new_train.num_rows == 0:
        raise ValueError(f"{batch.batch_id} has no valid rows to prepare")

    if current_manifest is None:
        train_table = new_train
        fixed_tables, fixed_metadata, fixed_stats = _build_fixed_splits(
            source_manifest,
            registered_tools,
            fallback_label,
        )
    else:
        if current_train is None or current_report is None:
            raise RuntimeError("current dataset state is incomplete")
        previous_ids = set(current_train["sample_id"].to_pylist())
        new_ids = set(new_train["sample_id"].to_pylist())
        if previous_ids & new_ids:
            raise ValueError("new batch overlaps existing training dataset")
        train_table = pa.concat_tables([current_train, new_train])
        fixed_tables, fixed_metadata, fixed_stats = _reuse_fixed_splits(
            current_manifest,
            current_report,
        )

    dataset_version = _next_dataset_version(pipeline_state.processed_batch_ids)
    train_path = PROCESSED_DIR / dataset_version / "train.parquet"
    _write_parquet(train_table, train_path)
    train_metadata = _metadata_for(train_path, train_table.num_rows)
    split_metadata = {
        DatasetSplit.TRAIN: train_metadata,
        **fixed_metadata,
    }
    all_tables = {DatasetSplit.TRAIN: train_table, **fixed_tables}
    _assert_disjoint_splits(all_tables)

    new_train_stats = _make_split_stats(
        input_count=validation_report.input_count,
        output_count=new_train.num_rows,
        registered_tool_count=registered_count,
        fallback_count=fallback_count,
        rejected_by_reason=validation_report.rejected_by_reason,
    )
    train_stats = (
        new_train_stats
        if current_report is None
        else _combine_split_stats(
            current_report.splits[DatasetSplit.TRAIN],
            new_train_stats,
        )
    )
    preparation_report = DataPreparationReport(
        dataset_version=dataset_version,
        processed_batch_id=batch.batch_id,
        splits={
            DatasetSplit.TRAIN: train_stats,
            DatasetSplit.VALIDATION: fixed_stats[DatasetSplit.VALIDATION],
            DatasetSplit.TEST: fixed_stats[DatasetSplit.TEST],
        },
    )
    included_batch_ids = [*pipeline_state.processed_batch_ids, batch.batch_id]
    dataset_manifest = DatasetManifest(
        dataset_version=dataset_version,
        dataset_sha256=_dataset_sha256(
            source_manifest,
            included_batch_ids,
            split_metadata,
        ),
        source_dataset=source_manifest.dataset_name,
        source_config=source_manifest.dataset_config,
        source_revision=source_manifest.revision,
        included_batch_ids=included_batch_ids,
        splits=split_metadata,
    )
    _write_json(_preparation_report_path(dataset_version), preparation_report)
    _write_json(_dataset_manifest_path(dataset_version), dataset_manifest)
    next_state = _finalize_state(
        batches_manifest,
        pipeline_state,
        batch,
        dataset_version,
    )
    _print_summary(dataset_manifest, preparation_report, reused=False)
    print(f"  next batch: {next_state.next_batch_id or 'none'}")


if __name__ == "__main__":
    main()
