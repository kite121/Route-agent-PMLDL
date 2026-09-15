"""Validate and clean the next immutable raw batch.

Run from the repository root after `create_batches.py`:
    .venv/bin/python code/datasets/validate_data.py
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
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
    PipelineState,
    RawSample,
    ValidationReport,
)


TOOLS_CONFIG_PATH = PROJECT_ROOT / "configs" / "tools.yaml"
BATCHES_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "batches_manifest.json"
PIPELINE_STATE_PATH = PROJECT_ROOT / "data" / "manifests" / "pipeline_state.json"
VALIDATION_REPORTS_DIR = (
    PROJECT_ROOT / "data" / "manifests" / "validation_reports"
)
VALIDATED_BATCHES_DIR = PROJECT_ROOT / "data" / "processed" / "validated"

EXPECTED_COLUMNS = ["sample_id", "text", "locale", "intent", "split"]
MIN_TEXT_LENGTH = 2
MAX_TEXT_LENGTH = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registered_tools() -> set[str]:
    try:
        config = yaml.safe_load(TOOLS_CONFIG_PATH.read_text())
        tools = config["tools"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError(
            "configs/tools.yaml must define a non-empty tools mapping"
        ) from error

    if not isinstance(tools, dict) or not tools:
        raise ValueError("configs/tools.yaml must define a non-empty tools mapping")
    if not all(isinstance(name, str) and name for name in tools):
        raise ValueError("tool names must be non-empty strings")
    return set(tools)


def _read_manifests() -> tuple[BatchesManifest, PipelineState]:
    try:
        batches_manifest = BatchesManifest.model_validate_json(
            BATCHES_MANIFEST_PATH.read_text()
        )
        pipeline_state = PipelineState.model_validate_json(
            PIPELINE_STATE_PATH.read_text()
        )
    except (OSError, ValidationError) as error:
        raise ValueError(
            "run code/datasets/create_batches.py before validating batches"
        ) from error
    return batches_manifest, pipeline_state


def _select_next_batch(
    batches_manifest: BatchesManifest,
    pipeline_state: PipelineState,
) -> BatchMetadata:
    batch_id = pipeline_state.next_batch_id
    if batch_id is None:
        raise RuntimeError("there is no pending batch to validate")

    for batch in batches_manifest.batches:
        if batch.batch_id == batch_id:
            return batch
    raise ValueError(f"next_batch_id {batch_id} is absent from batches_manifest.json")


def _validated_batch_path(batch: BatchMetadata) -> Path:
    return VALIDATED_BATCHES_DIR / f"{batch.batch_id}.parquet"


def _validation_report_path(batch: BatchMetadata) -> Path:
    return VALIDATION_REPORTS_DIR / f"{batch.batch_id}.json"


def _update_batch_status(
    manifest: BatchesManifest,
    batch_id: str,
    status: BatchStatus,
) -> BatchesManifest:
    batches = [
        batch.model_copy(update={"status": status})
        if batch.batch_id == batch_id
        else batch
        for batch in manifest.batches
    ]
    return manifest.model_copy(update={"batches": batches})


def _write_json(path: Path, payload: BatchesManifest | ValidationReport) -> None:
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


def _read_batch_table(batch: BatchMetadata) -> pa.Table:
    path = PROJECT_ROOT / batch.path
    if not path.is_file():
        raise FileNotFoundError(f"raw batch is missing: {batch.path}")
    if _sha256(path) != batch.sha256:
        raise ValueError(f"raw batch checksum does not match: {batch.batch_id}")

    table = pq.read_table(path)
    if table.column_names != EXPECTED_COLUMNS:
        raise ValueError(f"raw batch has an unexpected schema: {batch.batch_id}")
    if table.num_rows != batch.sample_count:
        raise ValueError(f"raw batch row count does not match: {batch.batch_id}")
    return table


def clean_raw_table(
    table: pa.Table,
    registered_tools: set[str],
) -> tuple[pa.Table, dict[str, int], int]:
    valid_rows: list[dict[str, str]] = []
    rejected_by_reason: Counter[str] = Counter()
    seen_sample_ids: set[str] = set()
    seen_examples: set[tuple[str, str, str]] = set()
    registered_tool_count = 0

    for row in table.to_pylist():
        try:
            sample = RawSample.model_validate(row)
        except ValidationError:
            rejected_by_reason["invalid_schema"] += 1
            continue

        if len(sample.text) < MIN_TEXT_LENGTH:
            rejected_by_reason["text_too_short"] += 1
            continue
        if len(sample.text) > MAX_TEXT_LENGTH:
            rejected_by_reason["text_too_long"] += 1
            continue
        if sample.sample_id in seen_sample_ids:
            rejected_by_reason["duplicate_sample_id"] += 1
            continue

        example_key = (sample.locale, sample.text.casefold(), sample.intent)
        if example_key in seen_examples:
            rejected_by_reason["duplicate_example"] += 1
            continue

        seen_sample_ids.add(sample.sample_id)
        seen_examples.add(example_key)
        valid_rows.append(sample.model_dump(mode="json"))
        if sample.intent in registered_tools:
            registered_tool_count += 1

    validated_table = pa.Table.from_pylist(valid_rows, schema=table.schema)
    return validated_table, dict(rejected_by_reason), registered_tool_count


def _validate_batch(
    batch: BatchMetadata,
    registered_tools: set[str],
) -> tuple[pa.Table, ValidationReport, int]:
    table = _read_batch_table(batch)
    validated_table, rejected_by_reason, registered_tool_count = clean_raw_table(
        table,
        registered_tools,
    )
    report = ValidationReport(
        batch_id=batch.batch_id,
        input_count=table.num_rows,
        valid_count=validated_table.num_rows,
        dropped_count=table.num_rows - validated_table.num_rows,
        rejected_by_reason=rejected_by_reason,
    )
    return validated_table, report, registered_tool_count


def _print_summary(
    batch: BatchMetadata,
    report: ValidationReport,
    registered_tool_count: int | None,
    registered_tools: set[str],
    *,
    reused: bool,
) -> None:
    action = "Reused" if reused else "Validated"
    print(
        f"{action} {batch.batch_id}: {report.valid_count}/{report.input_count} "
        "rows accepted"
    )
    if registered_tool_count is not None:
        fallback_candidate_count = report.valid_count - registered_tool_count
        print(
            f"  registered-tool candidates: {registered_tool_count} "
            f"across {len(registered_tools)} configured tools"
        )
        print(f"  fallback candidates: {fallback_candidate_count}")
    if report.rejected_by_reason:
        print(f"  rejected: {dict(report.rejected_by_reason)}")


def main() -> None:
    registered_tools = _load_registered_tools()
    batches_manifest, pipeline_state = _read_manifests()
    batch = _select_next_batch(batches_manifest, pipeline_state)
    output_path = _validated_batch_path(batch)
    report_path = _validation_report_path(batch)

    if batch.status in {BatchStatus.VALIDATED, BatchStatus.PROCESSED}:
        if output_path.is_file() and report_path.is_file():
            report = ValidationReport.model_validate_json(report_path.read_text())
            _print_summary(
                batch,
                report,
                registered_tool_count=None,
                registered_tools=registered_tools,
                reused=True,
            )
            return
        raise RuntimeError(
            f"{batch.batch_id} is marked {batch.status} but lacks output"
        )

    if batch.status is not BatchStatus.PENDING:
        raise RuntimeError(
            f"{batch.batch_id} cannot be validated from status {batch.status}"
        )

    processing_manifest = _update_batch_status(
        batches_manifest,
        batch.batch_id,
        BatchStatus.PROCESSING,
    )
    _write_json(BATCHES_MANIFEST_PATH, processing_manifest)

    try:
        validated_table, report, registered_tool_count = _validate_batch(
            batch,
            registered_tools,
        )
        _write_json(report_path, report)
        if report.valid_count == 0:
            raise ValueError(f"{batch.batch_id} has no valid rows")
        _write_parquet(validated_table, output_path)
    except Exception:
        failed_manifest = _update_batch_status(
            processing_manifest,
            batch.batch_id,
            BatchStatus.FAILED,
        )
        _write_json(BATCHES_MANIFEST_PATH, failed_manifest)
        raise

    validated_manifest = _update_batch_status(
        processing_manifest,
        batch.batch_id,
        BatchStatus.VALIDATED,
    )
    _write_json(BATCHES_MANIFEST_PATH, validated_manifest)
    _print_summary(
        batch,
        report,
        registered_tool_count,
        registered_tools,
        reused=False,
    )


if __name__ == "__main__":
    main()
