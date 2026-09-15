"""Download a pinned Amazon MASSIVE snapshot and store normalized raw splits.

Run from the repository root:
    .venv/bin/python code/datasets/download_data.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# The Xet transport can stall on some unauthenticated networks.  Use the
# standard HTTPS download path unless the caller explicitly chose otherwise.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pydantic import ValidationError


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from schemas import DataFileMetadata, DatasetSplit, RawSample, SourceManifest


DATASET_NAME = "AmazonScience/massive"
DATASET_CONFIG = "all_1.1"
DATASET_REVISION = "e9957e4e8eb17fc67e3faf1a908829895653e775"

HF_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "source"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "source_manifest.json"

SOURCE_FILENAMES = {
    DatasetSplit.TRAIN: "all_1.1/massive-train.parquet",
    DatasetSplit.VALIDATION: "all_1.1/massive-validation.parquet",
    DatasetSplit.TEST: "all_1.1/massive-test.parquet",
}
REQUIRED_SOURCE_COLUMNS = {"id", "locale", "intent", "utt"}
NORMALIZED_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("locale", pa.string()),
        pa.field("intent", pa.string()),
        pa.field("split", pa.string()),
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


def _intent_names(source_path: Path) -> tuple[str, ...]:
    """Read the ClassLabel mapping embedded in the official Parquet metadata."""

    metadata = pq.ParquetFile(source_path).schema_arrow.metadata
    if metadata is None or b"huggingface" not in metadata:
        raise ValueError(f"{source_path.name} has no Hugging Face metadata")

    try:
        payload = json.loads(metadata[b"huggingface"])
        names = payload["info"]["features"]["intent"]["names"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{source_path.name} has no readable MASSIVE intent mapping"
        ) from error

    if not isinstance(names, list) or len(names) != 60 or len(set(names)) != 60:
        raise ValueError(f"{source_path.name} must define 60 unique intent labels")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{source_path.name} has invalid intent labels")
    return tuple(names)


def _intent_name(intent_index: object, intent_names: tuple[str, ...]) -> str:
    if isinstance(intent_index, bool) or not isinstance(intent_index, int):
        raise ValueError(f"MASSIVE intent must be an integer, got {intent_index!r}")
    if not 0 <= intent_index < len(intent_names):
        raise ValueError(f"unknown MASSIVE intent index: {intent_index}")
    return intent_names[intent_index]


def _ensure_source_columns(source_path: Path) -> None:
    source_columns = set(pq.ParquetFile(source_path).schema_arrow.names)
    missing = REQUIRED_SOURCE_COLUMNS - source_columns
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{source_path.name} is missing required columns: {names}")


def _normalized_record_batch(
    batch: pa.RecordBatch,
    split: DatasetSplit,
    intent_names: tuple[str, ...],
) -> pa.RecordBatch:
    values = batch.to_pydict()
    normalized = {field.name: [] for field in NORMALIZED_SCHEMA}

    for source_id, locale, intent_index, utterance in zip(
        values["id"],
        values["locale"],
        values["intent"],
        values["utt"],
        strict=True,
    ):
        try:
            sample = RawSample(
                sample_id=f"{split.value}:{locale}:{source_id}",
                text=utterance,
                locale=locale,
                intent=_intent_name(intent_index, intent_names),
                split=split,
            )
        except (ValidationError, ValueError) as error:
            raise ValueError(
                f"cannot normalize source row {source_id!r} in {split.value}"
            ) from error

        normalized["sample_id"].append(sample.sample_id)
        normalized["text"].append(sample.text)
        normalized["locale"].append(sample.locale)
        normalized["intent"].append(sample.intent)
        normalized["split"].append(sample.split.value)

    return pa.RecordBatch.from_pydict(normalized, schema=NORMALIZED_SCHEMA)


def _write_normalized_split(
    source_path: Path,
    target_path: Path,
    split: DatasetSplit,
) -> DataFileMetadata:
    _ensure_source_columns(source_path)
    intent_names = _intent_names(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_suffix(".parquet.tmp")
    source_file = pq.ParquetFile(source_path)
    sample_count = 0

    try:
        with pq.ParquetWriter(
            temporary_path,
            NORMALIZED_SCHEMA,
            compression="zstd",
        ) as writer:
            for batch in source_file.iter_batches(
                batch_size=10_000,
                columns=["id", "locale", "intent", "utt"],
            ):
                normalized_batch = _normalized_record_batch(
                    batch,
                    split,
                    intent_names,
                )
                writer.write_batch(normalized_batch)
                sample_count += normalized_batch.num_rows
        temporary_path.replace(target_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return DataFileMetadata(
        path=_relative_to_project(target_path),
        sha256=_sha256(target_path),
        sample_count=sample_count,
    )


def _read_valid_manifest() -> SourceManifest | None:
    if not MANIFEST_PATH.is_file():
        return None

    try:
        manifest = SourceManifest.model_validate_json(MANIFEST_PATH.read_text())
    except (OSError, ValidationError):
        return None

    if (
        manifest.dataset_name != DATASET_NAME
        or manifest.dataset_config != DATASET_CONFIG
        or manifest.revision != DATASET_REVISION
    ):
        return None

    for split in DatasetSplit:
        metadata = manifest.splits.get(split)
        if metadata is None:
            return None

        path = PROJECT_ROOT / metadata.path
        if not path.is_file() or _sha256(path) != metadata.sha256:
            return None
        if pq.ParquetFile(path).metadata.num_rows != metadata.sample_count:
            return None

    return manifest


def _write_manifest(manifest: SourceManifest) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    temporary_path.replace(MANIFEST_PATH)


def _print_summary(manifest: SourceManifest, *, reused: bool) -> None:
    action = "Reused" if reused else "Downloaded"
    print(
        f"{action} {manifest.dataset_name} ({manifest.dataset_config}) "
        f"at revision {manifest.revision}"
    )
    for split in DatasetSplit:
        metadata = manifest.splits[split]
        print(
            f"  {split.value}: {metadata.sample_count} rows, "
            f"sha256={metadata.sha256}, path={metadata.path}"
        )


def main() -> None:
    existing_manifest = _read_valid_manifest()
    if existing_manifest is not None:
        _print_summary(existing_manifest, reused=True)
        return

    split_metadata: dict[DatasetSplit, DataFileMetadata] = {}
    for split, filename in SOURCE_FILENAMES.items():
        print(f"Downloading {filename}...", flush=True)
        source_path = Path(
            hf_hub_download(
                repo_id=DATASET_NAME,
                repo_type="dataset",
                filename=filename,
                revision=DATASET_REVISION,
                cache_dir=HF_CACHE_DIR,
            )
        )
        target_path = SOURCE_DIR / f"{split.value}.parquet"
        split_metadata[split] = _write_normalized_split(
            source_path,
            target_path,
            split,
        )

    manifest = SourceManifest(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        revision=DATASET_REVISION,
        splits=split_metadata,
    )
    _write_manifest(manifest)
    _print_summary(manifest, reused=False)


if __name__ == "__main__":
    main()
