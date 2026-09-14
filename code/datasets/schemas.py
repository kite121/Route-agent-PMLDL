"""Validated data contracts shared by the data engineering pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    model_validator,
)


SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def _validate_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("path must use POSIX separators")

    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("path must be relative and must not contain '..'")
    return path.as_posix()


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
Locale = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z]{2,3}-[A-Z]{2}$"),
]
Sha256 = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
SourceRevision = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{40}$"),
]
BatchId = Annotated[
    str,
    StringConstraints(pattern=r"^batch_[0-9]{5}$"),
]
DatasetVersion = Annotated[
    str,
    StringConstraints(pattern=r"^v[0-9]{4}$"),
]
RelativePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(_validate_relative_path),
]
UtcDatetime = Annotated[datetime, AfterValidator(_normalize_utc)]


class DatasetSplit(StrEnum):
    """Supported MASSIVE dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class BatchStatus(StrEnum):
    """Lifecycle states of an immutable input batch."""

    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    PROCESSED = "processed"
    FAILED = "failed"


class SchemaModel(BaseModel):
    """Base settings for all persisted pipeline schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class RawSample(SchemaModel):
    """Normalized sample read from the source MASSIVE dataset."""

    sample_id: NonEmptyString
    text: NonEmptyString
    locale: Locale
    intent: NonEmptyString
    split: DatasetSplit


class ProcessedSample(SchemaModel):
    """Sample ready for model-pair construction and evaluation."""

    sample_id: NonEmptyString
    text: NonEmptyString
    locale: Locale
    source_intent: NonEmptyString
    label: NonEmptyString
    split: DatasetSplit
    batch_id: BatchId | None = None

    @model_validator(mode="after")
    def validate_batch_assignment(self) -> Self:
        if self.split is DatasetSplit.TRAIN and self.batch_id is None:
            raise ValueError("train samples must include batch_id")
        if self.split is not DatasetSplit.TRAIN and self.batch_id is not None:
            raise ValueError("validation and test samples must not include batch_id")
        return self


class DataFileMetadata(SchemaModel):
    """Integrity metadata for one generated data file."""

    path: RelativePath
    sha256: Sha256
    sample_count: PositiveInt


class BatchMetadata(DataFileMetadata):
    """Identity, integrity information, and state of one raw batch."""

    batch_id: BatchId
    status: BatchStatus = BatchStatus.PENDING


class SourceManifest(SchemaModel):
    """Pinned source dataset and its downloaded split files."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    dataset_name: NonEmptyString
    dataset_config: NonEmptyString
    revision: SourceRevision
    downloaded_at: UtcDatetime = Field(default_factory=_utc_now)
    splits: dict[DatasetSplit, DataFileMetadata]

    @model_validator(mode="after")
    def require_all_splits(self) -> Self:
        missing = set(DatasetSplit) - set(self.splits)
        if missing:
            names = ", ".join(sorted(split.value for split in missing))
            raise ValueError(f"source manifest is missing splits: {names}")
        return self


class BatchesManifest(SchemaModel):
    """Ordered collection of batches derived from a pinned train split."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source_revision: SourceRevision
    source_train_sha256: Sha256
    batch_size: PositiveInt
    created_at: UtcDatetime = Field(default_factory=_utc_now)
    batches: list[BatchMetadata] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batches(self) -> Self:
        batch_ids = [batch.batch_id for batch in self.batches]
        paths = [batch.path for batch in self.batches]

        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("batch IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("batch paths must be unique")
        if batch_ids != sorted(batch_ids):
            raise ValueError("batches must be ordered by batch_id")
        return self


class ValidationReport(SchemaModel):
    """Counts produced while validating and cleaning one input batch."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    batch_id: BatchId
    created_at: UtcDatetime = Field(default_factory=_utc_now)
    input_count: NonNegativeInt
    valid_count: NonNegativeInt
    dropped_count: NonNegativeInt
    rejected_by_reason: dict[NonEmptyString, NonNegativeInt] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.valid_count + self.dropped_count != self.input_count:
            raise ValueError("valid_count + dropped_count must equal input_count")
        if sum(self.rejected_by_reason.values()) < self.dropped_count:
            raise ValueError("every dropped sample must have a rejection reason")
        return self


class DatasetManifest(SchemaModel):
    """Lineage and integrity information for one processed dataset version."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    dataset_version: DatasetVersion
    dataset_sha256: Sha256
    source_dataset: NonEmptyString
    source_config: NonEmptyString
    source_revision: SourceRevision
    included_batch_ids: list[BatchId] = Field(min_length=1)
    created_at: UtcDatetime = Field(default_factory=_utc_now)
    splits: dict[DatasetSplit, DataFileMetadata]

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        missing = set(DatasetSplit) - set(self.splits)
        if missing:
            names = ", ".join(sorted(split.value for split in missing))
            raise ValueError(f"dataset manifest is missing splits: {names}")

        if len(self.included_batch_ids) != len(set(self.included_batch_ids)):
            raise ValueError("included batch IDs must be unique")
        if self.included_batch_ids != sorted(self.included_batch_ids):
            raise ValueError("included batch IDs must be ordered")
        return self


class PipelineState(SchemaModel):
    """Minimal restart state used to select the next unprocessed batch."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    current_dataset_version: DatasetVersion | None = None
    processed_batch_ids: list[BatchId] = Field(default_factory=list)
    next_batch_id: BatchId | None = None
    updated_at: UtcDatetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def validate_batch_progress(self) -> Self:
        if len(self.processed_batch_ids) != len(set(self.processed_batch_ids)):
            raise ValueError("processed batch IDs must be unique")
        if self.processed_batch_ids != sorted(self.processed_batch_ids):
            raise ValueError("processed batch IDs must be ordered")
        if self.next_batch_id in self.processed_batch_ids:
            raise ValueError("next_batch_id cannot already be processed")
        return self


__all__ = [
    "SCHEMA_VERSION",
    "BatchMetadata",
    "BatchStatus",
    "BatchesManifest",
    "DataFileMetadata",
    "DatasetManifest",
    "DatasetSplit",
    "PipelineState",
    "ProcessedSample",
    "RawSample",
    "SourceManifest",
    "ValidationReport",
]
