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


class PairKind(StrEnum):
    """Relation encoded by one training pair for the routing bi-encoder."""

    TOOL_POSITIVE = "tool_positive"
    TOOL_NEGATIVE = "tool_negative"
    FALLBACK_NEGATIVE = "fallback_negative"


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


class TrainingPair(SchemaModel):
    """One labelled query-to-tool relation used by OnlineContrastiveLoss."""

    pair_id: NonEmptyString
    sample_id: NonEmptyString
    query_text: NonEmptyString
    source_label: NonEmptyString
    candidate_tool: NonEmptyString
    tool_text: NonEmptyString
    label: Literal[0, 1]
    pair_kind: PairKind

    @model_validator(mode="after")
    def validate_pair_kind(self) -> Self:
        if self.pair_kind is PairKind.TOOL_POSITIVE and self.label != 1:
            raise ValueError("tool_positive pairs must have label 1")
        if self.pair_kind is not PairKind.TOOL_POSITIVE and self.label != 0:
            raise ValueError("negative pairs must have label 0")
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


class PreparationSplitStats(SchemaModel):
    """Cleaning and label-mapping counts for one processed dataset split."""

    input_count: NonNegativeInt
    output_count: NonNegativeInt
    dropped_count: NonNegativeInt
    registered_tool_count: NonNegativeInt
    fallback_count: NonNegativeInt
    rejected_by_reason: dict[NonEmptyString, NonNegativeInt] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.output_count + self.dropped_count != self.input_count:
            raise ValueError("output_count + dropped_count must equal input_count")
        if self.registered_tool_count + self.fallback_count != self.output_count:
            raise ValueError(
                "registered_tool_count + fallback_count must equal output_count"
            )
        if sum(self.rejected_by_reason.values()) < self.dropped_count:
            raise ValueError("every dropped sample must have a rejection reason")
        return self


class DataPreparationReport(SchemaModel):
    """Auditable preparation result for one versioned training dataset."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    dataset_version: DatasetVersion
    processed_batch_id: BatchId
    created_at: UtcDatetime = Field(default_factory=_utc_now)
    splits: dict[DatasetSplit, PreparationSplitStats]

    @model_validator(mode="after")
    def require_all_splits(self) -> Self:
        missing = set(DatasetSplit) - set(self.splits)
        if missing:
            names = ", ".join(sorted(split.value for split in missing))
            raise ValueError(f"preparation report is missing splits: {names}")
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


class TrainingPairsManifest(SchemaModel):
    """Lineage, integrity, and class balance of generated training pairs."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    dataset_version: DatasetVersion
    dataset_sha256: Sha256
    source_train_sha256: Sha256
    tools_sha256: Sha256
    seed: int
    negatives_per_query: PositiveInt
    tool_names: list[NonEmptyString] = Field(min_length=2)
    created_at: UtcDatetime = Field(default_factory=_utc_now)
    pairs: DataFileMetadata
    pair_counts: dict[PairKind, NonNegativeInt]
    fallback_negative_tool_counts: dict[NonEmptyString, NonNegativeInt]

    @model_validator(mode="after")
    def validate_pair_metadata(self) -> Self:
        if self.tool_names != sorted(self.tool_names):
            raise ValueError("tool_names must be ordered")
        if len(self.tool_names) != len(set(self.tool_names)):
            raise ValueError("tool_names must be unique")
        if set(self.pair_counts) != set(PairKind):
            raise ValueError("pair_counts must include every pair kind")
        if sum(self.pair_counts.values()) != self.pairs.sample_count:
            raise ValueError("pair_counts must equal the number of pairs")
        if set(self.fallback_negative_tool_counts) != set(self.tool_names):
            raise ValueError(
                "fallback negative counts must include every configured tool"
            )
        counts = self.fallback_negative_tool_counts.values()
        if max(counts) - min(counts) > 1:
            raise ValueError("fallback negative tool counts must be balanced")
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
    "DataPreparationReport",
    "DataFileMetadata",
    "DatasetManifest",
    "DatasetSplit",
    "PairKind",
    "PipelineState",
    "PreparationSplitStats",
    "ProcessedSample",
    "RawSample",
    "SourceManifest",
    "TrainingPair",
    "TrainingPairsManifest",
    "ValidationReport",
]
