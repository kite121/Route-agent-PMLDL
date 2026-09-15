"""Build reproducible labelled pairs for routing-encoder fine-tuning.

Each registered-tool request creates one positive and three negative pairs.
Each ``no_tool`` request creates three balanced negative pairs across the tool
registry. The output is intended for ``OnlineContrastiveLoss``.

Run from the repository root after `prepare_data.py`:
    .venv/bin/python code/models/build_training_pairs.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from pydantic import ValidationError


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATASETS_DIR = PROJECT_ROOT / "code" / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from schemas import (
    DataFileMetadata,
    DatasetManifest,
    DatasetSplit,
    PairKind,
    PipelineState,
    ProcessedSample,
    TrainingPair,
    TrainingPairsManifest,
)


TRAINING_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"
TOOLS_CONFIG_PATH = PROJECT_ROOT / "configs" / "tools.yaml"
PIPELINE_STATE_PATH = PROJECT_ROOT / "data" / "manifests" / "pipeline_state.json"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"

PROCESSED_COLUMNS = [
    "sample_id",
    "text",
    "locale",
    "source_intent",
    "label",
    "split",
    "batch_id",
]
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
PAIR_SCHEMA = pa.schema(
    [
        pa.field("pair_id", pa.string()),
        pa.field("sample_id", pa.string()),
        pa.field("query_text", pa.string()),
        pa.field("source_label", pa.string()),
        pa.field("candidate_tool", pa.string()),
        pa.field("tool_text", pa.string()),
        pa.field("label", pa.int8()),
        pa.field("pair_kind", pa.string()),
    ]
)
NEGATIVES_PER_QUERY = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_project(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _dataset_manifest_path(dataset_version: str) -> Path:
    return MANIFESTS_DIR / f"dataset_{dataset_version}.json"


def _pairs_manifest_path(dataset_version: str) -> Path:
    return MANIFESTS_DIR / f"training_pairs_{dataset_version}.json"


def _pairs_path(dataset_version: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / dataset_version / "training_pairs.parquet"


def _write_json(path: Path, payload: TrainingPairsManifest) -> None:
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


def _load_seed() -> int:
    try:
        config = yaml.safe_load(TRAINING_CONFIG_PATH.read_text())
        seed = config["training"]["seed"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError("configs/training.yaml must define training.seed") from error
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training.seed must be an integer")
    return seed


def _load_tool_registry() -> tuple[dict[str, str], str, str]:
    try:
        config = yaml.safe_load(TOOLS_CONFIG_PATH.read_text())
        tools = config["tools"]
        fallback_label = config["fallback"]["name"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError(
            "configs/tools.yaml must define tools and fallback.name"
        ) from error

    if not isinstance(tools, dict) or len(tools) <= NEGATIVES_PER_QUERY:
        raise ValueError(
            f"tools.yaml must define more than {NEGATIVES_PER_QUERY} tools"
        )
    if not isinstance(fallback_label, str) or not fallback_label.strip():
        raise ValueError("fallback.name must be a non-empty string")

    descriptions: dict[str, str] = {}
    for name, specification in tools.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool names must be non-empty strings")
        if not isinstance(specification, dict):
            raise ValueError(f"tool {name} must be a mapping")
        description = specification.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"tool {name} must have a non-empty description")
        descriptions[name] = description.strip()

    if fallback_label in descriptions:
        raise ValueError("fallback.name must not be a registered tool")
    if len({description.casefold() for description in descriptions.values()}) != len(
        descriptions
    ):
        raise ValueError("tool descriptions must be unique")

    ordered_descriptions = dict(sorted(descriptions.items()))
    canonical = json.dumps(
        ordered_descriptions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        ordered_descriptions,
        fallback_label,
        hashlib.sha256(canonical.encode()).hexdigest(),
    )


def _load_current_dataset() -> tuple[DatasetManifest, pa.Table]:
    try:
        state = PipelineState.model_validate_json(PIPELINE_STATE_PATH.read_text())
    except (OSError, ValidationError) as error:
        raise ValueError("pipeline_state.json is missing or invalid") from error
    if state.current_dataset_version is None:
        raise ValueError("run code/datasets/prepare_data.py before building pairs")

    try:
        manifest = DatasetManifest.model_validate_json(
            _dataset_manifest_path(state.current_dataset_version).read_text()
        )
    except (OSError, ValidationError) as error:
        raise ValueError("current dataset manifest is missing or invalid") from error
    if manifest.dataset_version != state.current_dataset_version:
        raise ValueError("dataset manifest version does not match pipeline state")

    train_metadata = manifest.splits[DatasetSplit.TRAIN]
    train_path = PROJECT_ROOT / train_metadata.path
    if not train_path.is_file():
        raise FileNotFoundError(f"processed train file is missing: {train_metadata.path}")
    if _sha256(train_path) != train_metadata.sha256:
        raise ValueError("processed train checksum does not match dataset manifest")

    table = pq.read_table(train_path)
    if table.column_names != PROCESSED_COLUMNS:
        raise ValueError("processed train file has an unexpected schema")
    if table.num_rows != train_metadata.sample_count:
        raise ValueError("processed train count does not match dataset manifest")
    if any(table[column].null_count for column in PROCESSED_COLUMNS):
        raise ValueError("processed train file contains null values")
    if not pc.all(pc.equal(table["split"], DatasetSplit.TRAIN.value)).as_py():
        raise ValueError("processed train file contains non-train rows")
    if pc.count_distinct(table["sample_id"]).as_py() != table.num_rows:
        raise ValueError("processed train file contains duplicate sample IDs")
    return manifest, table


def _balanced_fallback_candidates(
    sample_ids: list[str],
    tool_names: list[str],
    seed: int,
) -> dict[str, tuple[str, ...]]:
    """Assign distinct, seeded tool negatives with globally balanced counts."""

    total_assignments = len(sample_ids) * NEGATIVES_PER_QUERY
    base_count, extra_count = divmod(total_assignments, len(tool_names))
    rng = random.Random(seed)
    tool_order = list(tool_names)
    rng.shuffle(tool_order)
    remaining = {
        name: base_count + int(index < extra_count)
        for index, name in enumerate(tool_order)
    }
    assignments: dict[str, tuple[str, ...]] = {}

    for sample_id in sample_ids:
        selected: list[str] = []
        for _ in range(NEGATIVES_PER_QUERY):
            available = [
                name
                for name in tool_names
                if remaining[name] > 0 and name not in selected
            ]
            if not available:
                raise RuntimeError("cannot allocate distinct balanced fallback negatives")
            largest_remaining = max(remaining[name] for name in available)
            candidates = [
                name for name in available if remaining[name] == largest_remaining
            ]
            chosen = rng.choice(candidates)
            remaining[chosen] -= 1
            selected.append(chosen)
        assignments[sample_id] = tuple(selected)

    if any(remaining.values()):
        raise RuntimeError("fallback negative allocation did not consume all slots")
    return assignments


def _pair_row(
    *,
    sample: ProcessedSample,
    candidate_tool: str,
    tool_description: str,
    label: int,
    pair_kind: PairKind,
) -> dict[str, str | int]:
    pair = TrainingPair(
        pair_id=f"{sample.sample_id}::{candidate_tool}",
        sample_id=sample.sample_id,
        query_text=f"query: {sample.text}",
        source_label=sample.label,
        candidate_tool=candidate_tool,
        tool_text=f"passage: {tool_description}",
        label=label,
        pair_kind=pair_kind,
    )
    return pair.model_dump(mode="json")


def _build_pairs(
    train_table: pa.Table,
    tool_descriptions: dict[str, str],
    fallback_label: str,
    seed: int,
) -> pa.Table:
    tool_names = list(tool_descriptions)
    rng = random.Random(seed)
    samples = [
        ProcessedSample.model_validate(row)
        for row in sorted(train_table.to_pylist(), key=lambda row: row["sample_id"])
    ]
    fallback_sample_ids = [
        sample.sample_id for sample in samples if sample.label == fallback_label
    ]
    fallback_candidates = _balanced_fallback_candidates(
        fallback_sample_ids,
        tool_names,
        seed,
    )
    pairs: list[dict[str, str | int]] = []

    for sample in samples:
        if sample.label == fallback_label:
            for candidate_tool in fallback_candidates[sample.sample_id]:
                pairs.append(
                    _pair_row(
                        sample=sample,
                        candidate_tool=candidate_tool,
                        tool_description=tool_descriptions[candidate_tool],
                        label=0,
                        pair_kind=PairKind.FALLBACK_NEGATIVE,
                    )
                )
            continue

        if sample.label not in tool_descriptions:
            raise ValueError(
                f"{sample.sample_id} has label outside the current tool registry"
            )
        pairs.append(
            _pair_row(
                sample=sample,
                candidate_tool=sample.label,
                tool_description=tool_descriptions[sample.label],
                label=1,
                pair_kind=PairKind.TOOL_POSITIVE,
            )
        )
        negative_candidates = rng.sample(
            [name for name in tool_names if name != sample.label],
            k=NEGATIVES_PER_QUERY,
        )
        for candidate_tool in negative_candidates:
            pairs.append(
                _pair_row(
                    sample=sample,
                    candidate_tool=candidate_tool,
                    tool_description=tool_descriptions[candidate_tool],
                    label=0,
                    pair_kind=PairKind.TOOL_NEGATIVE,
                )
            )

    return pa.Table.from_pylist(pairs, schema=PAIR_SCHEMA)


def _validate_pair_table(
    table: pa.Table,
    train_samples: dict[str, ProcessedSample],
    tool_descriptions: dict[str, str],
    fallback_label: str,
) -> tuple[Counter[PairKind], Counter[str]]:
    if table.column_names != PAIR_COLUMNS:
        raise ValueError("training pairs have an unexpected schema")
    if any(table[column].null_count for column in PAIR_COLUMNS):
        raise ValueError("training pairs contain null values")
    if pc.count_distinct(table["pair_id"]).as_py() != table.num_rows:
        raise ValueError("training pair IDs are not unique")

    pair_counts: Counter[PairKind] = Counter()
    fallback_tool_counts: Counter[str] = Counter()
    pairs_by_sample: dict[str, list[TrainingPair]] = defaultdict(list)
    for row in table.to_pylist():
        pair = TrainingPair.model_validate(row)
        sample = train_samples.get(pair.sample_id)
        if sample is None:
            raise ValueError(f"pair uses sample outside train: {pair.sample_id}")
        if pair.query_text != f"query: {sample.text}":
            raise ValueError(f"pair query does not match source: {pair.pair_id}")
        if pair.source_label != sample.label:
            raise ValueError(f"pair label does not match source: {pair.pair_id}")
        if pair.candidate_tool not in tool_descriptions:
            raise ValueError(f"pair uses unknown tool: {pair.pair_id}")
        if pair.tool_text != f"passage: {tool_descriptions[pair.candidate_tool]}":
            raise ValueError(f"pair tool text does not match registry: {pair.pair_id}")

        if pair.pair_kind is PairKind.TOOL_POSITIVE:
            if sample.label != pair.candidate_tool:
                raise ValueError(f"positive pair has wrong tool: {pair.pair_id}")
        elif pair.pair_kind is PairKind.TOOL_NEGATIVE:
            if sample.label in {fallback_label, pair.candidate_tool}:
                raise ValueError(f"tool negative pair is invalid: {pair.pair_id}")
        elif pair.pair_kind is PairKind.FALLBACK_NEGATIVE:
            if sample.label != fallback_label:
                raise ValueError(f"fallback pair has a routed source: {pair.pair_id}")
            fallback_tool_counts[pair.candidate_tool] += 1
        pair_counts[pair.pair_kind] += 1
        pairs_by_sample[pair.sample_id].append(pair)

    if set(pairs_by_sample) != set(train_samples):
        raise ValueError("at least one train sample has no generated pairs")
    for sample_id, sample_pairs in pairs_by_sample.items():
        kinds = Counter(pair.pair_kind for pair in sample_pairs)
        candidates = [pair.candidate_tool for pair in sample_pairs]
        if train_samples[sample_id].label == fallback_label:
            if kinds != Counter({PairKind.FALLBACK_NEGATIVE: NEGATIVES_PER_QUERY}):
                raise ValueError(f"fallback pair count is invalid: {sample_id}")
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"fallback negatives are not distinct: {sample_id}")
        else:
            expected = Counter(
                {
                    PairKind.TOOL_POSITIVE: 1,
                    PairKind.TOOL_NEGATIVE: NEGATIVES_PER_QUERY,
                }
            )
            if kinds != expected:
                raise ValueError(f"tool pair count is invalid: {sample_id}")
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"tool pair candidates are not distinct: {sample_id}")

    expected_tools = set(tool_descriptions)
    if set(fallback_tool_counts) != expected_tools:
        raise ValueError("some tools have no fallback-negative examples")
    if max(fallback_tool_counts.values()) - min(fallback_tool_counts.values()) > 1:
        raise ValueError("fallback-negative examples are not balanced")
    return pair_counts, fallback_tool_counts


def _metadata_for(path: Path, sample_count: int) -> DataFileMetadata:
    return DataFileMetadata(
        path=_relative_to_project(path),
        sha256=_sha256(path),
        sample_count=sample_count,
    )


def _read_reusable_pairs(
    *,
    dataset_manifest: DatasetManifest,
    seed: int,
    tool_names: list[str],
    tools_sha256: str,
    train_samples: dict[str, ProcessedSample],
    tool_descriptions: dict[str, str],
    fallback_label: str,
) -> TrainingPairsManifest | None:
    manifest_path = _pairs_manifest_path(dataset_manifest.dataset_version)
    if not manifest_path.is_file():
        return None
    try:
        manifest = TrainingPairsManifest.model_validate_json(manifest_path.read_text())
    except (OSError, ValidationError):
        return None

    source_train = dataset_manifest.splits[DatasetSplit.TRAIN]
    if (
        manifest.dataset_version != dataset_manifest.dataset_version
        or manifest.dataset_sha256 != dataset_manifest.dataset_sha256
        or manifest.source_train_sha256 != source_train.sha256
        or manifest.tools_sha256 != tools_sha256
        or manifest.seed != seed
        or manifest.negatives_per_query != NEGATIVES_PER_QUERY
        or manifest.tool_names != tool_names
        or manifest.pairs.path
        != _relative_to_project(_pairs_path(dataset_manifest.dataset_version))
    ):
        return None

    pairs_path = PROJECT_ROOT / manifest.pairs.path
    if not pairs_path.is_file() or _sha256(pairs_path) != manifest.pairs.sha256:
        return None
    table = pq.read_table(pairs_path)
    if table.num_rows != manifest.pairs.sample_count:
        return None
    try:
        pair_counts, fallback_tool_counts = _validate_pair_table(
            table,
            train_samples,
            tool_descriptions,
            fallback_label,
        )
    except (ValidationError, ValueError):
        return None
    if (
        dict(pair_counts) != manifest.pair_counts
        or dict(fallback_tool_counts) != manifest.fallback_negative_tool_counts
    ):
        return None
    return manifest


def _print_summary(manifest: TrainingPairsManifest, *, reused: bool) -> None:
    action = "Reused" if reused else "Built"
    print(
        f"{action} {manifest.pairs.sample_count} training pairs for "
        f"{manifest.dataset_version}"
    )
    print(
        "  pair kinds: "
        + ", ".join(
            f"{kind.value}={manifest.pair_counts[kind]}" for kind in PairKind
        )
    )
    fallback_counts = manifest.fallback_negative_tool_counts
    print(
        "  fallback negatives per tool: "
        f"min={min(fallback_counts.values())}, max={max(fallback_counts.values())}"
    )
    print(f"  sha256={manifest.pairs.sha256}, path={manifest.pairs.path}")


def main() -> None:
    seed = _load_seed()
    tool_descriptions, fallback_label, tools_sha256 = _load_tool_registry()
    tool_names = list(tool_descriptions)
    dataset_manifest, train_table = _load_current_dataset()
    train_samples = {
        sample.sample_id: sample
        for sample in (
            ProcessedSample.model_validate(row) for row in train_table.to_pylist()
        )
    }
    reusable = _read_reusable_pairs(
        dataset_manifest=dataset_manifest,
        seed=seed,
        tool_names=tool_names,
        tools_sha256=tools_sha256,
        train_samples=train_samples,
        tool_descriptions=tool_descriptions,
        fallback_label=fallback_label,
    )
    if reusable is not None:
        _print_summary(reusable, reused=True)
        return

    pairs_table = _build_pairs(
        train_table,
        tool_descriptions,
        fallback_label,
        seed,
    )
    pair_counts, fallback_tool_counts = _validate_pair_table(
        pairs_table,
        train_samples,
        tool_descriptions,
        fallback_label,
    )
    pairs_path = _pairs_path(dataset_manifest.dataset_version)
    _write_parquet(pairs_table, pairs_path)
    manifest = TrainingPairsManifest(
        dataset_version=dataset_manifest.dataset_version,
        dataset_sha256=dataset_manifest.dataset_sha256,
        source_train_sha256=dataset_manifest.splits[DatasetSplit.TRAIN].sha256,
        tools_sha256=tools_sha256,
        seed=seed,
        negatives_per_query=NEGATIVES_PER_QUERY,
        tool_names=tool_names,
        pairs=_metadata_for(pairs_path, pairs_table.num_rows),
        pair_counts=dict(pair_counts),
        fallback_negative_tool_counts=dict(fallback_tool_counts),
    )
    _write_json(_pairs_manifest_path(dataset_manifest.dataset_version), manifest)
    _print_summary(manifest, reused=False)


if __name__ == "__main__":
    main()
