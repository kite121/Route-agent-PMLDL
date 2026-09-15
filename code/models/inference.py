"""Load and use the packaged semantic tool router locally.

The loader accepts only the ``models/current/agent_router.tar.gz`` package
that matches its adjacent manifest. The API layer will create one ``Router``
instance on startup and call ``route`` for each request.

For a local smoke prediction after packaging:
    .venv/bin/python code/models/inference.py "Set an alarm for 7 tomorrow"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "current"
ARCHIVE_NAME = "agent_router.tar.gz"
MANIFEST_NAME = "model_manifest.json"
REQUIRED_ARCHIVE_PATHS = {"encoder", "tools.yaml", MANIFEST_NAME}


@dataclass(frozen=True)
class ToolScore:
    tool: str
    score: float


@dataclass(frozen=True)
class RoutingResult:
    decision: Literal["route", "fallback"]
    tool: str | None
    score: float
    top_k: list[ToolScore]
    model_version: str


@dataclass(frozen=True)
class ModelPackage:
    model_version: str
    base_model: str
    fallback_threshold: float
    tools_sha256: str
    artifact_path: Path
    extracted_dir: Path
    encoder_path: Path
    tools_path: Path
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_descriptions_sha256(path: Path) -> str:
    """Hash the canonical tool-description mapping used during training."""

    try:
        registry = yaml.safe_load(path.read_text())
        tools = registry["tools"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError("packaged tools.yaml is invalid") from error
    if not isinstance(tools, dict) or not tools:
        raise ValueError("packaged tools.yaml has no tools")
    descriptions: dict[str, str] = {}
    for name, specification in tools.items():
        description = specification.get("description") if isinstance(specification, dict) else None
        if not isinstance(name, str) or not isinstance(description, str) or not description.strip():
            raise ValueError("packaged tools.yaml has an invalid tool description")
        descriptions[name] = description.strip()
    canonical = json.dumps(
        dict(sorted(descriptions.items())),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is missing or invalid: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _required_string(payload: dict[str, object], key: str, *, description: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} is missing {key}")
    return value


def _manifest_value(
    manifest: dict[str, object],
    *,
    section: str,
    key: str,
) -> object:
    payload = manifest.get(section)
    if not isinstance(payload, dict) or key not in payload:
        raise ValueError(f"model manifest is missing {section}.{key}")
    return payload[key]


def _load_manifest(model_dir: Path) -> tuple[dict[str, object], Path, str, str, float, str]:
    manifest_path = model_dir / MANIFEST_NAME
    manifest = _read_json(manifest_path, description="model manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("model manifest has an unsupported schema version")
    model_version = _required_string(manifest, "model_version", description="model manifest")
    if not all(character.isalnum() or character in {"-", "_", "."} for character in model_version):
        raise ValueError("model manifest has an unsafe model_version")
    base_model = _required_string(manifest, "base_model", description="model manifest")
    artifact_name = _manifest_value(manifest, section="artifact", key="filename")
    encoder_path = _manifest_value(manifest, section="artifact", key="encoder_path")
    tools_path = _manifest_value(manifest, section="artifact", key="tool_registry_path")
    threshold = _manifest_value(manifest, section="routing", key="fallback_threshold")
    tools_sha256 = _manifest_value(manifest, section="lineage", key="tools_sha256")
    if artifact_name != ARCHIVE_NAME or encoder_path != "encoder" or tools_path != "tools.yaml":
        raise ValueError("model manifest has an unsupported artifact layout")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("model manifest has an invalid fallback threshold")
    if not -1.0 <= float(threshold) <= 1.0:
        raise ValueError("model manifest fallback threshold must be from -1 to 1")
    if (
        not isinstance(tools_sha256, str)
        or len(tools_sha256) != 64
        or any(character not in "0123456789abcdef" for character in tools_sha256.lower())
    ):
        raise ValueError("model manifest has an invalid tools_sha256")
    artifact_path = model_dir / ARCHIVE_NAME
    if not artifact_path.is_file():
        raise FileNotFoundError(f"model package is missing: {artifact_path}")
    return (
        manifest,
        artifact_path,
        model_version,
        base_model,
        float(threshold),
        tools_sha256,
    )


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination_resolved):
                raise ValueError("model archive contains an unsafe path")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("model archive must not contain links or device files")
        archive.extractall(destination)


def _verify_extracted_layout(extracted_dir: Path, manifest: dict[str, object]) -> None:
    encoder_path = extracted_dir / "encoder"
    tools_path = extracted_dir / "tools.yaml"
    archive_manifest_path = extracted_dir / MANIFEST_NAME
    if not encoder_path.is_dir() or not any(encoder_path.iterdir()):
        raise ValueError("extracted model archive has no encoder")
    if not tools_path.is_file() or not archive_manifest_path.is_file():
        raise ValueError("extracted model archive has an incomplete layout")
    archive_manifest = _read_json(archive_manifest_path, description="archived manifest")
    if archive_manifest != manifest:
        raise ValueError("archived manifest differs from models/current manifest")


def _extract_package(
    *,
    cache_dir: Path,
    artifact_path: Path,
    model_version: str,
    manifest: dict[str, object],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_sha256 = _sha256(artifact_path)
    extracted_dir = cache_dir / f"{model_version}-{archive_sha256[:12]}"
    if extracted_dir.exists():
        if not extracted_dir.is_dir():
            raise ValueError("model extraction cache path is not a directory")
        _verify_extracted_layout(extracted_dir, manifest)
        return extracted_dir

    with tempfile.TemporaryDirectory(dir=cache_dir) as temporary_dir:
        staging_dir = Path(temporary_dir)
        _safe_extract(artifact_path, staging_dir)
        _verify_extracted_layout(staging_dir, manifest)
        try:
            os.replace(staging_dir, extracted_dir)
        except FileExistsError:
            _verify_extracted_layout(extracted_dir, manifest)
    return extracted_dir


def _load_tool_registry(
    tools_path: Path,
    expected_sha256: str,
) -> tuple[list[str], str, list[str]]:
    if _tool_descriptions_sha256(tools_path) != expected_sha256:
        raise ValueError("packaged tool descriptions differ from model manifest")
    try:
        registry = yaml.safe_load(tools_path.read_text())
        tools = registry["tools"]
        fallback = registry["fallback"]["name"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise ValueError("packaged tools.yaml is invalid") from error
    if not isinstance(tools, dict) or not tools:
        raise ValueError("packaged tools.yaml has no tools")
    if not isinstance(fallback, str) or not fallback:
        raise ValueError("packaged tools.yaml has an invalid fallback")
    names = sorted(tools)
    if fallback in names:
        raise ValueError("fallback must not duplicate a tool name")
    descriptions: list[str] = []
    for name in names:
        specification = tools[name]
        if not isinstance(name, str) or not isinstance(specification, dict):
            raise ValueError("packaged tools.yaml has an invalid tool")
        description = specification.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"packaged tool {name} has no description")
        descriptions.append(f"passage: {description.strip()}")
    return names, fallback, descriptions


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Router:
    """A loaded bi-encoder and the immutable tool registry it was trained against."""

    def __init__(
        self,
        package: ModelPackage,
        *,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.package = package
        self.device = device or _select_device()
        self.tool_names, self.fallback_label, tool_texts = _load_tool_registry(
            package.tools_path,
            package.tools_sha256,
        )
        self.model = SentenceTransformer(str(package.encoder_path), device=self.device)
        self.model.eval()
        self.tool_embeddings = self.model.encode(
            tool_texts,
            batch_size=len(tool_texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )

    def model_info(self) -> dict[str, object]:
        return {
            "model_version": self.package.model_version,
            "base_model": self.package.base_model,
            "dataset_version": self.package.manifest["lineage"]["dataset_version"],
            "fallback_threshold": self.package.fallback_threshold,
            "tool_count": len(self.tool_names),
        }

    def route(self, text: str, *, top_k: int = 3) -> RoutingResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_embedding = self.model.encode(
            [f"query: {text.strip()}"],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )[0]
        scores = query_embedding @ self.tool_embeddings.T
        count = min(top_k, len(self.tool_names))
        indices = np.argsort(-scores)[:count]
        candidates = [
            ToolScore(tool=self.tool_names[index], score=float(scores[index]))
            for index in indices
        ]
        best = candidates[0]
        is_route = best.score >= self.package.fallback_threshold
        return RoutingResult(
            decision="route" if is_route else "fallback",
            tool=best.tool if is_route else None,
            score=best.score,
            top_k=candidates,
            model_version=self.package.model_version,
        )


def load_router(
    model_dir: Path = DEFAULT_MODEL_DIR,
    *,
    cache_dir: Path | None = None,
) -> Router:
    """Load a verified router and unpack it into a writable cache directory."""

    (
        manifest,
        artifact_path,
        model_version,
        base_model,
        threshold,
        tools_sha256,
    ) = _load_manifest(model_dir)
    extracted_dir = _extract_package(
        cache_dir=cache_dir or model_dir / ".extracted",
        artifact_path=artifact_path,
        model_version=model_version,
        manifest=manifest,
    )
    package = ModelPackage(
        model_version=model_version,
        base_model=base_model,
        fallback_threshold=threshold,
        tools_sha256=tools_sha256,
        artifact_path=artifact_path,
        extracted_dir=extracted_dir,
        encoder_path=extracted_dir / "encoder",
        tools_path=extracted_dir / "tools.yaml",
        manifest=manifest,
    )
    return Router(package)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local routing inference")
    parser.add_argument("text", help="User request to route")
    parser.add_argument("--top-k", type=int, default=3)
    arguments = parser.parse_args()
    result = load_router().route(arguments.text, top_k=arguments.top_k)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
