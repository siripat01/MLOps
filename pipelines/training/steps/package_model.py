from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from zenml import step


@dataclass(frozen=True)
class PackagedModel:
    archive_path: Path
    manifest_path: Path
    model_sha256: str
    archive_sha256: str
    model_name: str
    model_version: str


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _autogluon_version() -> str:
    try:
        return importlib.metadata.version("autogluon.timeseries")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def package_model(
    model_path: Path,
    *,
    model_name: str,
    model_version: str,
    artifact_root: Path,
    prediction_length: int = 16,
    monitoring_baseline: dict[str, dict[str, float]] | None = None,
) -> PackagedModel:
    source = Path(model_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"AutoGluon predictor directory does not exist: {source}")
    if not model_version.strip():
        raise ValueError("model_version must not be empty")

    artifact_root.mkdir(parents=True, exist_ok=True)
    work_dir = artifact_root / f"{model_name}-{model_version}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    model_target = work_dir / "model"
    shutil.copytree(source, model_target)

    manifest = {
        "model_name": model_name,
        "model_version": model_version,
        "framework": "autogluon-timeseries",
        "autogluon_version": _autogluon_version(),
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "artifact_sha256": _sha256_directory(model_target),
        "prediction_length": prediction_length,
        "monitoring_baseline": monitoring_baseline or {},
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    archive_path = artifact_root / f"{model_name}-{model_version}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(model_target, arcname="model")
        archive.add(manifest_path, arcname="manifest.json")

    return PackagedModel(
        archive_path=archive_path,
        manifest_path=manifest_path,
        model_sha256=manifest["artifact_sha256"],
        archive_sha256=_sha256_file(archive_path),
        model_name=model_name,
        model_version=model_version,
    )


@step(enable_cache=False)
def package_model_step(
    model_path: Path,
    model_name: str,
    model_version: str,
    artifact_root: str = "artifacts/model-publication",
    quality_gate_passed: bool = True,
    prediction_length: int = 16,
    monitoring_baseline: dict[str, dict[str, float]] | None = None,
) -> tuple[
    Annotated[Path, "archive_path"],
    Annotated[str, "model_version"],
    Annotated[str, "archive_sha256"],
]:
    if not quality_gate_passed:
        raise RuntimeError("Refusing to package model because quality gate failed")
    packaged = package_model(
        model_path,
        model_name=model_name,
        model_version=model_version,
        artifact_root=Path(artifact_root),
        prediction_length=prediction_length,
        monitoring_baseline=monitoring_baseline,
    )
    # Return the archive as a ZenML Path artifact. A raw path embedded in a
    # dataclass points to the previous step's container and is not portable.
    return packaged.archive_path, packaged.model_version, packaged.archive_sha256
