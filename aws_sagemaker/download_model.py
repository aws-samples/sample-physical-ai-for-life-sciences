#!/usr/bin/env python3
"""Download and safely extract a completed SageMaker training artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

MODEL_NAME = "labflow_isaac_dreamer.pt"
RETRY_CONFIG = Config(retries={"max_attempts": 5, "mode": "adaptive"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-job-name", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    return parser


def _model_artifact_uri(job: dict[str, object]) -> str:
    status = job.get("TrainingJobStatus")
    if status != "Completed":
        raise ValueError(f"Training job is {status!r}, not 'Completed'")
    artifacts = cast(dict[str, object], job.get("ModelArtifacts", {}))
    uri = artifacts.get("S3ModelArtifacts")
    if not isinstance(uri, str):
        raise ValueError("Completed job did not report an S3 model artifact")
    return uri


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 model artifact URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _validated_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise ValueError(f"Unsupported archive entry: {member.name}")
    return members


def _safe_extract(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    destination.mkdir(parents=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in _validated_members(archive):
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive entry: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _download(job_name: str, region: str, output_dir: Path) -> Path:
    sagemaker = boto3.client("sagemaker", region_name=region, config=RETRY_CONFIG)
    job = sagemaker.describe_training_job(TrainingJobName=job_name)
    artifact_uri = _model_artifact_uri(cast(dict[str, object], job))
    bucket, key = _parse_s3_uri(artifact_uri)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "model.tar.gz"
    boto3.client("s3", region_name=region, config=RETRY_CONFIG).download_file(
        bucket, key, str(archive_path)
    )
    model_dir = output_dir / "model"
    _safe_extract(archive_path, model_dir)
    model_path = model_dir / MODEL_NAME
    if not model_path.is_file():
        raise ValueError(f"Artifact does not contain {MODEL_NAME}")
    return model_path


def main() -> None:
    """Download the final model reported by a completed SageMaker job."""
    args = _build_parser().parse_args()
    try:
        model_path = _download(args.training_job_name, args.region, args.output_dir)
    except (BotoCoreError, ClientError, OSError, tarfile.TarError, ValueError) as error:
        print(json.dumps({"level": "ERROR", "message": str(error)}))
        raise SystemExit(1) from error
    print(json.dumps({"level": "INFO", "model_path": str(model_path)}))


if __name__ == "__main__":
    main()
