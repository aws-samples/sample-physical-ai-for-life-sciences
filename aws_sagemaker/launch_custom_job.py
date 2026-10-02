#!/usr/bin/env python3
"""Create a secure SageMaker job for the Isaac Lab training image."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone


def _add_storage_and_security_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-s3-uri", required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--checkpoint-s3-uri", required=True)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument("--subnet-ids", nargs="+", required=True)
    parser.add_argument("--security-group-ids", nargs="+", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--environment", default="demo")
    parser.add_argument("--no-network-isolation", action="store_true")


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--volume-size-gb", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-runtime-seconds", type=int, default=14_400)
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=21_600)


def build_parser() -> argparse.ArgumentParser:
    """Build the SageMaker training-job argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"))
    parser.add_argument("--job-name")
    parser.add_argument("--submit", action="store_true")
    _add_storage_and_security_args(parser)
    _add_training_args(parser)
    return parser


def _stopping_condition(args: argparse.Namespace) -> dict[str, int]:
    condition = {"MaxRuntimeInSeconds": args.max_runtime_seconds}
    if args.spot:
        condition["MaxWaitTimeInSeconds"] = args.max_wait_seconds
    return condition


def _hyperparameters(args: argparse.Namespace) -> dict[str, str]:
    return {
        "steps": str(args.steps),
        "warmup_steps": str(args.warmup_steps),
        "num_envs": str(args.num_envs),
        "seed": str(args.seed),
        "device": "cuda:0",
        "checkpoint_interval_steps": "5000",
    }


def _job_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return args.job_name or f"labflow-isaaclab-{timestamp}"


def _output_config(args: argparse.Namespace) -> dict[str, str]:
    return {"S3OutputPath": args.output_s3_uri, "KmsKeyId": args.kms_key_id}


def _checkpoint_config(args: argparse.Namespace) -> dict[str, str]:
    return {
        "S3Uri": args.checkpoint_s3_uri,
        "LocalPath": "/opt/ml/checkpoints",
    }


def _resource_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "InstanceType": args.instance_type,
        "InstanceCount": args.instance_count,
        "VolumeSizeInGB": args.volume_size_gb,
        "VolumeKmsKeyId": args.kms_key_id,
    }


def _request(args: argparse.Namespace) -> dict[str, object]:
    return {
        "TrainingJobName": _job_name(args),
        "RoleArn": args.role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": args.image_uri,
            "TrainingInputMode": "File",
        },
        "InputDataConfig": [_input_channel(args.input_s3_uri)],
        "OutputDataConfig": _output_config(args),
        "CheckpointConfig": _checkpoint_config(args),
        "ResourceConfig": _resource_config(args),
        "StoppingCondition": _stopping_condition(args),
        "HyperParameters": _hyperparameters(args),
        "VpcConfig": _vpc_config(args),
        "EnableNetworkIsolation": not args.no_network_isolation,
        "EnableInterContainerTrafficEncryption": True,
        "EnableManagedSpotTraining": args.spot,
        "Environment": {"PYTHONUNBUFFERED": "1"},
        "Tags": _tags(args),
    }


def _input_channel(s3_uri: str) -> dict[str, object]:
    return {
        "ChannelName": "bootstrap",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": s3_uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
        "ContentType": "application/octet-stream",
        "InputMode": "File",
    }


def _vpc_config(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "Subnets": args.subnet_ids,
        "SecurityGroupIds": args.security_group_ids,
    }


def _tags(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {"Key": "Project", "Value": "labflow-dreamer"},
        {"Key": "Environment", "Value": args.environment},
        {"Key": "Owner", "Value": args.owner},
    ]


def _validate(args: argparse.Namespace) -> None:
    if not args.region:
        raise ValueError("--region or AWS_REGION is required")
    if args.spot and args.max_wait_seconds <= args.max_runtime_seconds:
        raise ValueError("Spot max wait must exceed max runtime")
    if args.no_network_isolation:
        message = "Network isolation disabled for runtime asset access"
        print(json.dumps({"level": "WARNING", "message": message}))


def _submit(args: argparse.Namespace, request: dict[str, object]) -> str:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    retry_config = Config(retries={"max_attempts": 5, "mode": "adaptive"})
    try:
        client = boto3.client(
            "sagemaker", region_name=args.region, config=retry_config
        )
        response = client.create_training_job(**request)
    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(f"SageMaker submission failed: {error}") from error
    return str(response["TrainingJobArn"])


def main() -> None:
    """Print or submit an Isaac Lab SageMaker training job request."""
    args = build_parser().parse_args()
    try:
        _validate(args)
        request = _request(args)
        print(json.dumps(request, indent=2))
        if not args.submit:
            print(json.dumps({"level": "INFO", "submitted": False}))
            return
        print(json.dumps({"TrainingJobArn": _submit(args, request)}))
    except (RuntimeError, ValueError) as error:
        print(json.dumps({"level": "ERROR", "message": str(error)}))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
