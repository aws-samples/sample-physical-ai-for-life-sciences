#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="937566678454"
ECR_REPOSITORY="labflow-dreamer"
ISAACLAB_BASE_IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta1"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login \
      --username AWS \
      --password-stdin \
      "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build \
  --build-arg "ISAACLAB_BASE_IMAGE=${ISAACLAB_BASE_IMAGE}" \
  --file "${PROJECT_ROOT}/docker/Dockerfile.isaaclab-sagemaker" \
  --tag "${IMAGE_URI}" \
  "${PROJECT_ROOT}"
docker push "${IMAGE_URI}"

printf '%s\n' "${IMAGE_URI}"
