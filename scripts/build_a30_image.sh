#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${KERNELAGENT_IMAGE:-kernelagent-a30:latest}"

docker build \
  --file "$REPO_ROOT/docker/Dockerfile" \
  --tag "$IMAGE_NAME" \
  "$REPO_ROOT"
