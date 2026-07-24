#!/bin/bash
set -e

BACKEND="${1,,}"
if [[ "${BACKEND}" != "cuda" && "${BACKEND}" != "openmp" && "${BACKEND}" != "hip" ]]; then
    echo "Usage: $0 <cuda|openmp|hip>"
    exit 1
fi

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
CONTAINER_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_ROOT=$(dirname "$CONTAINER_DIR")

CONTAINER_NAME="pyk-${BACKEND}"
USERNAME="pkdb"
PASSWORD="pkdb"

echo "Building docker container: ${CONTAINER_NAME}"
docker build \
    -t "${CONTAINER_NAME}:latest" \
    -f "${CONTAINER_DIR}/Dockerfile.${BACKEND}" \
    "${PROJECT_ROOT}"

echo "Stopping ${CONTAINER_NAME}"
docker stop "${CONTAINER_NAME}" 2>/dev/null || true

echo "Removing ${CONTAINER_NAME}"
docker rm "${CONTAINER_NAME}" 2>/dev/null || true

RUN_ARGS=(-dit --security-opt seccomp=unconfined --name "${CONTAINER_NAME}" --restart unless-stopped)
[ "${BACKEND}" = "cuda" ] && RUN_ARGS+=(--gpus all)
[ "${BACKEND}" = "hip" ] && RUN_ARGS+=(--device=/dev/kfd --device=/dev/dri --group-add video --group-add render)

echo "Starting ${CONTAINER_NAME}"
docker run "${RUN_ARGS[@]}" "${CONTAINER_NAME}" bash

echo "##################"
echo "Container started."
echo "To access container from terminal, enter:"
echo "> docker exec -it -u ${USERNAME} ${CONTAINER_NAME} bash"
echo "Password: ${PASSWORD}"
echo "##################"
