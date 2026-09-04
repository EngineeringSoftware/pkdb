#!/bin/bash
# build apptainer container if necessary

if [ "$#" -gt 1 ]; then
    printf "Usage: source ensure_apptainer.sh [apptainer image or directory]\n" >&2
    exit 1
fi

PKDB_TACC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PKDB_DEF_FILE="$PKDB_TACC_DIR/pkdb_env.def"
PKDB_DEFAULT_IMAGE="${PKDB_APPTAINER_IMAGE:-$PKDB_TACC_DIR/pkdb_env.sif}"

PKDB_ROOT="$(realpath "$PKDB_TACC_DIR/../..")"
if [ ! -d "$PKDB_ROOT/benchmarks" ]; then
    printf "No benchmarks/ under inferred pkdb root: %s\n" "$PKDB_ROOT" >&2
    exit 1
fi

APPTAINER_PATH=""
if [ -d "$1" ]; then
    pkdb_images=("$1"/*.sif)
    if [ ! -f "${pkdb_images[0]}" ]; then
        printf "No .sif image in directory: %s\n" "$1" >&2
        exit 1
    fi
    if [ "${#pkdb_images[@]}" -ne 1 ]; then
        printf "More than one .sif image in directory: %s\n" "$1" >&2
        exit 1
    fi
    APPTAINER_PATH="$(realpath "${pkdb_images[0]}")"
elif [ -n "$1" ]; then
    if [ ! -f "$1" ]; then
        printf "No apptainer image at: %s\n" "$1" >&2
        exit 1
    fi
    APPTAINER_PATH="$(realpath "$1")"
elif [ -f "$PKDB_DEFAULT_IMAGE" ]; then
    printf "Reusing apptainer image %s\n" "$PKDB_DEFAULT_IMAGE" >&2
    APPTAINER_PATH="$(realpath "$PKDB_DEFAULT_IMAGE")"
elif [ ! -f "$PKDB_DEF_FILE" ]; then
    printf "No apptainer image given and no definition file at: %s\n" "$PKDB_DEF_FILE" >&2
    exit 1
else
    printf "Building %s from %s\n" "$PKDB_DEFAULT_IMAGE" "$PKDB_DEF_FILE" >&2
    apptainer build --fakeroot "$PKDB_DEFAULT_IMAGE" "$PKDB_DEF_FILE" >&2 || exit 1
    APPTAINER_PATH="$(realpath "$PKDB_DEFAULT_IMAGE")"
fi
