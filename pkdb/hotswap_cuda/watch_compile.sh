#!/bin/bash
# watch_compile.sh — Watch CUDA source files and recompile to PTX on change.
#
# Usage:
#   ./watch_compile.sh [source.cu] [sm_arch]
#
# Examples:
#   ./watch_compile.sh                    # watches main.cu, auto-detects arch
#   ./watch_compile.sh main.cu sm_89      # explicit arch

SRC="${1:-main.cu}"
ARCH="${2:-}"
PTX_DIR="./ptx_output"
NVCC="${CUDA_HOME:-/usr/local/cuda-12.9}/bin/nvcc"

detect_arch() {
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$cap" ]; then
        echo "sm_${cap/./}"
    else
        echo "sm_80"
    fi
}

if [ -z "$ARCH" ]; then
    ARCH=$(detect_arch)
    echo "Auto-detected GPU arch: $ARCH"
fi

mkdir -p "$PTX_DIR"
BASENAME=$(basename "${SRC%.cu}")

compile_ptx() {
    echo ""
    echo "=== Compiling: $SRC → $PTX_DIR/$BASENAME.ptx (arch=$ARCH) ==="
    "$NVCC" --ptx -arch="$ARCH" -o "$PTX_DIR/$BASENAME.ptx" "$SRC" 2>&1
    if [ $? -eq 0 ]; then
        echo "PTX ready: $PTX_DIR/$BASENAME.ptx"
        echo "Available kernels (mangled names for hs-load):"
        grep '\.entry' "$PTX_DIR/$BASENAME.ptx" | sed 's/.*\.entry[[:space:]]*/    /' | sed 's/(.*//'
        echo ""
    else
        echo "Compilation FAILED"
        echo ""
    fi
}

# Initial compile
compile_ptx

echo "Watching $SRC for changes...  (Ctrl+C to stop)"
echo ""

if command -v inotifywait &>/dev/null; then
    while inotifywait -q -e modify "$SRC" >/dev/null 2>&1; do
        compile_ptx
    done
else
    echo "(inotify-tools not found — falling back to 1-second polling)"
    echo "(Install with: sudo apt install inotify-tools)"
    LAST_MOD=$(stat -c %Y "$SRC" 2>/dev/null || echo 0)
    while true; do
        sleep 1
        CUR_MOD=$(stat -c %Y "$SRC" 2>/dev/null || echo 0)
        if [ "$CUR_MOD" != "$LAST_MOD" ]; then
            LAST_MOD="$CUR_MOD"
            compile_ptx
        fi
    done
fi
