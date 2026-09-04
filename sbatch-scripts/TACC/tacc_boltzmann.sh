#!/bin/bash

#SBATCH -A <Your project>
#SBATCH -J pkdb_boltzmann_evaluation
#SBATCH -o boltzmann_job.out
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

# ------------------
# Main code section
# ------------------

# sbatch runs a spooled copy of this file, so the submit directory is where we
# look for the helper when this path is not the one in the repo.
TACC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
for dir in "$TACC_DIR" "$SLURM_SUBMIT_DIR" "$SLURM_SUBMIT_DIR/sbatch-scripts/TACC"; do
    if [ -f "$dir/ensure_apptainer.sh" ]; then
        TACC_DIR="$dir"
        break
    fi
done

if [ ! -f "$TACC_DIR/ensure_apptainer.sh" ]; then
    printf "Cannot find ensure_apptainer.sh; submit from sbatch-scripts/TACC.\n" >&2
    exit 1
fi

module load tacc-apptainer

# Sets PKDB_ROOT (the repo above this directory) and APPTAINER_PATH (built from
# pkdb_env.def unless an image is passed).
source "$TACC_DIR/ensure_apptainer.sh" "$@"

apptainer exec --nv --fakeroot "$APPTAINER_PATH" bash -c '
    set -e
    export CUDACXX=/opt/conda/bin/nvcc
    export CXX=/usr/bin/g++
    export CC=/usr/bin/gcc
    eval "$(conda shell.bash hook)"
    conda activate pkdb

    export OMP_NUM_THREADS=72
    BENCH_ROOT="$(realpath "$1/benchmarks")"
    cd "$BENCH_ROOT"
    python run_boltzmann_debuggers.py --spaces "DebugCuda, DebugOpenMP"
' _ "$PKDB_ROOT"

echo "Done."
