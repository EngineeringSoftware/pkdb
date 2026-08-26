#!/bin/bash

#SBATCH -A <Your project>
#SBATCH -J pkdb_examinimd_evaluation
#SBATCH -o examinimd_job.out
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

# ------------------
# Main code section
# ------------------

if [ -z "$1" ] || [ -z "$2" ]; then
    printf "Script requires two paths: \n\t(1) apptainer directory (holds conda_git.sif)\n\t(2) pkdb root directory\n"
    exit 1
fi

APPTAINER_DIR="$(realpath "$1")"
PKDB_ROOT="$(realpath "$2")"

module load tacc-apptainer
pushd "$APPTAINER_DIR"
apptainer exec --nv --fakeroot conda_git.sif bash -c '
    set -e
    export CUDACXX=/opt/conda/bin/nvcc
    export CXX=/usr/bin/g++
    export CC=/usr/bin/gcc
    eval "$(conda shell.bash hook)"
    conda activate pkdb

    BENCH_ROOT="$(realpath "$1/benchmarks")"
    cd "$BENCH_ROOT"
    python run_examinimd_debuggers.py --spaces "DebugCuda, DebugOpenMP"
' _ "$PKDB_ROOT"

popd
echo "Done."
