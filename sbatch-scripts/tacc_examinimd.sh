#!/bin/bash

#SBATCH -A <Paste your project here>
#SBATCH -J pkdb_examinimd_evaluation
#SBATCH -o examinimd_job.out
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

# ------------------
# Main code section
# ------------------

if [ -z "$1" ] && [ -z "$2" ]; then
    printf "Script requires two paths: \n\t(1) run_examinimd.py script path\n\t(2) main.py of examinimd benchmark.\n"
	exit 1
fi

EXAMINIMD_MAIN="$2"
EXAMINIMD_RUNNER="$1"

eval "$(conda shell.bash hook)"
conda activate pkdb

export OMP_NUM_THREADS="$(nproc)"
python "$EXAMINIMD_RUNNER" --spaces "OpenMP, DebugOpenMP, Cuda, DebugCuda" "$EXAMINIMD_MAIN"
