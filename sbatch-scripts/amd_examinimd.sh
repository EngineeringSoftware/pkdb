#!/bin/bash

#SBATCH -J pkdb_examinimd_evaluation # Job name
#SBATCH -o examinimd_job.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                         # Total number of nodes requested
#SBATCH -n 4                         # Total number of mpi tasks requested
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi3008x                   # Desired partition

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

export CXX=hipcc
export OMP_NUM_THREADS="$(nproc)"
python "$EXAMINIMD_RUNNER" --spaces "OpenMP, DebugOpenMP, HIP, DebugHIP" "$EXAMINIMD_MAIN"
