#!/bin/bash

#SBATCH -J pkdb_boltzmann_evaluation # Job name
#SBATCH -o boltzmann_job.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                         # Total number of nodes requested
#SBATCH -n 4                         # Total number of mpi tasks requested
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi3008x                   # Desired partition

# ------------------
# Main code section
# ------------------

if [ -z "$1" ] && [ -z "$2" ]; then
    printf "Script requires two paths: \n\t(1) run_boltzmann.py script path\n\t(2) boltzmann.py script of boltzmann benchmark.\n"
	exit 1
fi

BOLTZMANN_RUNNER="$1"
BOLTZMANN_MAIN="$2"

eval "$(conda shell.bash hook)"
conda activate pkdb

export CXX=hipcc
export OMP_NUM_THREADS="$(nproc)"
python "$BOLTZMANN_RUNNER" --spaces "OpenMP, DebugOpenMP, HIP, DebugHIP" "$BOLTZMANN_MAIN"
