#!/bin/bash

#SBATCH -J pkdb_ewald_evaluation # Job name
#SBATCH -o ewald_job.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                         # Total number of nodes requested
#SBATCH -n 4                         # Total number of mpi tasks requested
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi3008x                   # Desired partition

# ------------------
# Main code section
# ------------------

if [ -z "$1" ] && [ -z "$2" ]; then
    printf "Script requires two paths: \n\t(1) run_ewald_debuggers.py script path\n"
    exit 1
fi

EWALD_RUNNER="$1"

eval "$(conda shell.bash hook)"
conda activate pkdb

export CXX=hipcc
python "$EWALD_RUNNER" --spaces "HIP, DebugHIP"
