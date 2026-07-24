#!/bin/bash

#SBATCH -A <Paste your project here>
#SBATCH -J pkdb_ewald_evaluation
#SBATCH -o ewald_job.out
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

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

export OMP_NUM_THREADS="$(nproc)"
python "$BOLTZMANN_RUNNER" --spaces "OpenMP, DebugOpenMP, Cuda, DebugCuda" "$BOLTZMANN_MAIN"
