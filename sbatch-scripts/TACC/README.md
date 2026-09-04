# TACC sbatch scripts and apptainer file

Submit from this directory - the job scripts infer the `pkdb` root from their own
location (two levels up) and build the container themselves, so they take no
arguments:

```bash
sbatch tacc_examinimd.sh
```

The image is built once from [pkdb_env.def](pkdb_env.def) as `pkdb_env.sif` next
to it and reused by later jobs. Set `PKDB_APPTAINER_IMAGE` to build/reuse it
somewhere else.

An image you built yourself can still be passed, as a `.sif` or as a directory
holding one - that is the only argument the scripts take:

```bash
module load tacc-apptainer
apptainer build pkdb_env.sif pkdb_env.def

sbatch tacc_examinimd.sh ./pkdb_env.sif
```

Submit from this directory or from the repository root: `sbatch` runs a copy of
the job script, so the scripts fall back to `SLURM_SUBMIT_DIR` (the directory
you ran `sbatch` in, set by Slurm) to find themselves in the repo.

Do `pkdb` installation of `PyKokkos` and `pkdb` once - the container carries
conda and CUDA, the `pkdb` environment lives in your home directory:

```bash
apptainer shell --nv --fakeroot pkdb_env.sif
Apptainer> cd [pkdb_dir]/pykokkos
Apptainer> conda create -n pkdb python=3.13 -y
Apptainer> conda env update -n pkdb -f base/environment.yml
Apptainer> conda activate pkdb
Apptainer> python install_base.py install --verbose -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=4 -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON -DENABLE_CUDA=ON
Apptainer> conda install -c conda-forge pybind11 patchelf -y
Apptainer> pip install -e .
Apptainer> cd [pkdb_dir]/pkdb
Apptainer> pip install -r requirements.txt
Apptainer> pip install -e .
```
