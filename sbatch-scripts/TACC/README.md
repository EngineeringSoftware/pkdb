# TACC sbatch scripts and apptainer file

First, build container with apptainer:

```bash
module load tacc-apptainer
apptainer build pkdb_container.sif pkdb_container.def
```

Do `pkdb` installation of `PyKokkos` and `pkdb`:

```bash
apptainer shell --nv --fakeroot conda_git.sif
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

Now, when our environment is ready we can run sbatch scripts. Execute scripts
and pass apptainer directory (`./`, if you are doing it from current directory)
and `pkdb` directory (`../../` if you are doing it from current directory). It's
not required to pass absolute path.  
For example:

```bash
sbatch tacc_examinimd.sh ./ ../../
```
