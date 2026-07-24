nvcc -Xcompiler -fPIC -arch=sm_80 kernels-new.cu -O3 \
 `python3 -m pybind11 --includes` \
 -shared -o cuda_red`python3.10-config --extension-suffix` \
 -L.
