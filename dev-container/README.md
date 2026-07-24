# Docker containers for `pkdb` debugger

This directory contains Dockerfile(s) for different platforms and installs basic
requirements for each of them. After initial Docker container configuration user
can install `pkdb`.


---

The main requirement is to have installed
[Docker](https://docs.docker.com/engine/install/ubuntu/)>=29.4.2.

Additional platform-specific requirements:

## CUDA docker

You need to have
[`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
installed in order to have correct Docker builds. Step-by-step installation
process can be found in [official NVIDIA
documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian).
After installation, Docker [daemon should be reloaded](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
