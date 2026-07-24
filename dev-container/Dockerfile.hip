FROM rocm/dev-ubuntu-22.04:7.2.4

ARG USERNAME=pkdb
ARG PASSWORD=pkdb

# Dependencies installation
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    build-essential \
    micro \
    nano \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Default user without sudo right
RUN useradd -m -u 1000 -s /bin/bash "${USERNAME}" && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd && \
    usermod -aG video,render "${USERNAME}"

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV CONDA_DIR=/home/${USERNAME}/miniconda3

# Install conda in the user's home directory
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p "${CONDA_DIR}" && \
    rm -f miniconda.sh

ENV PATH="${CONDA_DIR}/bin:${CONDA_DIR}/condabin:${PATH}"

# Accept conda TOS and ensure it's loaded in every shell
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda init bash

CMD ["/bin/bash"]
