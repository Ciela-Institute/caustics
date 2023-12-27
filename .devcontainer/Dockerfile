FROM pangeo/base-notebook:latest

USER root

# Install apt packages specified in a apt.txt file if it exists.
# Unlike repo2docker, blank lines nor comments are supported here.
RUN apt-get update --fix-missing > /dev/null \
    # Read apt.txt line by line, and execute apt-get install -y for each line in apt.txt
    && apt-get install -y \
        lsb-release \
        wget \
        software-properties-common \
        gnupg \
        git \
        ncdu \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CLANG if version is specified
ARG CLANG_VERSION
RUN if [ -n "$CLANG_VERSION" ]; then \
    wget https://apt.llvm.org/llvm.sh; \
    chmod +x llvm.sh; \
    ./llvm.sh "${CLANG_VERSION}"; \
    echo 'export CC=clang' >> ~/.bashrc; \
    echo 'export CXX=clang++' >> ~/.bashrc; \
    apt update; \
    apt install -y clang; \
    apt install -y libomp-dev; \
    fi


# Install cuda if version is specified
ARG CUDA_VERSION
RUN if [ -n "$CUDA_VERSION" ]; then \
    mamba install --yes --name ${CONDA_ENV} -c "nvidia/label/cuda-${CUDA_VERSION}" \
    ; fi \
    && mamba clean -yaf \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete ;

# Set jovyan as the owner of everything under ${HOME}
RUN chown -R jovyan:jovyan ${HOME}

USER ${NB_USER}
