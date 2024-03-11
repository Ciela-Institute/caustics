ARG CUDA_VERSION=11.8.0
FROM mambaorg/micromamba:focal-cuda-${CUDA_VERSION}

ARG CAUSTICS_VERSION=0.7.0

# Tell apt-get to not block installs by asking for interactive human input
ENV DEBIAN_FRONTEND=noninteractive \
    # Setup locale to be UTF-8, avoiding gnarly hard to debug encoding errors
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8 \
    # Set caustics directory home
    # this will be empty if not in dev mode
    CAUSTICS_HOME=/opt/caustics \
    CAUSTICS_TEST_SCRIPT=/usr/local/bin/run-tests

# Switch over to root user to install apt-get packages
USER root

# Setup caustics directory in case we are working with
# dev repo
RUN echo "Setup caustics directory..." \
    && mkdir ${CAUSTICS_HOME}

# Copying caustics over
COPY --chown=$MAMBA_USER:$MAMBA_USER . ${CAUSTICS_HOME}

# Make everything in opt rootless
RUN chown -R ${MAMBA_USER}:${MAMBA_USER} /opt

# Copy over run test script for the case we are working with dev repo
COPY --chown=$MAMBA_USER:$MAMBA_USER ./resources/docker/run-tests.sh ${CAUSTICS_TEST_SCRIPT}
RUN chmod +x ${CAUSTICS_TEST_SCRIPT} \
    && ln -s ${CAUSTICS_TEST_SCRIPT} /bin/run-tests

# Install basic apt packages
RUN echo "Installing Apt-get packages..." \
    && apt-get update --fix-missing > /dev/null \
    && apt-get install -y apt-utils \
    && apt-get install -y \
    git \
    wget \
    zip \
    tzdata \
    vim \
    tmux \
    python3-venv \
    graphviz > /dev/null \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to mamba user after all apt-get packages are installed
USER $MAMBA_USER

# Copy over the conda lock file from host machine to docker build engine
COPY --chown=$MAMBA_USER:$MAMBA_USER ./resources/docker/conda-linux-64.lock /tmp/conda-linux-64.lock

# Install the packages listed in the conda lock file
RUN micromamba install --name base --yes --file /tmp/conda-linux-64.lock \
    && micromamba clean --all --yes

# Set to activate mamba environment, otherwise python will not be found
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install caustics from a branch or distribution
RUN echo "Installing caustics ..." \
    # Allow for "main" or "dev" to be used as a version
    # or if it's pull request 'merge'
    ; if [[ $CAUSTICS_VERSION = *"main"* || $CAUSTICS_VERSION = *"dev"* || $CAUSTICS_VERSION = *"merge"* ]]; then \
    echo "Installing development caustics: ${CAUSTICS_VERSION}..." ; \
    pip install -e "${CAUSTICS_HOME}[dev]" \
    ; else echo "Installing from production distribution version: ${CAUSTICS_VERSION}" ; \
    pip install caustics==${CAUSTICS_VERSION} \
    ; fi
