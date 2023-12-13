FROM quay.io/pangeo/pytorch-notebook:latest

USER root
COPY --chown=jovyan:jovyan . /home/jovyan

# Install apt packages specified in a apt.txt file if it exists.
# Unlike repo2docker, blank lines nor comments are supported here.
RUN echo "Checking for 'apt.txt'..." \
    ; if test -f "apt.txt" ; then \
    apt-get update --fix-missing > /dev/null \
    # Read apt.txt line by line, and execute apt-get install -y for each line in apt.txt
    && xargs -a apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    ; fi

# If a conda-requirements.txt file exists, use conda to install packages listed there.
# After installing the packages, we cleanup some unnecessary files
# to try reduce image size - see https://jcristharif.com/conda-docker-tips.html
RUN echo "Checking for conda 'conda-requirements.txt'..." \
    ; if test -f "conda-requirements.txt" ; then \
    mamba install --yes --name ${CONDA_ENV} --file conda-requirements.txt \
    ; fi \
    && mamba clean -yaf \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete \
    ; if [ -d ${NB_PYTHON_PREFIX}/lib/python*/site-packages/bokeh/server/static ]; then \
    find ${NB_PYTHON_PREFIX}/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete \
    ; fi

# If a pip-requirements.txt file exists, use pip to install packages
# listed there. We don't want to save cached wheels in the image
# to avoid wasting space.
RUN echo "Checking for pip 'pip-requirements.txt'..." \
    ; if test -f "pip-requirements.txt" ; then \
    ${NB_PYTHON_PREFIX}/bin/pip install --no-cache -r pip-requirements.txt \
    ; fi

# If a start file exists, put that under /srv/start. Used in the
# same way as a start file in repo2docker.
RUN echo "Checking for 'start'..." \
    ; if test -f "start" ; then \
    chmod +x start \
    && cp start /srv/start \
    ; fi

# Set jovyan as the owner of everything under ${HOME}
RUN chown -R jovyan:jovyan ${HOME}

USER ${NB_USER}
WORKDIR ${HOME}

EXPOSE 8888
ENTRYPOINT ["/srv/start"]
