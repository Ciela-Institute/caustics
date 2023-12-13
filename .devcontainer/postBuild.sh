# For writing commands that will be executed after the container is created

# Installs dev dependencies
${NB_PYTHON_PREFIX}/bin/pip install nox pre-commit jupyter-book

# Installs `caustics` as local library without resolving dependencies (--no-deps)
${NB_PYTHON_PREFIX}/bin/pip install --no-deps -e ".[dev]"
