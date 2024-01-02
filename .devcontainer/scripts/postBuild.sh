# For writing commands that will be executed after the container is created

# Installs dev dependencies
pip install nox pre-commit jupyter-book

# Installs `caustics` as local library without resolving dependencies (--no-deps)
pip install -e ".[dev]"
