# For writing commands that will be executed after the container is created

# Installs `caustics` as local library without resolving dependencies (--no-deps)
python3 -m pip install -e /workspaces/caustics --no-deps
python3 -m pip install -e ".[dev]"
