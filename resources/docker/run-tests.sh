#!/bin/bash -l

set -e

pytest -vvv ${CAUSTICS_HOME}/tests
