# Caustics Docker Resources

This directory contains the files needed to create a GPU Optimized Docker image
for caustics. Currently, it is optimized for CUDA Version 11.8.0 only.

The following files can be found:

- `Dockerfile`: The docker specification file that contains the instructions on
  how to build the image using docker.
- `env.yaml`: The conda environment yaml file that list out the required
  packages to install for the python conda environment within the docker
  container when spun up.
- `conda-linux-64.lock`: The `linux-64` architecture specific conda lock file
  that specifies the exact urls to the conda packages. This was generated
  manually with the `conda-lock` program using the `env.yaml` file.

  ```bash
  conda-lock -f env.yaml --kind explicit -p linux-64
  ```

- `run-tests.sh`: A really simple script file that is accessible via command
  line within the container by calling `/bin/run-tests`. This file essentially
  runs `pytests`.

## Using the docker image within HPC

To use the docker image within the HPC, firstly, you will need to figure out
which image to use within the github registry. Then pull the image using
apptainer, which will create an apptainer singularity image format (.sif). Use
this image to run any commands with caustics environment already setup and
configured.

_Note: You will need to have the apptainer program installed on your system to
use the following commands. Additionally, please ignore any warning from
Apptainer when pulling the image._

```bash
apptainer pull docker://ghcr.io/ciela-institute/caustics:dev-cuda-11.8.0
apptainer run --nv caustics_dev-cuda-11.8.0.sif /bin/run-tests
```

In the example above, we are pulling the `dev-cuda-11.8.0` image from the github
registry and then running the `/bin/run-tests` script within the container to
run all of the tests for caustics. The `--nv` flag is used to enable GPU access
within the container.
