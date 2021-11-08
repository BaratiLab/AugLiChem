---
layout: page
title: Installation
---

Welcome to the installation guide. First off, AugLiChem is a `python3.8+` package.
Guides for installation in Linux (Ubuntu) and MacOS ARM64 (M1 chip) architecture are given.
Installation for Linux is straightforward, but for MacOS ARM64, installation is more involved due to dependencies not yet being supported for the architecture.

### Linux

It is recommended to use an environment manager such as conda to install AugLiChem.
Instructions can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
If using conda, creating a new environment is ideal and can be done simply by running the following command:

`conda create -n auglichem python=3.8`

Then activating the new environment with

`conda activate auglichem`

AugLiChem is built primarily with `pytorch` and that should be installed independently according to your system specifications.
After activating your conda environment, `pytorch` can be installed easily and instructions are found [here](https://pytorch.org/).


`torch_geometric` needs to be installed with `conda install pyg -c pyg -c conda-forge`.


Once you have `pytorch` and `torch_geometric` installed, installing AugLiChem can be done using PyPI:

`pip install auglichem`


### MacOS ARM64 Architecture

A more involved install is required to run on the new M1 chips since some of the packages do not have official support yet.
We are working on a more elegant solution given the current limitations.

First, download this repo.

If you do not have it yet,, conda for ARM64 architecture needs to be installed.
 This can be done with Miniforge (which contains conda installer) which is installed by following the guide [here](https://github.com/conda-forge/miniforge)

Once you have miniforge compatible with ARM64 architecture, a new environment with rdkit can be installed.
If you do not specify `python=3.8` it will default to `python=3.9.6` as of the time of writing this.

`conda create -n auglichem python=3.8 rdkit`

Now activate the environment:

`conda activate auglichem`

From here, individual packages can be installed:

`conda install -c pytorch pytorch`

`conda install -c fastchan torchvision`

`conda install scipy`

`conda install cython`

`conda install scikit-learn`

`pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cpu.html`

`pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html`

`pip install torch-geometric`

Before installing the package, you must go into `setup.py` in the main directory and comment out `rdkit-pypi` and `tensorboard` from the `install_requires` list since they are already installed.
Not commenting these packages out will result in an error during installation.
It is not supported through pip this way, and since it is already installed, it will throw an error.

Finally, run: 

`pip install .`

