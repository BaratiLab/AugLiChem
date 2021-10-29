# AugLiChem
[![Build Status](https://travis-ci.com/BaratiLab/AugLiChem.svg?token=JCkBR1Zx861ey4B3mNiz&branch=main)](https://travis-ci.com/BaratiLab/AugLiChem)
[![codecov](https://codecov.io/gh/BaratiLab/AugLiChem/branch/main/graph/badge.svg?token=p5hPdWXEW1)](https://codecov.io/gh/BaratiLab/AugLiChem)

Welcome to AugLiChem!
The augmentation library for chemical systems.
This package supports augmentation for both crystaline and molecular systems, as well as provides automatic downloading for our benchmark datasets, and easy to use model implementations.


## Installation

AugLiChem is a `python3.8+` package.

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


After pytorch, torch-sparse and torch-scatter need to be installed according to your hardware, with guide here: [torch-sparse](https://github.com/rusty1s/pytorch_sparse).


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

`conda create -n auglichem python=3.8 rdkit'

Now activate the environment:

`conda activate auglichem`

From here, individual packages can be installed:

`conda install -c pytorch pytorch`

`conda install -c fastchan torchvision`

`conda install scipy`

`pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html`

`conda install cython`

`conda install scikit-learn`

Before installing the package, you must go into `setup.py` and comment out `rdkit-pypi` from the `install\_requires` list.
It is not supported through pip this way, and since it is already installed, it will throw an error.

Finally, navigate to where you downloaded the repo and run: 

`pip install .`


## Use

Usage guides are provided in the `examples/` directory and provide useful guides for using both the molecular and crystal sides of the package.
Make sure to install `jupyter` before working with examples, using `conda install jupyter`.

## Authors

## Paper

Our paper can be found at [PAPER URL]

## Citation

(Once on ArXiv add BibTex citation)

## License

