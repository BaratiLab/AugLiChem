# AugLiChem
[![Build Status](https://travis-ci.com/BaratiLab/AugLiChem.svg?token=JCkBR1Zx861ey4B3mNiz&branch=main)](https://travis-ci.com/BaratiLab/AugLiChem)
[![codecov](https://codecov.io/gh/BaratiLab/AugLiChem/branch/main/graph/badge.svg?token=p5hPdWXEW1)](https://codecov.io/gh/BaratiLab/AugLiChem)

Welcome to AugLiChem!
The augmentation library for chemical systems.
This package supports augmentation for both crystaline and molecular systems, as well as provides automatic downloading for our benchmark datasets, and easy to use model implementations.


## Installation

AugLiChem is a `python3.8+` package.

It is recommended to use an environment manager such as conda to install AugLiChem.
Instructions can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
If using conda, creating a new environment is ideal and can be done simply by running the following command:

`conda create -n "auglichem" python=3.8; conda activate auglichem`

AugLiChem is built primarily with `pytorch` and that should be installed independently according to your system specifications.
After activating your conda environment, `pytorch` can be installed easily and instructions are found [here](https://pytorch.org/).

Once you have `pytorch` installed, installing AugLiChem can be done simply using PyPI:

`pip install auglichem`

**Note:** Installation may take a while due to pymatgen and torch-sparse dependencies.


## Use

Usage guides are provided in the `examples/` directory and provide useful guides for using both the molecular and crystal sides of the package.

## Authors

## Paper

Our paper can be found at [PAPER URL]

## Citation

(Once on ArXiv add BibTex citation)

## License

