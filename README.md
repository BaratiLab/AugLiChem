# AugLiChem
<!--
[![Build Status](https://travis-ci.com/BaratiLab/AugLiChem.svg?token=JCkBR1Zx861ey4B3mNiz&branch=main)](https://travis-ci.com/BaratiLab/AugLiChem)
[![codecov](https://codecov.io/gh/BaratiLab/AugLiChem/branch/main/graph/badge.svg?token=p5hPdWXEW1)](https://codecov.io/gh/BaratiLab/AugLiChem)
-->

Welcome to AugLiChem!
The augmentation library for chemical systems.
This package supports augmentation for both crystaline and molecular systems, as well as provides automatic downloading for our benchmark datasets, and easy to use model implementations.
In depth documentation about how to use AugLiChem, make use of transformations, and train models is given on our [website](https://baratilab.github.io/AugLiChem/).


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


Once you have `pytorch` and `torch_geometric` installed, installing AugLiChem can be done using PyPI:

`pip install auglichem`


### MacOS ARM64 Architecture

A more involved install is required to run on the new M1 chips since some of the packages do not     have official support yet.
We are working on a more elegant solution given the current limitations.

First, download this repo.

If you do not have it yet,, conda for ARM64 architecture needs to be installed.
 This can be done with Miniforge (which contains conda installer) which is installed by following     the guide [here](https://github.com/conda-forge/miniforge)

Once you have miniforge compatible with ARM64 architecture, a new environment with rdkit can be i    nstalled.
If you do not specify `python=3.8` it will default to `python=3.9.6` as of the time of writing th    is.

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

Before installing the package, you must go into `setup.py` in the main directory and comment out     `rdkit-pypi` and `tensorboard` from the `install_requires` list since they are already installed.
Not commenting these packages out will result in an error during installation.

Finally, run:

`pip install .`


Usage guides are provided in the `examples/` directory and provide useful guides for using both the molecular and crystal sides of the package.
Make sure to install `jupyter` before working with examples, using `conda install jupyter`.
After installing the package as described above, the example notebooks can be downloaded separately and run locally.

## Authors

Rishikesh Magar\*, Yuyang Wang\*, Cooper Lorsung\*, Hariharan Ramasubramanian, Chen Liang, Peiyuan Li, Amir Barati Farimani

\*Equal contribution
<!-- \*Department of Mechanical Engineering, Carnegie Mellon University, Pittsburgh, PA 15213 -->

## Paper

Our paper can be found [here](https://iopscience.iop.org/article/10.1088/2632-2153/ac9c84)

## Citation

If you use AugLiChem in your work, please cite:

```
@article{Magar_2022,
doi = {10.1088/2632-2153/ac9c84},
url = {https://dx.doi.org/10.1088/2632-2153/ac9c84},
year = {2022},
month = {nov},
publisher = {IOP Publishing},
volume = {3},
number = {4},
pages = {045015},
author = {Rishikesh Magar and Yuyang Wang and Cooper Lorsung and Chen Liang and Hariharan Ramasubramanian and Peiyuan Li and Amir Barati Farimani},
title = {AugLiChem: data augmentation library of chemical structures for machine learning},
journal = {Machine Learning: Science and Technology},
abstract = {Machine learning (ML) has demonstrated the promise for accurate and efficient property prediction of molecules and crystalline materials. To develop highly accurate ML models for chemical structure property prediction, datasets with sufficient samples are required. However, obtaining clean and sufficient data of chemical properties can be expensive and time-consuming, which greatly limits the performance of ML models. Inspired by the success of data augmentations in computer vision and natural language processing, we developed AugLiChem: the data augmentation library for chemical structures. Augmentation methods for both crystalline systems and molecules are introduced, which can be utilized for fingerprint-based ML models and graph neural networks (GNNs). We show that using our augmentation strategies significantly improves the performance of ML models, especially when using GNNs. In addition, the augmentations that we developed can be used as a direct plug-in module during training and have demonstrated the effectiveness when implemented with different GNN models through the AugliChem library. The Python-based package for our implementation of Auglichem: Data augmentation library for chemical structures, is publicly available at: https://github.com/BaratiLab/AugLiChem.},highly accurate ML models for chemical str$
}
```

## License
AugLiChem is MIT licensed, as found in the [LICENSE](https://github.com/BaratiLab/AugLiChem/blob/main/LICENSE) file. Please note that some of the dependencies AugLiChem uses may be licensed under different terms.

