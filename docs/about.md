---
layout: page
title: About
permalink: /about/
---

AugLiChem provides models, data sets, and augmentation functions in order to make machine learning for molecular and crystalline systems simple and easy.
Models and data wrappers are built with PyTorch and take advantage of CPU and GPU support for faster execution.

## Package Structure

AugLiChem has two submodules: crystal and molecule.
The directory structure has been designed so using either submodule is as easy as switching `crystal` to `molecule` in your code, when possible.

The directory structure is as follows:

```
AugLiChem/
  -auglichem/
    -crystal/
      -__init__.py  -_compositions.py  -_transforms.py
      data/
        -__init__.py  -_load_sets.py  -_crystal_dataset.py
      -models/
        -__init__.py  -cgcnn.py  -gin.py  -schnet.py
    -molecule/
      -__init__.py  -_compositions.py  -_transforms.py
      -data/
        -__init__.py  -_load_sets.py  -_molecule_dataset.py
      -models/
        -__init__.py  -afp.py  -deepgcn.py  -gcn.py  -gine.py
    -test/
      -test_crystal.py  -test_molecule.py  -test_utils.py
    -utils/
      -__init__.py  -_constants.py  -_splitting.py
  -.travis.yml
  -LICENSE
  -README.md
  -setup.py
```

