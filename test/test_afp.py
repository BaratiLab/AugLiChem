import sys
sys.path.append(sys.path[0][:-4])

import shutil
import numpy as np

import torch
from torch_geometric.data import Data as PyG_Data

from auglichem.molecule import RandomAtomMask, RandomBondDelete, Compose, OneOf
from auglichem.molecule.data import MoleculeDataset, MoleculeDatasetWrapper
from auglichem.molecule.models import AttentiveFP as AFP

def _check_eq(model, model1):
    for p1, p2 in zip(model.parameters(), model1.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def test_initialization():
    model = AFP(2, 10, 2, 2, 1, 1, seed=0)
    model1 = AFP(2, 10, 2, 2, 1, 1, seed=0)
    model2 = AFP(2, 10, 2, 2, 1, 1, seed=1)

    assert _check_eq(model, model1)
    assert not _check_eq(model, model2)


if __name__ == '__main__':
    test_initialization()
