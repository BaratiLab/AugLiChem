import sys
sys.path.append(sys.path[0][:-4])

from auglichem.utils import *

import shutil
import numpy as np

import torch
from torch_geometric.data import Data as PyG_Data

from auglichem.molecule import RandomAtomMask, RandomBondDelete, Compose, OneOf
from auglichem.molecule.data import MoleculeDataset, MoleculeDatasetWrapper

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def test_scaffold_split():
    # Test molecule
    dataset = MoleculeDatasetWrapper('BACE', split='scaffold')
    train, val, test = dataset.get_data_loaders()
    all_idxs = list(range(len(dataset)))
    load_idxs = np.sort(np.concatenate([dataset.train_idx, dataset.valid_idx, dataset.test_idx]))
    assert all(all_idxs == load_idxs)

    # Test crystal
    assert True

def test_random_split():
    # Test molecule
    dataset = MoleculeDatasetWrapper('BACE', split='random')
    train, val, test = dataset.get_data_loaders()
    all_idxs = list(range(len(dataset)))
    load_idxs = np.sort(np.concatenate([dataset.train_idx, dataset.valid_idx, dataset.test_idx]))
    assert all(all_idxs == load_idxs)

    # Test crystal
    assert True

def test_constants():
    assert ATOM_LIST == list(range(1,120))
    assert NUM_ATOM_TYPE == 119

    assert len(CHIRALITY_LIST) == 4
    assert NUM_CHIRALITY_TAG == 4

    assert len(BOND_LIST) == 5
    assert NUM_BOND_TYPE == 6

    assert len(BONDDIR_LIST) == 4
    assert NUM_BOND_DIRECTION == 4

#if __name__ == '__main__':
#    test_scaffold_split()
#    test_random_split()
