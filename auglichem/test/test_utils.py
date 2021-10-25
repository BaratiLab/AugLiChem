import sys
sys.path.append(sys.path[0][:-14])

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
    dataset = MoleculeDatasetWrapper('BACE', split='scaffold', seed=0)

    # First split
    train, val, test = dataset.get_data_loaders()
    load_idxs = np.concatenate([dataset.train_idx, dataset.valid_idx, dataset.test_idx])

    # All splits are the same
    train, val, test = dataset.get_data_loaders()
    load_idxs1 = np.concatenate([dataset.train_idx, dataset.valid_idx, dataset.test_idx])

    all_idxs = list(range(len(dataset)))
    assert all(all_idxs == np.sort(load_idxs))
    assert all(all_idxs == np.sort(load_idxs1))
    assert all(load_idxs == load_idxs1)

    # Test crystal
    assert True

def test_random_split():
    # Test molecule
    dataset = MoleculeDatasetWrapper('BACE', split='random', seed=0)

    # First split
    train, val, test = dataset.get_data_loaders()
    load_idxs = np.concatenate([dataset.train_idx, dataset.valid_idx, dataset.test_idx])

    # Second split should be the same
    train, val, test = dataset.get_data_loaders()
    load_idxs1 = np.concatenate([dataset.train_idx, dataset.valid_idx, dataset.test_idx])

    all_idxs = list(range(len(dataset)))
    assert all(all_idxs == np.sort(load_idxs))
    assert all(all_idxs == np.sort(load_idxs1))
    assert all(load_idxs == load_idxs1)

    # Third split should be different
    dataset = MoleculeDatasetWrapper('BACE', split='random', seed=1)
    train, val, test = dataset.get_data_loaders()
    load_idxs2 = np.concatenate([dataset.train_idx, dataset.valid_idx, dataset.test_idx])
    assert any(load_idxs != load_idxs2)


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
