import sys
import os
sys.path.append(sys.path[0][:-14])

import shutil
import numpy as np

import torch
from torch_geometric.data import Data as PyG_Data

import warnings
from tqdm import tqdm

from auglichem.crystal._transforms import (
        RotationTransformation,
        PerturbStructureTransformation,
        RemoveSitesTransformation,
        SupercellTransformation,
        TranslateSitesTransformation,
        CubicSupercellTransformation,
        PrimitiveCellTransformation,
        SwapAxesTransformation
)
from auglichem.crystal._compositions import Compose, OneOf
from auglichem.crystal.data import CrystalDataset, CrystalDatasetWrapper
from auglichem.crystal.models import GINet, SchNet
from auglichem.crystal.models import CrystalGraphConvNet as CGCNN

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

def test_crystal_data():
    '''
        Since automated downloading isn't supported yet, this can't be tested without
        uploading the data set
    '''
    # Check general implementation
    dataset = CrystalDatasetWrapper("lanthanides", batch_size=1)
    transform = [SupercellTransformation()]
    train, valid, test = dataset.get_data_loaders(transform=transform)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_cifs = 0
        for v in tqdm(train):
            pass
        for v in tqdm(valid):
            pass
        for v in tqdm(test):
            pass
    assert True

    dataset = CrystalDatasetWrapper("lanthanides", batch_size=1, kfolds=2)
    train, valid, test = dataset.get_data_loaders(transform=transform, fold=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_cifs = 0
        for v in tqdm(train):
            pass
        for v in tqdm(valid):
            pass
        for v in tqdm(test):
            pass
    assert True

    # Check for CGCNN now
    dataset = CrystalDatasetWrapper("lanthanides", batch_size=1, cgcnn=True)
    transform = [SupercellTransformation()]
    train, valid, test = dataset.get_data_loaders(transform=transform)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_cifs = 0
        for v in tqdm(train):
            pass
        for v in tqdm(valid):
            pass
        for v in tqdm(test):
            pass
    assert True

    dataset = CrystalDatasetWrapper("lanthanides", batch_size=1, cgcnn=True, kfolds=2)
    train, valid, test = dataset.get_data_loaders(transform=transform, fold=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_cifs = 0
        for v in tqdm(train):
            pass
        for v in tqdm(valid):
            pass
        for v in tqdm(test):
            pass
    assert True
    

def _check_id_prop_augment(path, transform):
    # Checking if all cifs have been perturbed

    # Get transformation names
    transformation_names = []
    for t in transform:
        if(isinstance(t, SupercellTransformation)):
            transformation_names.append("supercell")
        if(isinstance(t, PerturbStructureTransformation)):
            transformation_names.append("perturbed")

    # Get original ids
    ids = np.loadtxt(path + "/id_prop.csv", delimiter=',').astype(int)[:,0]

    # Make sure originals and all transformed cifs exist
    for i in ids:
        assert os.path.exists(path + "/{}.cif".format(i))
        for t in transformation_names:
            assert os.path.exists(path + "/{}_{}.cif".format(i,t))


def _check_train_transform(path, transform, fold):
    # Get transformation names
    transformation_names = []
    for t in transform:
        if(isinstance(t, SupercellTransformation)):
            transformation_names.append("supercell")
        if(isinstance(t, PerturbStructureTransformation)):
            transformation_names.append("perturbed")

    # Get train ids
    ids = np.loadtxt(path + "/id_prop_train_{}.csv".format(fold), delimiter=',').astype(int)[:,0]

    for i in ids:
        assert os.path.exists(path + "/{}.cif".format(i))
        for t in transformation_names:
            assert os.path.exists(path + "/{}_{}.cif".format(i,t))


def _check_repeats(idx1, idx2):
    for v in idx1:
        assert not(v[0] in idx2[:,0]) # Only checking if cif file id is repeated


def _check_completeness(path, fold):

    # Get train and validation files
    train_prop = np.loadtxt(path + "/id_prop_train_{}.csv".format(fold),
                            delimiter=',')
    valid_prop = np.loadtxt(path + "/id_prop_valid_{}.csv".format(fold),
                            delimiter=',')
    test_prop = np.loadtxt(path + "/id_prop_test_{}.csv".format(fold),
                            delimiter=',')

    # Concatenate and sort by cif id
    together = np.concatenate((train_prop, valid_prop, test_prop))
    reconstructed = together[np.argsort(together[:,0])]

    # Get original ids
    id_prop = np.loadtxt(path + "/id_prop.csv", delimiter=',')
    
    # Check they are equal
    assert np.array_equal(reconstructed, id_prop)


def test_k_fold():
    #assert True
    #TODO: Automatic data downloading - this works locally
    dataset = CrystalDatasetWrapper("lanthanides", kfolds=5,
                                    data_path="../../examples/data_download")
    transform = [SupercellTransformation()]

    # Check no repeated indices in train and valid
    train_loader, valid_loader, test_loader  = dataset.get_data_loaders(transform=transform,
                                                                        fold=0)
    _check_id_prop_augment(dataset.data_path, transform)
    _check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    _check_repeats(valid_loader.dataset.id_prop_augment, test_loader.dataset.id_prop_augment)
    _check_repeats(test_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    _check_completeness(train_loader.dataset.data_path, 0)
    _check_train_transform(train_loader.dataset.data_path, transform, 0)

    train_loader, valid_loader, test_loader  = dataset.get_data_loaders(transform=transform,
                                                                        fold=1)
    _check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    _check_completeness(train_loader.dataset.data_path, 1)
    _check_train_transform(train_loader.dataset.data_path, transform, 1)

    train_loader, valid_loader, test_loader  = dataset.get_data_loaders(transform=transform,
                                                                        fold=2)
    _check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    _check_completeness(train_loader.dataset.data_path, 2)
    _check_train_transform(train_loader.dataset.data_path, transform, 2)

    train_loader, valid_loader, test_loader  = dataset.get_data_loaders(transform=transform,
                                                                        fold=3)
    _check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    _check_completeness(train_loader.dataset.data_path, 3)
    _check_train_transform(train_loader.dataset.data_path, transform, 3)

    train_loader, valid_loader, test_loader  = dataset.get_data_loaders(transform=transform,
                                                                        fold=4)
    _check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    _check_completeness(train_loader.dataset.data_path, 4)
    _check_train_transform(train_loader.dataset.data_path, transform, 4)

    try:
        train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=5)
    except ValueError as error:
        assert error.args[0] == "Please select a fold < 5"

    # Remove directory
    #shutil.rmtree(dataset.data_path)

    dataset = CrystalDatasetWrapper("lanthanides", kfolds=2,
                                    data_path="../../examples/data_download")
    transform = [SupercellTransformation(), PerturbStructureTransformation()]

    # Check no repeated indices in train and valid
    train_loader, valid_loader, test_loader  = dataset.get_data_loaders(transform=transform,
                                                                        fold=0)
    _check_id_prop_augment(dataset.data_path, transform)
    _check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    _check_completeness(train_loader.dataset.data_path, 0)
    _check_train_transform(train_loader.dataset.data_path, transform, 0)

    train_loader, valid_loader, test_loader  = dataset.get_data_loaders(transform=transform,
                                                                        fold=1)
    _check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    _check_completeness(train_loader.dataset.data_path, 1)
    _check_train_transform(train_loader.dataset.data_path, transform, 1)

    # Remove directory
    #shutil.rmtree(dataset.data_path)


def test_rotation():
    transform = [RotationTransformation()]
    pass


def test_perturb_structure():
    transform = [PerturbStructureTransformation()]
    pass


def test_remove_sites():
    tranform = [RemoveSitesTransformation([0])]
    pass


def test_supercell():
    transform = [SupercellTransformation()]
    pass


def test_translate():
    transform = [TranslateSitesTransformation()]
    pass


def test_cubic_supercell():
    transform = [CubicSupercellTransformation()]
    pass


def test_primitive():
    transfomr = [PrimitiveCellTransformation()]
    pass


def test_swap_axes():
    transform = [SwapAxesTransformation()]
    pass
    

if __name__ == '__main__':
    #test_crystal_data()
    #test_composition()
    #test_k_fold()
    test_rotation()
    test_perturb_structure()
    test_remove_sites()
    test_supercell()
    test_translate()
    test_cubic_supercell()
    test_primitive()
    test_swap_axes()
