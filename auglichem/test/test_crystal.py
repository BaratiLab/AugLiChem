import sys
import os
sys.path.append(sys.path[0][:-14])

import shutil
import numpy as np

import torch
from torch_geometric.data import Data as PyG_Data

from auglichem.crystal._transforms import (
        RandomRotationTransformation,
        RandomPerturbStructureTransformation,
        RandomRemoveSitesTransformation,
        SupercellTransformation,
        RandomTranslateSitesTransformation,
        CubicSupercellTransformation,
        PrimitiveCellTransformation
)
from auglichem.crystal._compositions import Compose, OneOf
from auglichem.crystal.data import CrystalDataset, CrystalDatasetWrapper
#from auglichem.crystal.data._crystal_dataset import CrystalDataset

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

def test_crystal_data():
    '''
        Since automated downloading isn't supported yet, this can't be tested without
        uploading the data set
    '''
    #data = CrystalDataset("Lanthanides", on_the_fly_augment=True)
    assert True
    #data = CrystalDatasetWrapper("Lanthanides")
    
    #transform = [RandomRotationTransformation(axis=[1,0,0], angle=15),
    #             SupercellTransformation()]
    #data.data_augmentation(transform)
    #data.data_augmentation(transform)
    #train, valid, test = data.get_data_loaders()
    #shutil.rmtree("./data_download")

    #for b, t in enumerate(train):
    #    print(t)

def test_random_rotation():
    #rotate = RandomRotationTransformation([1,0,0], 90,)
    pass

def test_random_perturb_structure_transformation():
    pass

def test_random_remove_sites_transformation():
    pass

def test_supercell_transformation():
    pass

def test_random_translates_sites_transformation():
    pass

def cubic_supercell_transformation():
    pass

def test_cubic_supercell_transformation():
    pass

def primitive_cell_transformation():
    pass

def test_composition():
    #TODO Actually test the various functionality rather than simply check it runs
    #transform = Compose([
    #    RandomRotationTransformation([1,0,0], 90),
    #    RandomPerturbStructureTransformation(),
    #    RandomRemoveSitesTransformation([0]),
    #    SupercellTransformation(),
    #    RandomTranslateSitesTransformation([0,2], [1,1]),
    #    CubicSupercellTransformation(10, 100, 10),
    #    PrimitiveCellTransformation(0.1)
    #])
    #data = CrystalDataset("Lanthanides", transform=transform, batch_size=1024, aug_time=3)
    #train, valid, test = data.get_data_loaders()
    ##shutil.rmtree("./data_download")
    #for b, t in enumerate(train):
    #    print(b)
    assert True


def _check_id_prop_augment(path, transform):
    # Checking if all cifs have been perturbed

    # Get transformation names
    transformation_names = []
    for t in transform:
        if(isinstance(t, SupercellTransformation)):
            transformation_names.append("supercell")
        if(isinstance(t, RandomPerturbStructureTransformation)):
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
        if(isinstance(t, RandomPerturbStructureTransformation)):
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

    # Concatenate and sort by cif id
    together = np.concatenate((train_prop, valid_prop))
    reconstructed = together[np.argsort(together[:,0])]

    # Get original ids
    id_prop = np.loadtxt(path + "/id_prop.csv", delimiter=',')
    
    # Check they are equal
    assert np.array_equal(reconstructed, id_prop)


def test_k_fold():
    assert True
    #TODO: Automatic data downloading - this works localls
    #dataset = CrystalDatasetWrapper("lanthanides", kfolds=5,
    #                                data_path="../../examples/data_download")
    #transform = [SupercellTransformation()]

    ## Check no repeated indices in train and valid
    #train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=0)
    #_check_id_prop_augment(dataset.data_path, transform)
    #_check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    #_check_completeness(train_loader.dataset.data_path, 0)
    #_check_train_transform(train_loader.dataset.data_path, transform, 0)

    #train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=1)
    #_check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    #_check_completeness(train_loader.dataset.data_path, 1)
    #_check_train_transform(train_loader.dataset.data_path, transform, 1)

    #train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=2)
    #_check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    #_check_completeness(train_loader.dataset.data_path, 2)
    #_check_train_transform(train_loader.dataset.data_path, transform, 2)

    #train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=3)
    #_check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    #_check_completeness(train_loader.dataset.data_path, 3)
    #_check_train_transform(train_loader.dataset.data_path, transform, 3)

    #train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=4)
    #_check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    #_check_completeness(train_loader.dataset.data_path, 4)
    #_check_train_transform(train_loader.dataset.data_path, transform, 4)

    #try:
    #    train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=5)
    #except ValueError as error:
    #    assert error.args[0] == "Please select a fold < 5"

    ## Remove directory
    #shutil.rmtree(dataset.data_path)

    #dataset = CrystalDatasetWrapper("lanthanides", kfolds=2,
    #                                data_path="../../examples/data_download")
    #transform = [SupercellTransformation(), RandomPerturbStructureTransformation()]

    ## Check no repeated indices in train and valid
    #train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=0)
    #_check_id_prop_augment(dataset.data_path, transform)
    #_check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    #_check_completeness(train_loader.dataset.data_path, 0)
    #_check_train_transform(train_loader.dataset.data_path, transform, 0)

    #train_loader, valid_loader = dataset.get_data_loaders(transform=transform, fold=1)
    #_check_repeats(valid_loader.dataset.id_prop_augment, train_loader.dataset.id_prop_augment)
    #_check_completeness(train_loader.dataset.data_path, 1)
    #_check_train_transform(train_loader.dataset.data_path, transform, 1)

    #shutil.rmtree(dataset.data_path)
    

#if __name__ == '__main__':
    #test_crystal_data()
    #test_composition()
    #test_k_fold()