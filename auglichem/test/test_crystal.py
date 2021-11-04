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
from auglichem.crystal.data import CrystalDataset, CrystalDatasetWrapper
from auglichem.crystal.models import GINet, SchNet
from auglichem.crystal.models import CrystalGraphConvNet as CGCNN

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

from pymatgen.core import Structure, Lattice, Molecule


def test_data_download():
    datasets = ["lanthanides", "band_gap", "perovskites", "formation_energy", "fermi_energy"]
    dir_names = dict(zip(datasets, ["lanths","band","abx3_cifs","FE","fermi"]))
    for d in datasets:
        dataset = CrystalDatasetWrapper(d, batch_size=1)
        assert os.path.isdir("./data_download/{}".format(dir_names[d]))
    shutil.rmtree("./data_download")
    

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
    dataset = CrystalDatasetWrapper("lanthanides", kfolds=5,
                                    data_path="./data_download")
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
    shutil.rmtree(dataset.data_path)

    dataset = CrystalDatasetWrapper("lanthanides", kfolds=2,
                                    data_path="./data_download")
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
    shutil.rmtree(dataset.data_path)


def test_rotation():

    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                  beta=90, gamma=60)
    struct = Structure(lattice, ["Si", "Si"], coords)
    
    transform = RotationTransformation([1,1,1], 360)
    struct1 = transform.apply_transformation(struct)
    assert struct == struct1

    transform = RotationTransformation([1,0,0], 180)
    struct2 = transform.apply_transformation(struct)
    assert np.allclose(struct.lattice.matrix[1], -struct2.lattice.matrix[1], atol=1e-8)
    assert np.allclose(struct.lattice.matrix[2], -struct2.lattice.matrix[2], atol=1e-8)


def test_perturb_structure():

    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                  beta=90, gamma=60)
    struct = Structure(lattice, ["Si", "Si"], coords)
    
    transform = PerturbStructureTransformation(distance=1, min_distance=0.1)
    struct1 = transform.apply_transformation(struct.copy(), seed=1)

    struct = Structure(lattice, ["Si", "Si"], coords)
    struct2 = transform.apply_transformation(struct.copy(), seed=1)

    struct = Structure(lattice, ["Si", "Si"], coords)
    struct3 = transform.apply_transformation(struct.copy(), seed=2)

    assert struct1 == struct2
    assert not(struct1 == struct3)
    assert not(struct2 == struct3)

    # We have to trust Pymatgen's implementation is correct here since so much is under the
    # hood for us. No distance to perturb results in an identical structure.
    transform = PerturbStructureTransformation(distance=0)
    struct = Structure(lattice, ["Si", "Si"], coords)
    struct1 = transform.apply_transformation(struct.copy(), seed=2)

    assert struct == struct1


def test_remove_sites():

    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                  beta=90, gamma=60)
    struct = Structure(lattice, ["Si", "Si"], coords)
    
    # Remove first site
    transform = RemoveSitesTransformation([0])
    struct1 = transform.apply_transformation(struct)
    assert len(struct1.sites) == 1
    assert struct.sites[1] == struct1.sites[0]

    # Remove second site
    struct = Structure(lattice, ["Si", "Si"], coords)
    transform = RemoveSitesTransformation([1])
    struct2 = transform.apply_transformation(struct)
    assert len(struct2.sites) == 1
    assert struct.sites[0] == struct2.sites[0]


def test_supercell():
    transform = SupercellTransformation([[2,0,0],[0,2,0],[0,0,2]])

    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                  beta=90, gamma=60)
    struct = Structure(lattice, ["Si", "Si"], coords)
    
    struct1 = transform.apply_transformation(struct)
    assert len(struct1.sites) == 8*len(struct.sites)
    assert np.allclose(struct1.lattice.abc, np.multiply(struct.lattice.abc,2), atol=1e-8)

    #TODO is more in depth testing here needed considering we use pymatgen's built-in function?


def test_translate():
    transform = TranslateSitesTransformation(indices_to_move=[0], translation_vector=[1,1,1],
                                             vector_in_frac_coords=True)

    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                  beta=90, gamma=60)
    struct = Structure(lattice, ["Si", "Si"], coords)
    
    # No translation
    struct1 = transform.apply_transformation(struct)
    assert struct1 == struct

    # No fractional coordinates
    transform = TranslateSitesTransformation(indices_to_move=[0], translation_vector=[0.2,0.2,0.2],
                                             vector_in_frac_coords=False)
    struct = Structure(lattice, ["Si", "Si"], coords)
    struct2 = transform.apply_transformation(struct)
    assert np.allclose(struct2.sites[0].coords, [0.2, 0.2, 0.2], atol=1e-8)

    # Fractional coordinates
    transform = TranslateSitesTransformation(indices_to_move=[0],
                                             translation_vector=[0.2,0,0],
                                             vector_in_frac_coords=True)
    struct = Structure(lattice, ["Si", "Si"], coords)
    struct3 = transform.apply_transformation(struct)
    coord1 = struct.lattice.abc[0] * 0.2
    assert np.allclose(struct3.sites[0].coords, [coord1,0,0], atol=1e-8)


#def test_cubic_supercell():
#    transform = CubicSupercellTransformation()
#
#    coords = [[0, 0, 0], [0.75,0.5,0.75]]
#    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
#                                  beta=90, gamma=60)
#    struct = Structure(lattice, ["Si", "Si"], coords)
#    
#    print(struct)
#    struct1 = transform.apply_transformation(struct)
#    print(struct)
#    pass
#
#
#def test_primitive():
#    transform = PrimitiveCellTransformation()
#
#    coords = [[0, 0, 0], [0.75,0.5,0.75]]
#    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
#                                  beta=90, gamma=60)
#    struct = Structure(lattice, ["Si", "Si"], coords)
#    
#    print(struct)
#    struct = transform.apply_transformation(struct)
#    print(struct)
#    pass


def test_swap_axes():
    #TODO: Test consistent augment
    transform = SwapAxesTransformation()

    coords = [[0, 0, 0], [0.75,0.5,0.75]]
    lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                  beta=90, gamma=60)
    
    struct = Structure(lattice, ["Si", "Si"], coords)
    struct1 = transform.apply_transformation(struct, seed=1)
    struct2 = transform.apply_transformation(struct, seed=2)
    struct3 = transform.apply_transformation(struct, seed=1)

    assert struct1 != struct2
    assert struct1 == struct3

    struct4 = transform.apply_transformation(struct, _test_choice=[0,2])
    assert np.allclose(struct.sites[1].coords, struct1.sites[1].coords[::-1], atol=1e-8)

    struct5 = transform.apply_transformation(struct, _test_choice=[0,1])
    coords = struct.sites[1].coords
    coords5 = struct5.sites[1].coords

    assert np.isclose(coords[0], coords5[1], atol=1e-8)
    assert np.isclose(coords[1], coords5[0], atol=1e-8)
    assert np.isclose(coords[2], coords5[2], atol=1e-8)
    

#if __name__ == '__main__':
    #test_data_download()
    #test_k_fold()
    #test_rotation()
    #test_perturb_structure()
    ##test_remove_sites()
    #test_supercell()
    #test_translate()
    ##test_cubic_supercell()
    ##test_primitive()
    #test_swap_axes()
