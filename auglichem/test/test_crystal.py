import sys
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

#if __name__ == '__main__':
    #test_crystal_data()
    #test_composition()
