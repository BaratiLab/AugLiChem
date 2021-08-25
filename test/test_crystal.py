import sys
sys.path.append(sys.path[0][:-4])

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
from auglichem.crystal.data import CrystalDataset
from auglichem.crystal.data._crystal_dataset import CrysData

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

def test_crystal_data():
    '''
        Since automated downloading isn't supported yet, this can't be tested without
        uploading the data set
    '''
    assert True
    #data = CrystalDataset("Lanthanides")
    #train, valid, test = data.get_data_loaders()
    #shutil.rmtree("./data_download")

    #for b, t in enumerate(train):
    #    print(b)

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
