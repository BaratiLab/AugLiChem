import sys
sys.path.append(sys.path[0][:-4])

import shutil
import numpy as np

import torch
from torch_geometric.data import Data as PyG_Data

from auglichem.crystal import *
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

if __name__ == '__main__':
    test_crystal_data()
