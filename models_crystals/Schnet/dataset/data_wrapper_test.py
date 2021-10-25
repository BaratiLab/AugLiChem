import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn import preprocessing

import ase
# from ase.io import cif
from ase.io import read as ase_read
from pymatgen.core.structure import Structure

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_geometric.data import Data, Dataset, DataLoader
from torch_cluster import knn_graph

from dataset.atom_feat import AtomCustomJSONInitializer
from dataset.augmentation import RotationTransformation, \
    PerturbStructureTransformation, SwapAxesTransformation, \
    TranslateSitesTransformation, CubicSupercellTransformation, \
    PrimitiveCellTransformation


ATOM_LIST = list(range(1,100))
print("Number of atoms:", len(ATOM_LIST))

class CrystalDataset(Dataset):
    def __init__(self, data_dir='data/BG_cifs', k=12, task='regression', fold = 0):
        self.k = k
        self.task = task
        self.data_dir = data_dir
        
        id_prop_file = os.path.join(self.data_dir, 'id_prop_test_{}.csv'.format(fold))
        assert os.path.exists(id_prop_file), 'id_prop_test_{}.csv does not exist!'.format(fold)
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        self.atom_featurizer = AtomCustomJSONInitializer(os.path.join(self.data_dir, 'atom_init.json'))
        self.feat_dim = self.atom_featurizer.get_length()

    def __getitem__(self, index):
        # get the cif id and path
        cif_id, self.labels = self.id_prop_data[index]
        cryst_path = os.path.join(self.data_dir, cif_id + '.cif')

        self.labels = np.array(self.labels)
        if self.task == 'regression':
            self.scaler = preprocessing.StandardScaler()
            self.scaler.fit(self.labels.reshape(-1,1))
            self.labels = self.scaler.transform(self.labels.reshape(-1,1))

        # read cif using pymatgen
        crys = Structure.from_file(cryst_path)
        pos = crys.frac_coords
        atom_indices = list(crys.atomic_numbers)
        cell = crys.lattice.get_cartesian_coords(1)
        feat = self.atom_featurizer.get_atom_features(atom_indices)
        N = len(pos)
        y = self.labels#[index]
        y = torch.tensor(y, dtype=torch.float).view(1,1)
        z = []
        for idx in atom_indices:
            z.append(ATOM_LIST.index(idx))
        z = torch.tensor(z, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        feat = torch.tensor(feat, dtype=torch.float)
        edge_index = knn_graph(pos, k=self.k, loop=False)
        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)

        data = Data(
            z=z, pos=pos, feat=feat, y=y, 
            edge_index=edge_index, edge_attr=edge_attr
        )
        # build the PyG graph 
        return data

    def __len__(self):
        return len(self.id_prop_data)

# class AugmentCrystalDataset(Dataset):
#     def __init__(self, data_dir='data/BG_cifs', k=12, task='regression', fold = 0):
#         self.k = k
#         self.task = task
#         self.data_dir = data_dir
        
#         id_prop_augment_file = os.path.join(self.data_dir, 'id_prop_augment_{}.csv'.format(fold))
#         assert os.path.exists(id_prop_augment_file), 'id_prop_augment_{}.csv does not exist!'.format(fold)
#         with open(id_prop_augment_file) as f:
#             reader = csv.reader(f)
#             self.id_prop_augment = [row for row in reader]    

#         self.atom_featurizer = AtomCustomJSONInitializer(os.path.join(self.data_dir, 'atom_init.json'))
#         self.feat_dim = self.atom_featurizer.get_length()

#     def __getitem__(self, index):
#         # get the cif id and path

#         augment_cif_id, self.aug_labels = self.id_prop_augment[index]
#         augment_cryst_path = os.path.join(self.data_dir, augment_cif_id + '.cif')

#         self.aug_labels = np.array(self.aug_labels)
#         if self.task == 'regression':
#             self.scaler = preprocessing.StandardScaler()
#             self.scaler.fit(self.aug_labels.reshape(-1,1))
#             self.aug_labels = self.scaler.transform(self.aug_labels.reshape(-1,1))

#         # read cif using pymatgen
#         aug_crys = Structure.from_file(augment_cryst_path)
#         pos = aug_crys.frac_coords
#         atom_indices = list(aug_crys.atomic_numbers)
#         cell = aug_crys.lattice.get_cartesian_coords(1)
#         feat = self.atom_featurizer.get_atom_features(atom_indices)
#         N = len(pos)
#         y = self.aug_labels#[index]
#         y = torch.tensor(y, dtype=torch.float).view(1,1)
#         atomics = []
#         for idx in atom_indices:
#             atomics.append(ATOM_LIST.index(idx))
#         atomics = torch.tensor(atomics, dtype=torch.long)
#         pos = torch.tensor(pos, dtype=torch.float)
#         feat = torch.tensor(feat, dtype=torch.float)
#         edge_index = knn_graph(pos, k=self.k, loop=False)
#         edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)

#         data = Data(
#             atomics=atomics, pos=pos, feat=feat, y=y, 
#             edge_index=edge_index, edge_attr=edge_attr
#         )

#         # build the PyG graph 
#         return data

#     def __len__(self):
#         return len(self.id_prop_augment)



class CrystalDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, data_dir='data', k=8, task='regression',fold = 0):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        #self.valid_size = valid_size
        #elf.test_size = test_size
        self.task = task
        self.fold = fold

    def get_data_loaders(self):
        test_dataset = CrystalDataset(self.data_dir,self.k, self.task,fold = self.fold)

        test_loader = self.get_test_data_loader(test_dataset)
        return test_loader#train_loader, valid_loader#, test_loader

    def get_test_data_loader(self, test_dataset):
        # obtain training indices that will be used for validation
        num_test = len(test_dataset)
        print(num_test)
        indices  = np.arange(num_test)

        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)
        test_idx = indices
        test_sampler = SubsetRandomSampler(test_idx)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                num_workers=self.num_workers, drop_last=False)

        return test_loader


if __name__ == "__main__":
    dataset = CrystalDataset()
    print(dataset)
    print(dataset.__getitem__(0))
    dataset = CrystalDatasetWrapper(batch_size=2, num_workers=0, valid_size=0.1, test_size=0.1, data_dir='data/BG_cifs')
    test_loader = dataset.get_data_loaders()
    for bn, data in enumerate(train_loader):
        print(data)
        print(data.atomics)
        print(data.pos)
        print(data.y)
        break