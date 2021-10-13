import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy
from sklearn import preprocessing
import warnings
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
#Structure()


ATOM_LIST = list(range(1,100))
print("Number of atoms:", len(ATOM_LIST))

class CrystalDataset(Dataset):
    def __init__(self, data_dir='data/BG_cifs', k=12, task='regression', fold = 0):
        self.k = k
        self.task = task
        self.data_dir = data_dir
        
        id_prop_file = os.path.join(self.data_dir, 'id_prop_train_{}.csv'.format(fold))
        assert os.path.exists(id_prop_file), 'id_prop_train_{}.csv does not exist!'.format(fold)
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

class AugmentCrystalDataset(Dataset):
    def __init__(self, data_dir='data/BG_cifs', k=12, task='regression', fold = 0):
        self.k = k
        self.task = task
        self.data_dir = data_dir
        
        id_prop_augment_file = os.path.join(self.data_dir, 'id_prop_augment_{}.csv'.format(fold))
        assert os.path.exists(id_prop_augment_file), 'id_prop_augment_{}.csv does not exist!'.format(fold)
        with open(id_prop_augment_file) as f:
            reader = csv.reader(f)
            self.id_prop_augment = [row for row in reader]    

        self.atom_featurizer = AtomCustomJSONInitializer(os.path.join(self.data_dir, 'atom_init.json'))
        self.feat_dim = self.atom_featurizer.get_length()

    def __getitem__(self, index):
        # get the cif id and path

        augment_cif_id, self.aug_labels = self.id_prop_augment[index]
        augment_cryst_path = os.path.join(self.data_dir, augment_cif_id + '.cif')

        self.aug_labels = np.array(self.aug_labels)
        if self.task == 'regression':
            self.scaler = preprocessing.StandardScaler()
            self.scaler.fit(self.aug_labels.reshape(-1,1))
            self.aug_labels = self.scaler.transform(self.aug_labels.reshape(-1,1))

        # read cif using pymatgen
        aug_crys = Structure.from_file(augment_cryst_path)
        pos = aug_crys.frac_coords
        atom_indices = list(aug_crys.atomic_numbers)
        cell = aug_crys.lattice.get_cartesian_coords(1)
        feat = self.atom_featurizer.get_atom_features(atom_indices)
        N = len(pos)
        y = self.aug_labels#[index]
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
        return len(self.id_prop_augment)



class CrystalDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, 
        data_dir='data', k=8, task='regression',fold = 0):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        #elf.test_size = test_size
        self.task = task
        self.fold = fold

    def get_data_loaders(self):
        train_dataset = AugmentCrystalDataset(self.data_dir, self.k, self.task, fold = self.fold)

        valid_dataset = CrystalDataset(self.data_dir,self.k, self.task,fold = self.fold)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset,valid_dataset)
        return train_loader, valid_loader#, test_loader

    def get_train_validation_data_loaders(self, train_dataset,valid_dataset):
        # obtain training indices that will be used for validation
        num_train = len(valid_dataset)
        indices  = np.arange(num_train)

        print((len((indices))))

        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)
        self.train_size = 1 - self.valid_size #- self.test_size
        train_size = int(self.train_size * num_train)
        valid_size = int(self.valid_size * num_train)
        # #test_size = int(self.test_size * num_train)
        # # print('Train size: {}, Validation size: {}, Test size: {}'.format(
        # #     train_size, valid_size, test_size
        # # ))

        print('Train size: {}, Validation size: {}'.format(train_size, valid_size))

        train_idx = indices[:train_size]

        num_aug = 4
        train_idx_augment = []
        for i in range (len(train_idx)):
            idx_correction = num_aug*train_idx[i]
            add_1 = idx_correction + 1
            add_2 = idx_correction + 2
            add_3 = idx_correction + 3
            add_  = idx_correction
            train_idx_augment.append(add_1)
            train_idx_augment.append(add_2)
            train_idx_augment.append(add_3)
            train_idx_augment.append(add_)
        valid_idx = indices[train_size:]

        train_sampler = SubsetRandomSampler(train_idx_augment)
        valid_sampler = SubsetRandomSampler(valid_idx)
        #test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False)
                                
        # test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_sampler,
        #                           num_workers=self.num_workers, drop_last=False)

        return train_loader, valid_loader#, test_loader


if __name__ == "__main__":
    dataset = CrystalDataset()
    print(dataset)
    print(dataset.__getitem__(0))
    dataset = CrystalDatasetWrapper(batch_size=2, num_workers=0, valid_size=0.1, test_size=0.1, data_dir='data/BG_cifs')
    train_loader, valid_loader = dataset.get_data_loaders()
    for bn, data in enumerate(train_loader):
        print(data)
        print(data.z)
        print(data.pos)
        print(data.y)
        break