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


def read_csv(csv_dir, task):
    csv_path = os.path.join(csv_dir, 'id_prop.csv')
    cif_ids, labels = [], []
    MP_energy = False
    if 'MP-formation-energy' in csv_dir:
        MP_energy = True
    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                cif_id = row['CIF_ID']
                label = row['val']
                if MP_energy:
                    cif_ids.append(cif_id+'.cif')
                else:
                    cif_ids.append(cif_id)

                if task == 'classification':
                    label = int(label == 'True')
                    labels.append(label)
                    # labels.append(int(label))
                elif task == 'regression':
                    labels.append(float(label))
                else:
                    ValueError('task must be either regression or classification')
    return cif_ids, labels


class CrystalDataset(Dataset):
    def __init__(self, data_dir='data/BG_cifs', k=5, task='regression'):
        super(Dataset, self).__init__()
        self.k = k
        self.task = task
        self.data_dir = data_dir
        #self.cif_ids, self.labels = read_csv(data_dir, task)
        self.atom_featurizer = AtomCustomJSONInitializer(os.path.join('dataset', 'atom_init.json'))
        self.feat_dim = self.atom_featurizer.get_length()
        

            # self.label_mean = scaler.mean_
            # self.label_std = scaler.var_
        
        id_prop_file = os.path.join(self.data_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]


        print(len((self.id_prop_data)))
    
    def __getitem__(self, index):
        # get the cif id and path
        cif_id, self.labels = self.id_prop_data[index]
        cryst_path = os.path.join(self.data_dir, cif_id + '.cif')

        self.labels = np.array(self.labels)
        if self.task == 'regression':
            self.scaler = preprocessing.StandardScaler()
            self.scaler.fit(self.labels.reshape(-1,1))
            self.labels = self.scaler.transform(self.labels.reshape(-1,1))
        # if self.swap_axes:
        #     # read cif using ASE
        #     crys = ase_read(cryst_path)
        #     crys = self.swapaxiser.apply_transformation(crys)
        #     # atom_indices = crys.numbers
        #     # pos = crys.positions
        #     # feat = self.atom_featurizer.get_atom_features(atom_indices)
        #     # N = len(pos)
        # else:
        #     # read cif using pymatgen
        #     crys = Structure.from_file(cryst_path)

        # read cif using pymatgen
        crys = Structure.from_file(cryst_path)
        pos = crys.frac_coords
        atom_indices = list(crys.atomic_numbers)
        cell = crys.lattice.get_cartesian_coords(1)
        feat = self.atom_featurizer.get_atom_features(atom_indices)
        N = len(pos)

        y = self.labels##[index]
        y = torch.tensor(y, dtype=torch.float).view(1,1)
        # if self.task == 'regression':
        #     y = torch.tensor(y, dtype=torch.float).view(1,1)
        # elif self.task == 'classification':
        #     # y = torch.tensor(y, dtype=torch.long).view(1,1)
        #     y = torch.tensor(y, dtype=torch.float).view(1,1)

        # build the PyG graph 
        atomics = []
        for idx in atom_indices:
            atomics.append(ATOM_LIST.index(idx))
        atomics = torch.tensor(atomics, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        feat = torch.tensor(feat, dtype=torch.float)
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
        #return data       # N = len(pos)
  
        #return data#, aug_data

    def __len__(self):
        return len(self.id_prop_data)### * self.aug_num


class CrystalDatasetWrapper(object):
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_dir='data', k=8, task='regression',fold = 0
    ):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.task = task


        self.fold = fold

    def get_data_loaders(self):
        train_dataset = CrystalDataset(
            self.data_dir, self.k, self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset) 
        indices = list(range(num_train))

        random_state = np.random.RandomState(seed=self.fold)
        random_state.shuffle(indices)
        #np.random.shuffle(indices)

        # split = int(np.floor(self.valid_size * num_train))
        # split2 = int(np.floor(self.test_size * num_train))
        # valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        # print('Data split:', len(train_idx), len(valid_idx), len(test_idx))

        self.train_size = 1 - self.valid_size - self.test_size
        train_size = int(self.train_size * num_train)
        valid_size = int(self.valid_size * num_train)
        test_size = int(self.test_size * num_train)
        print('Train size: {}, Validation size: {}, Test size: {}'.format(
            train_size, valid_size, test_size
        ))

        train_idx = indices[:train_size]

        valid_idx = indices[-(valid_size+test_size):-test_size]
        
        test_idx = indices[-test_size:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False)
                                
        test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                  num_workers=self.num_workers, drop_last=False)

        return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    dataset = CrystalDataset()
    print(dataset)
    print(dataset.__getitem__(0))
    dataset = CrystalDatasetWrapper(batch_size=2, num_workers=0, valid_size=0.1, test_size=0.1, data_dir='data/BG_cifs')
    train_loader, valid_loader, test_loader = dataset.get_data_loaders()
    for bn, data in enumerate(train_loader):
        print(data)
        print(data.atomics)
        print(data.pos)
        print(data.y)
        break