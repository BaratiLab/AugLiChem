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
    def __init__(self, data_dir='data/BG_cifs', k=5, task='regression', aug_num=10,
        atom_mask=True, edge_delete=True, rotate=True, perturb=True, swap_axes=False, 
        site_transform=False, super_cell_tranform=False, primitive_cell_transform=False
    ):
        super(Dataset, self).__init__()
        self.k = k
        self.task = task
        self.data_dir = data_dir
        self.cif_ids, self.labels = read_csv(data_dir, task)
        self.atom_featurizer = AtomCustomJSONInitializer(os.path.join('dataset', 'atom_init.json'))
        self.feat_dim = self.atom_featurizer.get_length()
        
        self.atom_mask = atom_mask
        self.edge_delete = edge_delete
        self.rotate = rotate
        self.perturb = perturb
        self.swap_axes = swap_axes
        self.site_transform = site_transform
        self.super_cell_tranform = super_cell_tranform
        self.primitive_cell_transform = primitive_cell_transform

        if self.rotate:
            self.rotater = RotationTransformation()
        if self.perturb:
            self.perturber = PerturbStructureTransformation(distance=0.05, min_distance=0.0)
        if self.swap_axes:
            self.swapaxiser = SwapAxesTransformation(p=0.5)
        if self.site_transform:
            self.site_transformer = TranslateSitesTransformation()
        if self.primitive_cell_transform:
            self.primitive_cell_transformer = PrimitiveCellTransformation(tolerance=0.5)
        if self.super_cell_tranform:
            self.super_cell_transformer = CubicSupercellTransformation()
        
        self.aug_num = aug_num

        self.labels = np.array(self.labels)
        if self.task == 'regression':
            self.scaler = preprocessing.StandardScaler()
            self.scaler.fit(self.labels.reshape(-1,1))
            self.labels = self.scaler.transform(self.labels.reshape(-1,1))
            # self.label_mean = scaler.mean_
            # self.label_std = scaler.var_

    def __getitem__(self, total_index):
        index = total_index // self.aug_num
        # get the cif id and path
        cif_id = self.cif_ids[index]
        cryst_path = os.path.join(self.data_dir, cif_id)

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

        y = self.labels[index]
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
        edge_index = knn_graph(pos, k=self.k, loop=False)
        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)

        data = Data(
            atomics=atomics, pos=pos, feat=feat, y=y, 
            edge_index=edge_index, edge_attr=edge_attr
        )

        # # random swap 2 axes
        # if self.swap_axes:
        #     crys = self.swapaxiser.apply_transformation(crys)
        
        # random rotate
        if self.rotate:
            for i in range(3):
                axis = np.zeros(3)
                axis[i] = 1
                # rot_ang = np.random.uniform(-90.0, 90.0)
                rot_ang = np.random.uniform(-np.pi/2, np.pi/2)
                crys = self.rotater.apply_transformation(crys, axis, rot_ang, angle_in_radians=True)
        
        # random perturb
        if self.perturb:
            crys = self.perturber.apply_transformation(crys)

        # build the PyG graph 
        atomics = []
        for idx in atom_indices:
            atomics.append(ATOM_LIST.index(idx))
        atomics = torch.tensor(atomics, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        feat = torch.tensor(feat, dtype=torch.float)
        edge_index = knn_graph(pos, k=self.k, loop=False)
        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)

        # randomly mask some atoms
        if self.atom_mask:
            num_mask_nodes = max([1, math.floor(0.25*N)])
            mask_indices = random.sample(list(range(N)), num_mask_nodes)

            for idx in mask_indices:
                atomics[idx] = torch.tensor(len(ATOM_LIST), dtype=torch.long)
                pos[idx,:] =  torch.zeros(3, dtype=torch.float)
                # feat[idx,:] = torch.zeros(self.feat_dim, dtype=torch.float)

        # randomly delete some edges
        if self.edge_delete:
            M = edge_index.size(1)
            num_mask_edges = math.floor(0.25*M)
            mask_edges = random.sample(list(range(M)), num_mask_edges)

            masked_edge_index = torch.zeros((2, M-num_mask_edges), dtype=torch.long)
            masked_edge_attr = torch.zeros(M-num_mask_edges, dtype=torch.long)
            count = 0
            for bond_idx in range(M):
                if bond_idx not in mask_edges:
                    masked_edge_index[:,count] = edge_index[:,bond_idx]
                    masked_edge_attr[count] = edge_attr[bond_idx]
                    count += 1
            edge_index = masked_edge_index
            edge_attr =- masked_edge_attr
        
        # return PyG Data object
        aug_data = Data(
            atomics=atomics, pos=pos, feat=feat, y=y, 
            edge_index=edge_index, edge_attr=edge_attr
        )
        return data, aug_data

    def __len__(self):
        return len(self.cif_ids) * self.aug_num


class CrystalDatasetWrapper(object):
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_dir='data', k=8, task='regression', aug_num=10,
        atom_mask=True, edge_delete=True, rotate=True, perturb=True, swap_axes=False, 
        site_transform=False, super_cell_tranform=False, primitive_cell_transform=False
    ):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.task = task

        self.atom_mask = atom_mask
        self.edge_delete = edge_delete
        self.rotate = rotate
        self.perturb = perturb
        self.swap_axes = swap_axes
        self.site_transform = site_transform
        self.super_cell_tranform = super_cell_tranform
        self.primitive_cell_transform = primitive_cell_transform

        self.aug_num = aug_num

    def get_data_loaders(self):
        train_dataset = CrystalDataset(
            self.data_dir, self.k, self.task, self.aug_num, 
            self.atom_mask, self.edge_delete, self.rotate, self.perturb, self.swap_axes, 
            self.site_transform, self.super_cell_tranform, self.primitive_cell_transform
        )
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset) // train_dataset.aug_num
        indices = list(range(num_train))

        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)

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

        train_idx = []
        for i in range(train_size):
            for j in range(train_dataset.aug_num):
                idx = indices[i]*train_dataset.aug_num + j
                train_idx.append(idx)
        np.random.shuffle(train_idx)

        valid_idx = []
        for i in range(train_size, train_size+valid_size):
            # for j in range(dataset.aug_num):
            idx = indices[i]*train_dataset.aug_num
            valid_idx.append(idx)
        np.random.shuffle(valid_idx)
        
        test_idx = []
        for i in range(train_size+valid_size, train_size+valid_size+test_size):
            # for j in range(dataset.aug_num):
            idx = indices[i]*train_dataset.aug_num
            test_idx.append(idx)
        np.random.shuffle(test_idx)

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