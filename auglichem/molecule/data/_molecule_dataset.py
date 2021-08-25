import os
import csv
import math
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data as PyG_Data


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

from auglichem.utils import (
        ATOM_LIST,
        CHIRALITY_LIST,
        BOND_LIST,
        BONDDIR_LIST,
        random_split,
        scaffold_split
)
from auglichem.molecule import RandomAtomMask, RandomBondDelete
from ._load_sets import read_smiles


#TODO docstrings for MoleculeData

class MoleculeDataset(Dataset):
    def __init__(self, dataset, data_path=None, transform=None, smiles_data=None, labels=None,
                 task=None, test_mode=False, aug_time=1, atom_mask_ratio=[0, 0.25],
                 bond_delete_ratio=[0, 0.25], target=None, class_labels=None, **kwargs):
        '''
            Initialize Molecular Data set object. This object tracks data, labels,
            task, test, and augmentation

            Input:
            -----------------------------------
            dataset (str): Name of data set
            data_path (str, optional default=None): path to store or load data
            smiles_data (np.ndarray of str): Data set in smiles format
            labels (np.ndarray of float): Data labels, maybe optional if unsupervised?
            task (str): 'regression' or 'classification' indicating the learning task
            test_mode (boolean, default=False): Does no augmentations if true
            aug_time (int, optional, default=1): Controls augmentations (not quite sure how yet)
            atom_mask_ratio (list, float): If list, sample mask ratio uniformly over [a, b]
                                           where a < b, if float, set ratio to input.
            bond_delete_ratio (list, float): If list, sample mask ratio uniformly over [a, b]
                                           where a < b, if float, set ratio to input.


            Output:
            -----------------------------------
            None
        '''
        super(Dataset, self).__init__()

        # Store class attributes
        self.dataset = dataset
        self.data_path = data_path
        self.transform = transform

        if(smiles_data is None):
            self.smiles_data, self.labels, self.task = read_smiles(dataset, data_path)
        else:
            self.smiles_data = smiles_data
            self.labels = labels
            self.task = task

        # No augmentation if no transform is specified
        if(self.transform is None):
            self.test_mode = True
        else:
            self.test_mode = test_mode
            self.aug_time = aug_time

        if self.test_mode:
            self.aug_time = 1

        assert type(aug_time) == int
        assert aug_time >= 1

        # For reproducibility
        self.reproduce_seeds = list(range(self.__len__()))
        np.random.shuffle(self.reproduce_seeds)

        # Store mask ratios
        self.atom_mask_ratio = atom_mask_ratio
        self.bond_delete_ratio = bond_delete_ratio

        if(target is not None):
            self.target = target
        else:
            self.target = list(self.labels.keys())[0]

        if(class_labels is not None):
            self.class_labels = class_labels
        else:
            self.class_labels = self.labels[self.target]


    def _get_data_x(self, mol):
        '''
            Get the transformed data features.

            Inputs:
            -----------------------------------
            mol ( object): Current molecule

            Outputs:
            -----------------------------------
            x (torch.Tensor of longs):
        '''

        # Set up data arrays
        type_idx, chirality_idx, atomic_number = [], [], []

        # Gather atom data
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        # Concatenate atom type with chirality index
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        return x


    def _get_data_y(self, index):
        '''
            Get the transformed data label.

            Inputs:
            -----------------------------------
            index (int): Index for current molecule

            Outputs:
            -----------------------------------
            y (torch.Tensor, long if classification, float if regression): Data label
        '''

        if self.task == 'classification':
            y = torch.tensor(self.class_labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.class_labels[index], dtype=torch.float).view(1,-1)

        return y


    def _get_edge_index_and_attr(self, mol):
        '''
            Create the edge index and attributes

            Inputs:
            -----------------------------------
            mol ():

            Outputs:
            -----------------------------------
            edge_index ():
            edge_attr ():
        '''

        # Set up data collection lists
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():

            # Get the beginning and end atom indices
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # Store bond atoms
            row += [start, end]
            col += [end, start]

            # Store edge featuers
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        # Create edge index and attributes
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        return edge_index, edge_attr


    def __getitem__(self, index):
        '''
            Selects an element of self.smiles_data according to the index.
            Edge and node masking are done here for each individual molecule

            Input:
            -----------------------------------
            index (int): Index of molecule we would like to augment

            Output:
            -----------------------------------
            masked_data (Data object): data that has been augmented with node and edge masking

        '''
        # If augmentation is done, actual dataset is smaller than given indices
        if self.test_mode:
            true_index = index
        else:
            true_index = index // self.aug_time

        # Create initial data set
        mol = Chem.MolFromSmiles(self.smiles_data[true_index])
        mol = Chem.AddHs(mol)

        # Get data x and y
        x = self._get_data_x(mol)
        y = self._get_data_y(true_index)

        # Get edge index and attributes
        edge_index, edge_attr = self._get_edge_index_and_attr(mol)

        # Store number of atoms and bonds
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        # Mask according to initialized rules

        # Set up PyG data object
        molecule = PyG_Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        if(not self.test_mode):
            aug_molecule = self.transform(molecule, seed=self.reproduce_seeds[index])
            return aug_molecule
        else:
            return molecule

        # Set up atom masking object
        #if(isinstance(self.atom_mask_ratio, list)):
        #    amr = random.uniform(self.atom_mask_ratio[0], self.atom_mask_ratio[1])
        #else:
        #    amr = self.atom_mask_ratio

        #atom_mask = RandomAtomMask(amr)

        ## Set up bond deletion object
        #if(isinstance(self.bond_delete_ratio, list)):
        #    bdr = random.uniform(self.bond_delete_ratio[0], self.bond_delete_ratio[1])
        #else:
        #    bdr = self.bond_delete_ratio

        #edge_mask = RandomBondDelete(bdr)

        ## Do masking
        #molecule = atom_mask(molecule, self.reproduce_seeds[index])
        #molecule = edge_mask(molecule, self.reproduce_seeds[index])

        #return molecule


    def __len__(self):
        return len(self.smiles_data) * self.aug_time


class MoleculeDatasetWrapper(MoleculeDataset):
    #TODO: Take in transformation/composition object as argument to match crystal?
    def __init__(self, dataset, transform=None, split="scaffold", batch_size=64, num_workers=0,
                 valid_size=0.1, test_size=0.1, aug_time=1, data_path=None, target=None):
        '''
            Input:
            ---
            dataset (str): One of the datasets available from MoleculeNet
                           (http://moleculenet.ai/datasets-1)
            transform (Compose, OneOf, RandomAtomMask, RandomBondDelete object): transormations
                           to apply to the data at call time.
            split (str, optional default=scaffold): random or scaffold. The splitting strategy
                                                    used for train/test/validation set creation.
            batch_size (int, optional default=64): Batch size used in training
            num_workers (int, optional default=0): Number of workers used in loading data
            valid_size (float in [0,1], optional default=0.1): 
            test_size (float in [0,1],  optional default=0.1): 
            aug_time (int, optional default=1):
            data_path (str, optional default=None): specify path to save/lookup data. Default
                        creates `data_download` directory and stores data there


            Output:
            ---
            None
        '''
        print("TRANSFORM IN MOLECULE DATASET: {}".format(transform))
        super().__init__(dataset, data_path, transform)
        self.split = split
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.aug_time = aug_time
        self.smiles_data = np.asarray(self.smiles_data)


    def get_data_loaders(self, target=None):
        '''

        '''
        if(not target):
            self.target = list(self.labels.keys())[0]

        # Get indices of data splits
        #TODO: Include different splits
        if(self.split == 'scaffold'):
            train_idx, valid_idx, test_idx = scaffold_split(self.smiles_data, self.valid_size,
                                                            self.test_size)
        elif(self.split == 'random'):
            raise NotImplementedError("Random splitting not supported yet")
        else:
            raise ValueError("Please select scaffold or random split")

        # Split
        #TODO: Fix this redundency of passing in dataset name?
        train_set = MoleculeDataset(self.dataset, transform=self.transform, smiles_data=self.smiles_data[train_idx],
                            class_labels=self.labels[self.target][train_idx], test_mode=False,
                            aug_time=self.aug_time, task=self.task, target=self.target)
        valid_set = MoleculeDataset(self.dataset, transform=self.transform, smiles_data=self.smiles_data[valid_idx],
                            class_labels=self.labels[self.target][valid_idx],
                            test_mode=True, task=self.task, target=self.target)
        test_set = MoleculeDataset(self.dataset, transform=self.transform, smiles_data=self.smiles_data[test_idx],
                           class_labels=self.labels[self.target][test_idx],
                           test_mode=True, task=self.task, target=self.target)

        train_loader = DataLoader(
            train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=len(valid_set),
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            test_set, batch_size=len(test_set),
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader

