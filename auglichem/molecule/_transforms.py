import os
import math
import numpy as np
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import BRICSDecompose

import torch
import torch_geometric
from torch_geometric.data import Data as PyG_Data

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

#TODO: Updated documentation to match other files


class BaseTransform(object):
    def __init__(self, prob: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0. If
                  a list is passed in, the value will be randomly and sampled between the two
                  end points.
        """
        if(isinstance(prob, list)):
            assert 0 <= prob[0] <= 1.0
            assert 0 <= prob[1] <= 1.0
            assert prob[0] < prob[1]
        else:
            assert 0 <= prob <= 1.0, "p must be a value in the range [0, 1]"
        self.prob = prob

    def __call__(self, mol_graph: PyG_Data, seed=None) -> PyG_Data:
        """
        @param mol_graph: PyG Data to be augmented
        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned
        @returns: Augmented PyG Data
        """
        if(isinstance(self.prob, list)):
            self.p = random.uniform(self.prob[0], self.prob[1])
        else:
            self.p = self.prob
        assert isinstance(self.p, (float, int))
        assert isinstance(mol_graph, PyG_Data), "mol_graph passed in must be a PyG Data"
        return self.apply_transform(mol_graph, seed)

    def apply_transform(self, mol_graph: PyG_Data) -> PyG_Data:
        """
        This function is to be implemented in the child classes.
        From this function, call the augmentation function with the
        parameters specified
        """
        raise NotImplementedError()


class RandomAtomMask(BaseTransform):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)

    def apply_transform(self, mol_graph: PyG_Data, seed: Optional[None]) -> PyG_Data:
        """
        Transform that randomly mask atoms given a certain ratio
        @param mol_graph: PyG Data to be augmented
        @param seed: 
        @returns: Augmented PyG Data
        """
        if(seed is not None):
            random.seed(seed)
        N = mol_graph.x.size(0)
        num_mask_nodes = max([1, math.floor(self.p*N)])
        mask_nodes = random.sample(list(range(N)), num_mask_nodes)

        aug_mol_graph = deepcopy(mol_graph)
        for atom_idx in mask_nodes:
            aug_mol_graph.x[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        
        return aug_mol_graph


class RandomBondDelete(BaseTransform):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)

    def apply_transform(self, mol_graph: PyG_Data, seed: Optional[None]) -> PyG_Data:
        """
        Transform that randomly delete chemical bonds given a certain ratio
        @param mol_graph: PyG Data to be augmented
        @returns: Augmented PyG Data
        """
        if(seed is not None):
            random.seed(seed)
        M = mol_graph.edge_index.size(1) // 2
        num_mask_edges = max([0, math.floor(self.p*M)])
        mask_edges_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges = [2*i for i in mask_edges_single] + [2*i+1 for i in mask_edges_single]

        aug_mol_graph = deepcopy(mol_graph)
        aug_mol_graph.edge_index = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        aug_mol_graph.edge_attr = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges:
                aug_mol_graph.edge_index[:,count] = mol_graph.edge_index[:,bond_idx]
                aug_mol_graph.edge_attr[count,:] = mol_graph.edge_attr[bond_idx,:]
                count += 1
        
        return aug_mol_graph


class MotifRemoval(object):
    def __init__(self, similarity_threshold=0.6):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.allowable_features = {
            'possible_atomic_num_list' : list(range(1, 119)),
            'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            'possible_chirality_list' : [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ],
            'possible_hybridization_list' : [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
            ],
            'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
            'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'possible_bonds' : [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ],
            'possible_bond_dirs' : [ # only for double bond stereo information
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]
        }
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


    #def apply_transform(self, mol: rdkit.Chem.rdchem.Mol) -> List[rdkit.Chem.rdchem.Mol]:
    def apply_transform(self, data: PyG_Data, seed=None) -> PyG_Data:
        """
        Transform that randomly remove a motif decomposed via BRICS
        @param mol: rdkit.Chem.rdchem.Mol to be augmented
        @returns: list of augmented rdkit.Chem.rdchem.Mol
        """

        mol = Chem.MolFromSmiles(data.smiles)
        aug_mols = []
        aug_mols.append(mol)
        fp = Chem.RDKFingerprint(mol)
        res = list(BRICSDecompose(mol, returnMols=False, singlePass=True))
        for r in res:
            mol_aug = Chem.MolFromSmiles(r)
            fp_aug = Chem.RDKFingerprint(mol_aug)
            #print(self.similarity_threshold)
            #print(DataStructs.FingerprintSimilarity(fp, fp_aug) > self.similarity_threshold)
            if DataStructs.FingerprintSimilarity(fp, fp_aug) > self.similarity_threshold:
                aug_mols.append(mol_aug)

        # Get data x and y
        x = self._get_data_x(mol)

        # Get edge index and attributes
        edge_index, edge_attr = self._get_edge_index_and_attr(mol)
        return PyG_Data(x=x, y=data.y, edge_index=edge_index, edge_attr=edge_attr,
                        smiles=data.smiles)


    def __call__(self, data, seed=None):
        return self.apply_transform(data, seed)

