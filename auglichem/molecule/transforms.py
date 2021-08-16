import os
import math
import numpy as np
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem

import torch
import torch_geometric
from torch_geometric.data import Data as PyG_Data


ATOM_LIST = list(range(1,119))

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]

BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC,
    BT.UNSPECIFIED,
]

BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE
]


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    data = PyG_Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class BaseTransform(object):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(self, mol_graph: PyG_Data) -> PyG_Data:
        """
        @param mol_graph: PyG Data to be augmented
        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned
        @returns: Augmented PyG Data
        """
        assert isinstance(mol_graph, PyG_Data), "mol_graph passed in must be a PyG Data"
        return self.apply_transform(mol_graph)

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

    def apply_transform(self, mol_graph: PyG_Data) -> PyG_Data:
        """
        Transform that randomly mask atoms given a certain ratio
        @param mol_graph: PyG Data to be augmented
        @returns: Augmented PyG Data
        """
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

    def apply_transform(self, mol_graph: PyG_Data) -> PyG_Data:
        """
        Transform that randomly delete chemical bonds given a certain ratio
        @param mol_graph: PyG Data to be augmented
        @returns: Augmented PyG Data
        """
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
