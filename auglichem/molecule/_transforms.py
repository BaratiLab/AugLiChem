import os
import math
import numpy as np
import random
import networkx as nx
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import BreakBRICSBonds, FindBRICSBonds

import torch
import torch_geometric
from torch_geometric.data import Data as PyG_Data

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST


def _mol2graph(mol):
    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        try:
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())
        except:
            continue

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
    def __init__(self):
        """"""
        super().__init__(p)

    def __call__(self, mol: rdkit.Chem.rdchem.Mol) -> PyG_Data:
        """
        @param mol: rdkit.Chem.rdchem.Mol
        @returns: PyG Data
        """
        return self.apply_transform(mol)

    def apply_transform(self, mol: rdkit.Chem.rdchem.Mol) -> PyG_Data:
        """
        This function is to be implemented in the child classes.
        From this function, call the function with the
        parameters specified
        """
        return _mol2graph(mol)


class BaseAugmentTransform(BaseTransform):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        if(isinstance(p, list)):
            assert 0 <= p[0] <= 1.0
            assert 0 <= p[1] <= 1.0
            assert p[0] < p[1]
        else:
            assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"
        self.p = p

    def __call__(self, mol: rdkit.Chem.rdchem.Mol, seed=None) -> PyG_Data:
        #TODO Fix this to use Optional[None]?
        """
        @param mol: rdkit.Chem.rdchem.Mol to be augmented
        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned
        @returns: Augmented PyG Data
        """
        if(isinstance(self.p, list)):
            self.p = random.uniform(self.p[0], self.p[1])
        else:
            self.p = self.p
        assert isinstance(self.p, (float, int))
        assert isinstance(mol, rdkit.Chem.rdchem.Mol), "mol_graph passed in must be a PyG Data"
        return self.apply_transform(mol_graph, seed)


class RandomAtomMask(BaseAugmentTransform):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)

    def apply_transform(self, mol: rdkit.Chem.rdchem.Mol, seed: Optional[None]) -> PyG_Data:
        """
        Transform that randomly mask atoms given a certain ratio
        @param mol: rdkit.Chem.rdchem.Mol to be augmented
        @param seed: 
        @returns: Augmented PyG Data
        """
        if(seed):
            random.seed(seed)
        
        mol_graph = _mol2graph(mol)

        N = mol_graph.x.size(0)
        num_mask_nodes = max([1, math.floor(self.p*N)])
        mask_nodes = random.sample(list(range(N)), num_mask_nodes)

        aug_mol_graph = deepcopy(mol_graph)
        for atom_idx in mask_nodes:
            aug_mol_graph.x[atom_idx,:] = torch.tensor([len(ATOM_LIST)-1, 0])
        
        return aug_mol_graph


class RandomBondDelete(BaseAugmentTransform):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)

    def apply_transform(self, mol: rdkit.Chem.rdchem.Mol, seed: Optional[None]) -> PyG_Data:
        """
        Transform that randomly delete chemical bonds given a certain ratio
        @param mol: rdkit.Chem.rdchem.Mol to be augmented
        @returns: Augmented PyG Data
        """
        if(seed):
            random.seed(seed)

        mol_graph = _mol2graph(mol)

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


class RandomSubgraphRemoval(BaseAugmentTransform):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
    
    def _remove_subgraph(self, nx_graph, center):
        assert self.p <= 1
        aug_nx_graph = nx_graph.copy()
        num = int(np.floor(len(aug_nx_graph.nodes) * self.p))
        removed = []
        temp = [center]
        
        while len(removed) < num:
            neighbors = []
            for n in temp:
                neighbors.extend([i for i in aug_nx_graph.neighbors(n) if i not in temp])      
            for n in temp:
                if len(removed) < num:
                    aug_nx_graph.remove_node(n)
                    removed.append(n)
                else:
                    break
            temp = list(set(neighbors))
        return aug_nx_graph, removed

    def apply_transform(self, mol: rdkit.Chem.rdchem.Mol, seed: Optional[None]) -> PyG_Data:
        """
        Transform that randomly remove an induced subgraph given a certain ratio
        @param mol: rdkit.Chem.rdchem.Mol to be augmented
        @returns: Augmented PyG Data
        """
        if(seed):
            random.seed(seed)

        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        nx_graph = nx.Graph(edges)

        # Get the graph after removing subgraphs
        start_idx = random.sample(list(range(len(atoms))), 1)[0]
        aug_nx_graph, removed_aug = self._remove_subgraph(nx_graph, start_idx)

        mol_graph = _mol2graph(mol)

        # Mask the atoms in the removed list
        x_aug = deepcopy(mol_graph.x)
        for atom_idx in removed_aug:
            # Change atom type to 118, and chirality to 0
            x_aug[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])

        # Only consider bond still exist after removing subgraph
        row_aug, col_aug, = [], []
        edge_feat_aug = []
        aug_nx_graph_edges = list(aug_nx_graph.edges)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            if (start, end) in aug_nx_graph_edges:
                row_aug += [start, end]
                col_aug += [end, start]
                edge_feat_aug.append(feature)
                edge_feat_aug.append(feature)

        edge_index_aug = torch.tensor([row_aug, col_aug], dtype=torch.long)
        edge_attr_aug = torch.tensor(np.array(edge_feat_aug), dtype=torch.long)
        
        aug_mol_graph = PyG_Data(x=x_aug, edge_index=edge_index_aug, edge_attr=edge_attr_aug)
        
        return aug_mol_graph


class RandomMotifRemoval(BaseAugmentTransform):
    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)
    
    def _remove_frag(self, mol):
        break_bonds = list(FindBRICSBonds(mol))
        num_breaks = len(break_bonds)
        num_frags = max([1, num_breaks])

        mol2 = BreakBRICSBonds(mol, break_bonds)
        res = Chem.MolToSmiles(mol2, True)
        frag_list = res.split('.')
        frag_smiles = '.'.join(random.sample(frag_list, num_frags))

        frag_mol = Chem.MolFromSmiles(frag_smiles)
        frag_mol = Chem.AddHs(frag_mol)
        
        return frag_mol

    def apply_transform(self, mol: rdkit.Chem.rdchem.Mol, seed: Optional[None]) -> PyG_Data:
        """
        Transform that randomly remove a motif decomposed via BRICS
        @param mol: rdkit.Chem.rdchem.Mol to be augmented
        @returns: Augmented PyG Data
        """
        if(seed):
            random.seed(seed)

        aug_mol = self._remove_frag(mol)
        aug_mol_graph = _mol2graph(aug_mol)
        
        return aug_mol_graph
