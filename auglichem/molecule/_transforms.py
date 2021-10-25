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

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST


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
