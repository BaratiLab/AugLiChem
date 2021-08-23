import sys
sys.path.append(sys.path[0][:-4])

import shutil
import numpy as np

import torch
from torch_geometric.data import Data as PyG_Data

from auglichem.molecule import RandomAtomMask, RandomBondDelete
from auglichem.molecule.data import MoleculeData
from auglichem.molecule.data._molecule_dataset import MolData

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

from rdkit import Chem


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

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


def test_atom_mask():

    # Always mask at least one atom
    data = smiles2graph("C")
    atom_masked = RandomAtomMask(p=0)(data, 0)
    assert atom_masked.x[:,0].numpy() == [118]

    # Mask every atom
    data = smiles2graph("CCCCCC")
    atom_masked = RandomAtomMask(p=1)(data, 0)
    assert all(atom_masked.x[:,0].numpy() == [118]*len(data.x))


def test_atom_mask_mol():

    # Dummy data set
    data = MolData("", smiles_data=["C"], labels={'target': [1]}, task='classification')
    data.target = 'target'

    # Mask every atom
    atom_masked = RandomAtomMask(p=1)(data.__getitem__(0), 0)
    assert all(atom_masked.x[:,0].numpy() == [118]*5)

    # Do the same with methane
    #mol = Chem.AddHs(mol)
    #x = data._get_data_x(mol)
    #x_mask = data.mask_nodes(x, len(x), 1)
    #assert all(x_mask[:,0].numpy() == [99]*5) # Mask every atom


def test_bond_delete():

    # No bonds in molecule (single atom)
    data = smiles2graph("C")
    bond_masked = RandomAtomMask(p=0)(data, 0)
    assert bond_masked.edge_index.numel() == 0
    assert bond_masked.edge_attr.numel() == 0

    # No bonds deleted
    data = smiles2graph("CC")
    bond_masked = RandomBondDelete(p=0)(data, 0)
    assert torch.all(torch.eq(bond_masked.edge_index, torch.Tensor([[0,1],[1,0]])))
    assert torch.all(torch.eq(bond_masked.edge_attr, torch.Tensor([[0,0],[0,0]]).long()))

    # Delete every bond
    data = smiles2graph("CCCCCC")
    bond_masked = RandomBondDelete(p=1)(data, 0)
    assert bond_masked.edge_index.numel() == 0
    assert bond_masked.edge_attr.numel() == 0


def test_bond_delete_mol():

    # Dummy data set
    data = MolData("", smiles_data=["C"], labels={'target': [1]}, task='classification')
    data.target = 'target'

    # Mask every atom
    atom_masked = RandomBondDelete(p=1)(data.__getitem__(0), 0)
    assert atom_masked.edge_index.numel() == 0
    assert atom_masked.edge_attr.numel() == 0


def test_smiles2graph():
    pass


def test_molecule_data():
    data = MoleculeData("BACE")
    train, valid, test = data.get_data_loaders()
    shutil.rmtree("./data_download")

    #for b, t in enumerate(train):
    #    print(b)


#if __name__ == '__main__':
    #test_atom_mask()
    #test_bond_delete()
    #test_atom_mask_mol()
    #test_bond_delete_mol()
    #test_molecule_data()
