import sys
sys.path.append(sys.path[0][:-14])

import shutil
import numpy as np
import warnings
from tqdm import tqdm

import torch
from torch_geometric.data import Data as PyG_Data

from auglichem.molecule import RandomAtomMask, RandomBondDelete, Compose, OneOf, MotifRemoval
from auglichem.molecule.data import MoleculeDataset, MoleculeDatasetWrapper

from auglichem.utils import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


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
    assert atom_masked.x[:,0].numpy() == [119]

    # Mask every atom
    data = smiles2graph("CCCCCC")
    atom_masked = RandomAtomMask(p=1)(data, 0)
    assert all(atom_masked.x[:,0].numpy() == [119]*len(data.x))

    # Check seed setting (same)
    data = smiles2graph("C"*50)
    atom_mask = RandomAtomMask(p=0.5)
    atom_masked1 = atom_mask(data, 42)
    atom_masked2 = atom_mask(data, 42)
    atom_masked3 = atom_mask(data, 4)
    assert all(atom_masked1.x[:,0].numpy() == atom_masked2.x[:,0].numpy())
    assert any(atom_masked1.x[:,0].numpy() != atom_masked3.x[:,0].numpy())


def test_atom_mask_mol():

    # Dummy data set
    data = MoleculeDataset("", smiles_data=["C"], labels={'target': [1]}, task='classification')
    data.target = 'target'

    # Mask every atom
    atom_masked = RandomAtomMask(p=1)(data.__getitem__(0), seed=0)
    assert all(atom_masked.x[:,0].numpy() == [119]*5)


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

    # Check seed setting
    data = smiles2graph("C"*50)
    bond_masked1 = RandomBondDelete(p=0.5)(data, 42)
    bond_masked2 = RandomBondDelete(p=0.5)(data, 42)
    bond_masked3 = RandomBondDelete(p=0.5)(data, 4)
    assert torch.all(torch.eq(bond_masked1.edge_index, bond_masked2.edge_index))
    assert torch.any(torch.eq(bond_masked1.edge_index, bond_masked3.edge_index))


def test_bond_delete_mol():

    # Dummy data set
    data = MoleculeDataset("", smiles_data=["C"], labels={'target': [1]}, task='classification')
    data.target = 'target'

    # Mask every atom
    atom_masked = RandomBondDelete(p=1)(data.__getitem__(0), None)
    assert atom_masked.edge_index.numel() == 0
    assert atom_masked.edge_attr.numel() == 0


def test_smiles2graph():
    data = smiles2graph("CC")
    assert torch.all(torch.eq(data.x, torch.Tensor([[5,0],[5,0]])))
    assert torch.all(torch.eq(data.edge_index, torch.Tensor([[0,1],[1,0]])))
    assert torch.all(torch.eq(data.edge_attr, torch.Tensor([[0,0],[0,0]])))


def test_molecule_data():
    # This has been tested locally
    datasets = [
            "QM7",
            "QM8",
            "QM9",

            "ESOL",
            "FreeSolv",
            "Lipophilicity",

            #"PCBA",
            "MUV",
            "HIV",
            "BACE",

            "BBBP",
            "Tox21",
            "ToxCast",
            "SIDER",
            "ClinTox"
    ]

    for ds in datasets:
        print("\nDATASET: {}".format(ds))
        data = MoleculeDatasetWrapper(ds)
        data = MoleculeDatasetWrapper(ds)
        train, valid, test = data.get_data_loaders()
    shutil.rmtree("./data_download")
    assert True


def test_composition():
    # Atom mask and Bond Delete tests ensure this runs properly also
    transform = Compose([
        RandomAtomMask(p=[0.1, 0.5]),
        RandomBondDelete(p=[0.6, 0.7])
    ])
    data = MoleculeDatasetWrapper("BACE", transform=transform, batch_size=1024, aug_time=3)
    train, valid, test = data.get_data_loaders()
    #shutil.rmtree("./data_download")
    assert True


def test_loading_multitask():
    transform = Compose([
        RandomAtomMask(p=[0.1, 0.5]),
        RandomBondDelete(p=[0.6, 0.7])
    ])
    dataset = MoleculeDatasetWrapper("ClinTox", transform=transform, batch_size=1024, aug_time=3)
    train, valid, test = dataset.get_data_loaders(['FDA_APPROVED', 'CT_TOX'])
    for idx, data in enumerate(train):
        if(idx == len(train)-1):
            continue
        assert list(data.y.shape) == [1024, 2]

    train, valid, test = dataset.get_data_loaders(['CT_TOX'])
    for idx, data in enumerate(train):
        if(idx == len(train)-1):
            continue
        assert list(data.y.shape) == [1024, 1]


    dataset = MoleculeDatasetWrapper("SIDER", transform=transform, batch_size=1, aug_time=3)
    train, valid, test = dataset.get_data_loaders('all')
    for idx, data in enumerate(train):
        if(idx == len(train)-1):
            continue
        assert list(data.y.shape) == [1, 27]

    # Remove downloaded data
    shutil.rmtree("./data_download")


def test_consistent_augment():
    aug_time = 2
    transform = Compose([
        RandomAtomMask(p=0.5),
    ])
    data = MoleculeDataset("BACE", transform=transform, batch_size=1024, aug_time=aug_time,
                           test_mode=False)

    # Dataset retains a copy of the original data
    original = data.__getitem__(0)
    first_aug = data.__getitem__(1)
    second_aug = data.__getitem__(2)
    second_aug_again = data.__getitem__(2)
    assert not torch.equal(original.x, first_aug.x)
    assert not torch.equal(original.x, second_aug.x)
    assert not torch.equal(first_aug.x, second_aug.x)
    assert torch.equal(second_aug.x, second_aug_again.x)

    # Check that original is retained. No masking is done.
    for i in range(0, len(data),  aug_time+1):
        assert all(data.__getitem__(i).x.numpy()[:,0] != 119)

    # Turn off retaining original
    transform = Compose([
        RandomAtomMask(p=1.),
    ])
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        data = MoleculeDataset("BACE", transform=transform, batch_size=1024, aug_time=aug_time,
                               test_mode=False, augment_original=True)
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        warn_string = "Augmenting original dataset may lead to unexpected results."
        assert warn_string in str(w[-1].message)

    # Check that every atom is masked
    for i in range(0, len(data)):
        assert all(data.__getitem__(i).x.numpy()[:,0] == 119)

    shutil.rmtree("./data_download")


def test_all_augment():
    aug_time = 1
    transform = Compose([
        RandomAtomMask(1.0),
        RandomBondDelete(1.0),
        MotifRemoval(similarity_threshold=0.9)
    ])
    data = MoleculeDatasetWrapper("ClinTox", transform=transform, batch_size=1)
    train, valid, test = data.get_data_loaders("all")

    # Checks to make sure all original data is found in loaders
    for d in data:
        in_train = d.smiles in train.dataset.smiles_data
        in_valid = d.smiles in valid.dataset.smiles_data
        in_test = d.smiles in test.dataset.smiles_data
        assert in_train or in_valid or in_test
        assert (int(in_train) + int(in_valid) + int(in_test)) == 1

    transform = Compose([
        RandomAtomMask(1.0),
        RandomBondDelete(1.0),
        MotifRemoval(similarity_threshold=0.1)
    ])
    data = MoleculeDatasetWrapper("ClinTox", transform=transform, batch_size=1)
    train, valid, test = data.get_data_loaders()
    for d in data:
        in_train = d.smiles in train.dataset.smiles_data
        in_valid = d.smiles in valid.dataset.smiles_data
        in_test = d.smiles in test.dataset.smiles_data
        assert in_train or in_valid or in_test
        assert (int(in_train) + int(in_valid) + int(in_test)) == 1



#if __name__ == '__main__':
    #test_smiles2graph()
    #test_atom_mask()
    #test_bond_delete()
    #test_atom_mask_mol()
    #test_bond_delete_mol()
    #test_molecule_data()
    #test_composition()
    #test_loading_multitask()
    #test_consistent_augment()
    #test_all_augment()
