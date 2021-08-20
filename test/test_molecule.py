import sys
sys.path.append(sys.path[0][:-4])

import shutil

from auglichem.molecule import smiles2graph, RandomAtomMask, RandomBondDelete
from auglichem.molecule.data import MoleculeData


def test_atom_mask():
    assert True

def test_bond_delete():
    assert True

def test_smiles2graph():
    assert True

def test_molecule_data():
    MoleculeData("SIDER")
    shutil.rmtree("./data_download")

if __name__ == '__main__':
    test_molecule_data()
