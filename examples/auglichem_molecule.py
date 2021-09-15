import sys
sys.path.append('../')

import os
from rdkit import Chem
import auglichem.molecule._transforms as auglichem_mol


smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
mol_graph = auglichem_mol._mol2graph(mol)
print(mol_graph)

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
mol_aug = auglichem_mol.RandomAtomMask(p=0.2)
aug_mol_graph = mol_aug.apply_transform(mol, seed=0)
print(aug_mol_graph)

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
mol_aug = auglichem_mol.RandomBondDelete(p=0.2)
aug_mol_graph = mol_aug.apply_transform(mol, seed=0)
print(aug_mol_graph)

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
mol_aug = auglichem_mol.RandomSubgraphRemoval(p=0.2)
aug_mol_graph = mol_aug.apply_transform(mol, seed=0)
print(aug_mol_graph)

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
mol_aug = auglichem_mol.RandomMotifRemoval(p=0.2)
aug_mol_graph = mol_aug.apply_transform(mol, seed=0)
print(aug_mol_graph)
