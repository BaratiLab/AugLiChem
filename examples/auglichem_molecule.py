import sys
sys.path.append('.')

import os
import auglichem.molecule.transforms as auglichem_mol


smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
mol_graph = auglichem_mol.smiles2graph(smiles)
print(mol_graph)

mol_aug = auglichem_mol.RandomAtomMask(p=0.2)
aug_mol_graph = mol_aug.apply_transform(mol_graph)
print(aug_mol_graph)

mol_aug = auglichem_mol.RandomBondDelete(p=0.2)
aug_mol_graph = mol_aug.apply_transform(aug_mol_graph)
print(aug_mol_graph)
