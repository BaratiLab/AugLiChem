from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

ATOM_LIST = list(range(1,120)) # Includes mask token
NUM_ATOM_TYPE = len(ATOM_LIST)

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
NUM_CHIRALITY_TAG = len(CHIRALITY_LIST)

BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC,
    BT.UNSPECIFIED,
]
NUM_BOND_TYPE = len(BOND_LIST) + 1 # including aromatic and self-loop edge


BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE
]
NUM_BOND_DIRECTION = len(BONDDIR_LIST)
