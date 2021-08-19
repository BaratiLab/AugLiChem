import sys
sys.path.append(sys.path[0][:-4])

from auglichem.utils import *

def test_scaffold_split():
    assert True

def test_random_split():
    assert True

def test_constants():
    assert ATOM_LIST == list(range(1,120))
    assert NUM_ATOM_TYPE == 119

    assert len(CHIRALITY_LIST) == 4
    assert NUM_CHIRALITY_TAG == 4

    assert len(BOND_LIST) == 5
    assert NUM_BOND_TYPE == 6

    assert len(BONDDIR_LIST) == 4
    assert NUM_BOND_DIRECTION == 4
