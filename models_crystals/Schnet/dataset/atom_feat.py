import csv
import json
import os
import random
import numpy as np
from ase.io import read as ase_read


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_feat(self, atom_type):
        # assert atom_type in self.atom_types
        return self._embedding[atom_type]
    
    def get_atom_features(self, atomics):
        return np.vstack([self.get_atom_feat(i) for i in atomics])

    def get_length(self):
        return len(self._embedding[1])


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


if __name__ == "__main__":
    root_dir = "data" 
    atom_init_file = os.path.join(root_dir, 'atom_init.json')
    assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
    ari = AtomCustomJSONInitializer(atom_init_file)

    crys = ase_read('data/Is_Metal_cifs/70685.cif')
    atomics = crys.get_atomic_numbers()
    # atom_feat = np.vstack([ari.get_atom_feat(i) for i in atomics])
    atom_feat = ari.get_atom_features(atomics)

    print(atom_feat)
    print(atom_feat.shape)
    print(ari.get_length())
