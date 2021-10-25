from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import random

def random_split(dataset, valid_size, test_size, seed=None):

    # Set seed
    if(seed is not None):
        random.seed(seed)

    # Get indices
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = 1.0 - valid_size - test_size

    train_cutoff = int(train_size * total_size)
    valid_cutoff = int((train_size + valid_size) * total_size)

    train_idx = indices[:train_cutoff]
    valid_idx = indices[train_cutoff:valid_cutoff]
    test_idx = indices[valid_cutoff:]
    return train_idx, valid_idx, test_idx


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(smiles_data, log_every_n=1000):
    scaffolds = {}
    data_len = len(smiles_data)

    print("Generating scaffolds...")
    for ind, smiles in enumerate(smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(smiles_data, valid_size, test_size, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(smiles_data)

    train_cutoff = train_size * len(smiles_data)
    valid_cutoff = (train_size + valid_size) * len(smiles_data)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds
