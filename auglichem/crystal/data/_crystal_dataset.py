from __future__ import print_function, division

import csv
import functools
import json
import os
import shutil
import random
import warnings
import random
import numpy as np
import torch
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.io import cif
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import warnings

from auglichem.crystal._transforms import (
        RandomRotationTransformation,
        RandomPerturbStructureTransformation,
        RandomRemoveSitesTransformation,
        SupercellTransformation,
        RandomTranslateSitesTransformation,
        CubicSupercellTransformation,
        PrimitiveCellTransformation
)

from auglichem.utils import (
        ATOM_LIST,
        CHIRALITY_LIST,
        BOND_LIST,
        BONDDIR_LIST,
        random_split,
        scaffold_split,
        random_split
)
#from auglichem.crystal.data import AtomCustomJSONInitializer as AJI
from ._load_sets import AtomCustomJSONInitializer, read_crystal


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class CrystalDataset(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── 0.cif
    ├── 1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, dataset, data_path=None, transform=None, id_prop_augment=None,
                 atom_init_file=None, id_prop_file=None, ari=None,fold = 0,
                 max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, test_mode=True, on_the_fly_augment=False):

        super(Dataset, self).__init__()
        
        self.dataset = dataset
        self.data_path = data_path
        self.transform = transform
        self._augmented = False # To control runaway augmentation

        # No augmentation if no transform is specified
        if(self.transform is None):
            self.test_mode = True
        else:
            self.test_mode = test_mode

        # After specifying data set
        if(id_prop_augment is None):
            self.id_prop_file, self.atom_init_file, self.ari, self.data_path, \
            self.target, self.task = read_crystal(dataset, data_path)
        else:
            self.id_prop_file = id_prop_file
            self.atom_init_file = atom_init_file
            self.ari = ari
        
        self.max_num_nbr, self.radius = max_num_nbr, radius

        assert os.path.exists(self.data_path), 'root_dir does not exist!'
        assert os.path.exists(self.id_prop_file), 'id_prop_augment.csv does not exist!'.format(fold)
        
        if(id_prop_augment is None):
            with open(self.id_prop_file) as f:
                reader = csv.reader(f)
                self.id_prop_augment = [row for row in reader]
        else:
            self.id_prop_augment = id_prop_augment

        assert os.path.exists(self.atom_init_file), 'atom_init.json does not exist!'
        self.gdf = lambda dist: self._gaussian_distance(dist, dmin=dmin, dmax=self.radius,
                                                        step=step)

        self.on_the_fly_augment = on_the_fly_augment
        if(self.on_the_fly_augment):
            warnings.warn("On-the-fly augmentations for crystals is untested and can lead to memory issues. Use with caution.", category=RuntimeWarning, stacklevel=2)


    def _aug_name(self, transformation):
        if(isinstance(transformation, RandomRotationTransformation)):
            suffix = '_rotated'
        elif(isinstance(transformation, RandomPerturbStructureTransformation)):
            suffix = '_perturbed'
        elif(isinstance(transformation, RandomRemoveSitesTransformation)):
            suffix = '_remove_sites'
        elif(isinstance(transformation, SupercellTransformation)):
            suffix = '_supercell'
        elif(isinstance(transformation, RandomTranslateSitesTransformation)):
            suffix = '_translate'
        elif(isinstance(transformation, CubicSupercellTransformation)):
            suffix = '_cubic_supercell'
        elif(isinstance(transformation, PrimitiveCellTransformation)):
            suffix = '_primitive_cell'
        return suffix


    def data_augmentation(self, transform=None):
        '''
            Function call to deliberately augment the data

            input:
            -----------------------
            transformation (AbstractTransformation): 

        '''
        if(self._augmented):
            print("Augmentation has already been done.")
            return

        if(self.on_the_fly_augment):
            print("Augmentation will be done on-the-fly.")
            return
        
        # Copy directory and rename it to augmented
        shutil.copytree(self.data_path, self.data_path + "_augmented", dirs_exist_ok=True)
        self.data_path += "_augmented"

        # Check transforms
        if(transform is None and self.transform is None):
            raise ValueError("No transform specified.")
        elif(not isinstance(transform, list)):
            transform = [transform]

        # Do augmentations
        new_id_prop_augment = []
        for id_prop in tqdm(self.id_prop_augment):
            new_id_prop_augment.append((id_prop[0], id_prop[1]))
            for t in transform:

                # Get augmented file name
                id_name = id_prop[0] + self._aug_name(t)
                new_id_prop_augment.append((id_name,id_prop[1]))
                
                # Don't create file if it already exists
                if(os.path.exists(self.data_path + '/' + id_name + '.cif')):
                    continue

                # Transform crystal
                aug_crystal = t.apply_transformation(
                                    Structure.from_file(os.path.join(self.data_path,
                                    id_prop[0]+'.cif')))
                cif.CifWriter(aug_crystal).write_file(self.data_path + '/' + id_name + '.cif')

        self.id_prop_augment = np.array(new_id_prop_augment)
        self._augmented = True


    def __len__(self):
        return len(self.id_prop_augment)

    def _gaussian_distance(self, distances, dmin, dmax, step, var=None):
        if var is None:
            var = step
        self.filter = np.arange(dmin, dmax+step, step)
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / var**2)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        #print(idx)
        cif_id, target = self.id_prop_augment[idx]
        crystal = Structure.from_file(os.path.join(self.data_path,
                                                   cif_id+'.cif'))

        if(self.on_the_fly_augment):
            if(self.transform is None):
                raise ValueError("Transformations need to be specified.")
            for t in self.transform:
                crystal = t.apply_transfromation(crystal)

        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


class CrystalDatasetWrapper(CrystalDataset):
    def __init__(self, dataset, transform=None, split="random", batch_size=64, num_workers=0,
                 valid_size=0.1, test_size=0.1, data_path=None, target=None,
                 **kwargs):
        '''
            Wrapper Class to handle splitting dataset into train, validation, and test sets

            inputs:
            -------------------------
            dataset (str): One of our dataset: lanthanides, perovskites, band_gap, fermi_energy,
                                               or formation_energy
            transform (AbstractTransformation, optional): A crystal transformation
            split (str, default=random): Method of splitting data into train, validation, and
                                         test
            batch_size (int, default=64): Data batch size for train_loader
            num_workers (int, default=0): Number of worker processes for parallel data loading
            valid_size (float, optional, between [0, 1]): Fraction of data used for validation
            test_size (float, optional, between [0, 1]): Fraction of data used for test
            data_path (str, optional default=None): specify path to save/lookup data. Default
                        creates `data_download` directory and stores data there
            target (str, optional, default=None): Target variable
            seed (int, optional, default=None): Random seed to use for reproducibility

            inputs:
            -------------------------
            None
        '''
        super().__init__(dataset, data_path, transform)
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.id_prop_augment = np.asarray(self.id_prop_augment)

        # What is this?
        self.collate_fn = collate_pool
        #self.cif_data = np.asarray(self.cif_data) # Might need to be different
        

    def get_data_loaders(self, target=None, transform=None):
        #TODO: Break down into Dataloaders for train/val/test
        if(not target and self.target is None):
             self.target = list(self.labels.keys())[0]

        # Get indices of data splits
        #TODO: Include different splits
        if(self.split == 'scaffold'):
            raise NotImplementedError("Scaffold only supports molecules currently.")
        elif(self.split == 'random'):
            train_idx, valid_idx, test_idx = random_split(self.id_prop_augment[:,0],
                                                          self.valid_size, self.test_size)
        else:
            raise ValueError("Please select scaffold or random split")

        # Need to pass in id_prop_augment with indices
        train_set = CrystalDataset(self.dataset, self.data_path, self.transform, self.id_prop_augment[train_idx],
                             atom_init_file=self.atom_init_file, id_prop_file=self.id_prop_file,
                             ari=self.ari)
        # Augment only training data
        train_set.data_augmentation(transform)

        valid_set = CrystalDataset(self.dataset, self.data_path, self.transform, self.id_prop_augment[valid_idx],
                             atom_init_file=self.atom_init_file, id_prop_file=self.id_prop_file,
                             ari=self.ari)
        test_set = CrystalDataset(self.dataset, self.data_path, self.transform, self.id_prop_augment[test_idx],
                             atom_init_file=self.atom_init_file, id_prop_file=self.id_prop_file,
                             ari=self.ari)

        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn, drop_last=True, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=len(valid_set),
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=len(test_set),
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn, drop_last=True, shuffle=True)
        return train_loader, valid_loader, test_loader
    
