#TODO: make device (cpu/gpu) an input option, default CPU

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
from sklearn import preprocessing

from torch_geometric.data import Data, Dataset, DataLoader

import pandas as pd
import warnings

from ._knn import knn_graph
from ._load_sets import AtomCustomJSONInitializer

from auglichem.crystal._transforms import (
        RotationTransformation,
        PerturbStructureTransformation,
        RemoveSitesTransformation,
        SupercellTransformation,
        TranslateSitesTransformation,
        CubicSupercellTransformation,
        PrimitiveCellTransformation,
        SwapAxesTransformation,
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
from ._load_sets import read_crystal


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
    """
    def __init__(self, dataset, data_path=None, transform=None, id_prop_augment=None,
                 atom_init_file=None, id_prop_file=None, ari=None,
                 radius=8, dmin=0, step=0.2,
                 on_the_fly_augment=False, kfolds=0,
                 num_neighbors=8, max_num_nbr=12, seed=None, cgcnn=False):
        """
            Inputs:
            -------
            dataset (str): One of our 5 datasets: lanthanides, perosvkites, band_gap,
                           fermi_energy, or formation_energy.
            data_path (str, optional): Path for our data, automatically checks if it is there
                           and downloads the data if it isn't.
            transform (list of AbstractTransformations, optional): The transformations
                           to do on our CIF files
            id_prop_augment (np.array of floats, shape=(N,2), optional):
            atom_init_file (str, optional):
            id_prop_file (str, optional):
            ari (CustomAtomJSONInitializer, optional):
            radius (float, optional, default=0):
            dmin (float, optional, default=0):
            step (float, optional, default=0.2):
            on_the_fly_augment (bool, optional, default=Faalse): Setting to true augments
                           cif files on-the-fly, like in MoleculeDataset. This feature is
                           experimental and may significantly slow down run times.
            kfolds (int, optional, default=0): Number of folds to use in k-fold cross
                           validation. Must be >= 2 in order to run.
            num_neighbors (int, optional, default=8): Number of neighbors to include for
                           torch_geometric based models.
            max_num_nbr (int, optional, default=12): Maximum number of neighboring atoms used
                           when building the crystal graph for CGCNN.
            random_seed (int, optional): Random seed  to use for data splitting.
            cgcnn (bool, optional, default=False): If using built-in CGCNN model, must be set
                           to True.

            Outputs:
            --------
            None

        """

        super(Dataset, self).__init__()
        
        self.dataset = dataset
        self.data_path = data_path
        self.transform = transform
        self._augmented = False # To control runaway augmentation
        self.num_neighbors = num_neighbors

        self.seed = seed


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

        # Seeding used for reproducible tranformations
        self.reproduce_seeds = list(range(self.__len__()))
        np.random.shuffle(self.reproduce_seeds)

        self.on_the_fly_augment = on_the_fly_augment
        if(self.on_the_fly_augment):
            warnings.warn("On-the-fly augmentations for crystals is untested and can lead to memory issues. Use with caution.", category=RuntimeWarning, stacklevel=2)

        # Set up for k-fold CV
        if(kfolds > 1):
            self._k_fold_cv = True
            self.kfolds = kfolds
            self._k_fold_cross_validation()
        elif(kfolds == 1):
            raise ValueError("kfolds > 1 to run.")
        else:
            self._k_fold_cv = False

        # Must be true to use built-in CGCNN model
        self._cgcnn = cgcnn

        # Set atom featurizer
        self.atom_featurizer = AtomCustomJSONInitializer(os.path.join(self.data_path,
                                   'atom_init.json'))


    def _aug_name(self, transformation):
        if(isinstance(transformation, RotationTransformation)):
            suffix = '_rotated'
        elif(isinstance(transformation, PerturbStructureTransformation)):
            suffix = '_perturbed'
        elif(isinstance(transformation, RemoveSitesTransformation)):
            suffix = '_remove_sites'
        elif(isinstance(transformation, SupercellTransformation)):
            suffix = '_supercell'
        elif(isinstance(transformation, TranslateSitesTransformation)):
            suffix = '_translate'
        elif(isinstance(transformation, CubicSupercellTransformation)):
            suffix = '_cubic_supercell'
        elif(isinstance(transformation, PrimitiveCellTransformation)):
            suffix = '_primitive_cell'
        elif(isinstance(transformation, SwapAxesTransformation)):
            suffix = '_swapaxes'
        return suffix


    def data_augmentation(self, transform=None):
        '''
            Function call to deliberately augment the data. Transformations are done one at
            a time. For example, if we're using the RotationTransformation and
            SupercellTransformation, 0.cif will turn into 0.cif, 0_supercell.cif, and
            0_rotated.cif. Note: 0_supercell_rotated.cif WILL NOT be created.

            input:
            -----------------------
            transformation (list of AbstractTransformations): The transformations

        '''
        if(self._augmented):
            print("Augmentation has already been done.")
            return

        if(self.on_the_fly_augment):
            print("Augmentation will be done on-the-fly.")
            return
        
        # Copy directory and rename it to augmented
        if(self._k_fold_cv):
            # Copy directory
            shutil.copytree(self.data_path,
                            self.data_path + "_augmented_{}folds".format(self.kfolds),
                            dirs_exist_ok=True)

            # Remove k-fold files from original directory
            for i in range(self.kfolds):
                os.remove(self.data_path + "/id_prop_train_{}.csv".format(i))
                os.remove(self.data_path + "/id_prop_test_{}.csv".format(i))
            
            # Update data path
            self.data_path += "_augmented_{}folds".format(self.kfolds)
        else:
            shutil.copytree(self.data_path, self.data_path + "_augmented", dirs_exist_ok=True)
            self.data_path += "_augmented"

        self.atom_featurizer = AtomCustomJSONInitializer(os.path.join(self.data_path,
                               'atom_init.json'))

        # Check transforms
        if(not isinstance(transform, list)):
            transform = [transform]

        # Do augmentations
        new_id_prop_augment = []
        for id_prop in tqdm(self.id_prop_augment):
            new_id_prop_augment.append((id_prop[0], id_prop[1]))

            # Transform crystal
            if(transform == [None] and self.transform is None):
                break

            for t in transform:

                # Get augmented file name
                id_name = id_prop[0] + self._aug_name(t)
                new_id_prop_augment.append((id_name,id_prop[1]))
                
                # Don't create file if it already exists
                if(os.path.exists(self.data_path + '/' + id_name + '.cif')):
                    continue

                try:
                    seed_idx = np.argwhere(self.id_prop_augment[:,0] == id_prop[0])[0][0]
                    aug_crystal = t.apply_transformation(
                                    Structure.from_file(os.path.join(self.data_path,
                                    id_prop[0]+'.cif')),
                                    seed=self.reproduce_seeds[seed_idx])
                except IndexError:
                    print(int(id_prop[0]))
                    print(len(self.reproduce_seeds))
                    raise
                            
                cif.CifWriter(aug_crystal).write_file(self.data_path + '/' + id_name + '.cif')

        if(not self._k_fold_cv):
            self.id_prop_augment = np.array(new_id_prop_augment)
        else:
            self.id_prop_augment_all = np.array(new_id_prop_augment)
        self._augmented = True


    def _updated_train_cifs(self, train_idx, num_transform):
        '''
            When doing k-fold CV. This function adds the augmented cif names to the train_idx
        '''
        updated_train_idx = []
        for idx in train_idx:
            num_idx = int(np.argwhere(self.id_prop_augment[:,0] == idx[0])[0][0])
            for jdx in range(num_transform+1):
                updated_train_idx.append(self.id_prop_augment_all[(num_transform+1)*num_idx+jdx])
        
        return np.array(updated_train_idx)


    def _k_fold_cross_validation(self):
        '''
            k-fold CV data splitting function. Uses class attributes to split into k folds.
            Works by shuffling original data then selecting folds one at a time.
        '''
        # Set seed and shuffle data
        np.random.seed(self.seed)
        np.random.shuffle(self.id_prop_augment)

        frac = 1./self.kfolds
        N = len(self.id_prop_augment)
        for i in range(self.kfolds):

            # Get all idxs
            idxs = list(range(N))

            # Get train and validation idxs
            test_idxs = idxs[int(i*frac*N):int((i+1)*frac*N)]
            del idxs[int(i*frac*N):int((i+1)*frac*N)]

            # Get train and validation sets
            test_set = np.array(self.id_prop_augment)[test_idxs]
            train_set = np.array(self.id_prop_augment)[idxs]

            # Save files
            np.savetxt(self.data_path + "/id_prop_test_{}.csv".format(i), test_set.astype(str),
                       delimiter=',', fmt="%s")
            np.savetxt(self.data_path + "/id_prop_train_{}.csv".format(i), train_set.astype(str),
                       delimiter=',', fmt="%s")


    def __len__(self):
        return len(self.id_prop_augment)


    def _gaussian_distance(self, distances, dmin, dmax, step, var=None):
        if var is None:
            var = step
        self.filter = np.arange(dmin, dmax+step, step)
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / var**2)


    def _getitem_crystal(self, idx):
        """
            Loads in and processes cif file for CGCNN at call time
        """
        cif_id, target = self.id_prop_augment[idx]
        crystal = Structure.from_file(os.path.join(self.data_path,
                                                   cif_id+'.cif'))

        if(self.on_the_fly_augment):
            if(self.transform is None):
                raise ValueError("Transformations need to be specified.")
            for t in self.transform:
                crystal = t.apply_transfromation(crystal)

        atom_fea = np.vstack([self.ari.get_atom_feat(crystal[i].specie.number)
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


    def _getitem_knn(self, idx):
        """
            Loads in and processes cif file at call time. Returns torch_geometric Data
            object, and uses knn to find atom neighbors.
        """
        # get the cif id and path
        augment_cif_id, self.aug_labels = self.id_prop_augment[idx]
        augment_cryst_path = os.path.join(self.data_path, augment_cif_id + '.cif')

        self.aug_labels = np.array(self.aug_labels)

        # read cif using pymatgen
        aug_crys = Structure.from_file(augment_cryst_path)
        pos = aug_crys.frac_coords
        atom_indices = list(aug_crys.atomic_numbers)
        cell = aug_crys.lattice.get_cartesian_coords(1)
        feat = self.atom_featurizer.get_atom_features(atom_indices)
        N = len(pos)
        y = self.aug_labels
        y = torch.tensor(float(y), dtype=torch.float).view(1,1)
        atomics = []
        for index in atom_indices:
            atomics.append(ATOM_LIST.index(index))
        atomics = torch.tensor(atomics, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        feat = torch.tensor(feat, dtype=torch.float)
        edge_index = knn_graph(pos, k=self.num_neighbors, loop=False)
        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)

        # build the PyG graph
        data = Data(
            atomics=atomics, pos=pos, feat=feat, y=y,
            edge_index=edge_index, edge_attr=edge_attr
        )

        return data


    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        """
            Loads and processes cif file. Takes care of cgcnn vs. torch_geometric models.
        """
        if(self._cgcnn):
            return self._getitem_crystal(idx)
        else:
            return self._getitem_knn(idx)


class CrystalDatasetWrapper(CrystalDataset):
    def __init__(self, dataset, transform=None, split="random", batch_size=64, num_workers=0,
                 valid_size=0.1, test_size=0.1, data_path=None, target=None, kfolds=0,
                 seed=None, cgcnn=False, **kwargs):
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
            kfolds (int, default=0, folds > 1): Number of folds to use in k-fold cross
                        validation. kfolds > 1 for data to be split
            seed (int, optional, default=None): Random seed set for data shuffling
            cgcnn (bool, optional, default=False): Set to True is using built-in CGCNN model.
             

            outputs:
            -------------------------
            None
        '''
        super().__init__(dataset, data_path, transform, kfolds=kfolds, seed=seed, cgcnn=cgcnn)
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.id_prop_augment = np.asarray(self.id_prop_augment)
        self.collate_fn = collate_pool
        self.cgcnn = cgcnn


    def _match_idx(self, cif_idxs):
        '''
            Match function that converts cif idxs to the index it appears at in id_prop_augment
        '''
        idxs = []
        for i in cif_idxs:
            idxs.append(self.id_prop_augment[np.argwhere(self.id_prop_augment == \
                                                         str(int(i[0])))[0][0]])
        return np.array(idxs)

    
    def _get_split_idxs(self, target=None, transform=None, fold=None):
        """
            This function returns the train, validation, and test id_prop_augment data..
        """
        if(not target and self.target is None):
             self.target = list(self.labels.keys())[0]

        # Get indices of data splits
        if(self.split == 'scaffold' and not self._k_fold_cv):
            raise NotImplementedError("Scaffold only supports molecules currently.")
        elif(self.split == 'random' and not self._k_fold_cv):
            train_idx, valid_idx, test_idx = random_split(self.id_prop_augment[:,0],
                                                          self.valid_size, self.test_size,
                                                          self.seed)
            return train_idx, valid_idx, test_idx

        # If using k-fold CV
        elif(fold is not None and not self._k_fold_cv):
            raise ValueError("Fold number specified but k-fold CV not called.")
        elif(fold is None and self._k_fold_cv):
            raise ValueError("Please select a fold < {}".format(self.kfolds))
        elif(fold >= self.kfolds):
            raise ValueError("Please select a fold < {}".format(self.kfolds))
        elif(fold is not None):
            print("Ignoring splitting. Using pre-split k folds.")

            #TODO: setting type here as int may not be helpful, could be optimized
            # Get train set
            train_cif_idx = np.loadtxt(self.data_path + "/id_prop_train_{}.csv".format(fold),
                                   delimiter=',')
            train_idx = self._match_idx(train_cif_idx)

            # Get validation set
            valid_size = int(len(train_idx)*self.valid_size)
            valid_idx = train_idx[:valid_size]
            train_idx = train_idx[valid_size:]

            # Update idx csv files
            np.savetxt(self.data_path + "/id_prop_train_{}.csv".format(fold),
                       train_idx.astype(str), delimiter=',', fmt="%s")
            np.savetxt(self.data_path + "/id_prop_valid_{}.csv".format(fold),
                       valid_idx.astype(str), delimiter=',', fmt="%s")

            # Get test set
            test_cif_idx = np.loadtxt(self.data_path + "/id_prop_test_{}.csv".format(fold),
                                   delimiter=',')
            test_idx = self._match_idx(test_cif_idx)

            # Do data transformation. With k_fold_cv, self.id_prop_augment is updated later
            self.data_augmentation(transform)
            self.atom_featurizer = AtomCustomJSONInitializer(os.path.join(self.data_path,
                                   'atom_init.json'))
            return train_idx, valid_idx, test_idx

        else:
            raise ValueError("Please select scaffold or random split")


    def get_data_loaders(self, target=None, transform=None, fold=None):
        '''
            This function splits the data into train, validation, and test data loaders for
            ease of use in model training

            inputs:
            -------------------------
            target (str, optional, default=None): The target label for training. Currently all
                                        crystal datasets are single-target, and so this parameter
                                        is truly optional.
            transform (AbstractTransformation, optional, default=None): The data transformation
                                        we will use for data augmentation.
            fold (int, optiona, default=None): Which of k folds to use for training. Will
                                        throw an error if specified and k-fold CV is not
                                        done in the class instantiaion. This overrides
                                        valid_size and test_size

            outputs:
            -------------------------
            train/valid/test_loader (DataLoader): The torch_geometric data loader initialized.
                                        The data loader can be iterated over, returning batches
                                        of the data specified by `batch_size`.
        '''
        train_idx, valid_idx, test_idx = self._get_split_idxs(target, transform, fold)

            
        # Get train loader
        if(self._k_fold_cv): # Need to add in augmented cif files to id_prop_augment
            transform = [transform] if(not isinstance(transform, list)) else transform
            train_id_prop_augment = self._updated_train_cifs(train_idx, len(transform))
            valid_id_prop_augment = valid_idx
            test_id_prop_augment = test_idx
        else: # Augmented cif files will be put in id_prop_augment
            train_id_prop_augment = self.id_prop_augment[train_idx]
            valid_id_prop_augment = self.id_prop_augment[valid_idx]
            test_id_prop_augment = self.id_prop_augment[test_idx]
        train_set = CrystalDataset(self.dataset, self.data_path, self.transform,
                             train_id_prop_augment,
                             atom_init_file=self.atom_init_file, id_prop_file=self.id_prop_file,
                             ari=self.ari, cgcnn=self.cgcnn)
        train_set._k_fold_cv = self._k_fold_cv


        # Augment only training data
        if(transform and not self._k_fold_cv):
            train_set.data_augmentation(transform)

        # torch_geometric does not require collate_fn, CGCNN requires torch Dataset/Loader
        if(not(self._cgcnn)):
            self.collate_fn = None
            from torch_geometric.data import Data, DataLoader
        else:
            from torch.utils.data import DataLoader

        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn, shuffle=True)

        # Get val loader
        valid_set = CrystalDataset(self.dataset,
                             data_path=self.data_path,
                             transform=self.transform,
                             id_prop_augment=valid_id_prop_augment,
                             atom_init_file=self.atom_init_file,
                             id_prop_file=self.id_prop_file,
                             ari=self.ari, cgcnn=self.cgcnn)
        valid_loader = DataLoader(valid_set, batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn, shuffle=True)
        valid_set._k_fold_cv = self._k_fold_cv


        # Get test loader
        test_set = CrystalDataset(self.dataset,
                             data_path=self.data_path,
                             transform=self.transform,
                             id_prop_augment=test_id_prop_augment,
                             atom_init_file=self.atom_init_file,
                             id_prop_file=self.id_prop_file,
                             ari=self.ari, cgcnn=self.cgcnn)
        test_loader = DataLoader(test_set, batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn, shuffle=True)
        return train_loader, valid_loader, test_loader
    
