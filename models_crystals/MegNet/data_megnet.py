import numpy as np

import csv
import functools
import json
import os
import random
import warnings
import numpy as np
from pymatgen.core.structure import Structure
import keras
from keras import utils
from keras.models import Sequential

class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self,root_dir, batch_size = 128, shuffle=True):
        '''Initialization'''
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        self.id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        self.list_IDs = np.arange(len(self.id_prop_data))
        self.on_epoch_end()


    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.id_prop_data) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.id_prop_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''Generates data containing batch_size samples''' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = []
        y = []

        # Generate data
        for i in range(len(list_IDs_temp)):
            cif_id, target = self.id_prop_data[list_IDs_temp[i]]
            # Store sample
            X.append(Structure.from_file(os.path.join(self.root_dir,cif_id+ '.cif')))

            # Store class
            y.append(target)
        print(X)
        print(y)
        return X, y