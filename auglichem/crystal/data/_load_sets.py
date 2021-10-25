import os
import pathlib
import urllib
from tqdm import tqdm
import csv
import gzip
import sys
import numpy as np
import csv
import json

from torch.utils.data.dataloader import default_collate

USER_AGENT = "auglichem"

#TODO: Automated data loading once we host the datasets

def get_train_val_test_loader(dataset, dataset_train, idx_map, collate_fn=default_collate, 
                              fold = 0,batch_size=64, train_ratio=None,
                              val_ratio=0.2, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, num_aug = 4, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)

    random.shuffle(indices)
    train_idx = indices[:train_size]
    train_idx_augment = []


    for i in range (len(train_idx)):
        #true_index = idx_map[train_idx[i]]
        idx_correction = num_aug*train_idx[i]
        # if train_idx[i]>15142:
        #     print(i)
        add_1 = idx_correction + 1
        add_2 = idx_correction + 2
        add_3 = idx_correction + 3
        add_  = idx_correction
        train_idx_augment.append(add_1)
        train_idx_augment.append(add_2)
        train_idx_augment.append(add_3)
        train_idx_augment.append(add_)
    	# train_idx_augment.append(add_4)
    	# train_idx_augment.append(add_5)
            	# add_4 = train_idx[i] + idx_correction + 4
    	# add_5 = train_idx[i] + idx_correction + 5
    #print((train_idx_augment))


    train_sampler = SubsetRandomSampler(train_idx_augment)

    val_sampler = SubsetRandomSampler(
        indices[train_size:])
    val_sampler = SubsetRandomSampler(
        indices[train_size:])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])


    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

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
        f.close()
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT    })) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)

def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of
                                  the URL
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if(os.path.isfile(fpath)):
        print("Using: {}".format(fpath))
        return fpath

    # download the file
    try:
        print('Downloading ' + url + ' to ' + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + fpath)
            _urlretrieve(url, fpath)
        else:
            raise e

    return fpath

def _load_data(dataset, data_path='./data_download'):
    ###
    #
    #   Need to host data sets and download them
    #
    ###
    if(dataset == 'lanthanides'):
        task = 'regression'
        target = ["formation_energy"] #TODO: Need to verify
        #csv_file_path = download_url("Nothing yet...", data_path)
        csv_file_path = data_path + "/lanths/id_prop.csv"
        embedding_path = data_path + "/lanths/atom_init.json"
        data_path += "/lanths"
    elif(dataset == 'band_gap'):
        print(data_path)
        task = 'regression'
        target = ["band_gap"]
        #csv_file_path = download_url("Nothing yet...", data_path)
        csv_file_path = data_path + "/band/id_prop.csv"
        embedding_path = data_path + "/band/atom_init.json"
        data_path += "./band/"
    elif(dataset == 'perovskites'):
        task = 'regression'
        target = ["energy"] #TODO: Need to verify
        #csv_file_path = download_url("Nothing yet...", data_path)
        csv_file_path = data_path + "/"
        embedding_path = data_path + "/"
        data_path = "./"
    elif(dataset == 'fermi_energy'):
        task = 'regression'
        target = ["fermi_energy"]
        #csv_file_path = download_url("Nothing yet...", data_path)
        csv_file_path = data_path + "/"
        embedding_path = data_path + "/"
        data_path = "./"
    elif(dataset == 'formation_energy'):
        task = 'regression'
        target = ["formation_energy"]
        #csv_file_path = download_url("Nothing yet...", data_path)
        csv_file_path = data_path + "/"
        embedding_path = data_path + "/"
        data_path = "./"
    else:
        raise ValueError("Please select one of the following datasets: lanthanides, band_gap, perovskites, fermi_energy, formation_energy")
        

    return data_path, embedding_path, csv_file_path, target, task


def read_crystal(dataset, data_path):

    # Create data download directory if it does not exist
    if(data_path is None):
        data_path = './data_download'
    if(not os.path.exists(data_path)):
        os.mkdir(data_path)

    # Download files if not already there
    data_path, embedding_path, csv_file_path, target, task = _load_data(dataset, data_path)
    return csv_file_path, \
           embedding_path, \
           AtomCustomJSONInitializer(embedding_path), \
           data_path, \
           target, \
           task

