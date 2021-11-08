import os
import urllib
from zipfile import ZipFile
from tqdm import tqdm

import requests
import zipfile

import numpy as np
import csv
import json

from torch.utils.data.dataloader import default_collate

USER_AGENT = "auglichem"

#TODO: Automated data loading once we host the datasets

# Data available at:
#        https://drive.google.com/drive/folders/1R8kjl1O1cGn-bzKij4IvzCrzLOUt0139?usp=sharing


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_feat(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def get_atom_features(self, atomics):
        return np.vstack([self.get_atom_feat(i) for i in atomics])

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


# Download functions taken from ManufactureNet
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def extract_files(name, root):
    Zip = ZipFile(name)
    Zip.extractall(root)


def remove_zip(name):
    os.remove(name)


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of
                                  the URL
    """
    URL = "https://docs.google.com/uc?export=download/"
    destination = root+filename
    if(os.path.isdir(destination[:-4])):
        print("Data found at: {}".format(destination[:-4]))
        return destination[:-4]
    print("Downloading data to: {}...".format(destination[:-4]))
    download_file_from_google_drive(url, destination)
    print("Extracting zipfile...")
    extract_files(destination, root)
    print("Removing zipfile...")
    remove_zip(destination)

    return destination[:-4]

def _load_data(dataset, data_path='./data_download'):
    '''
        Loads dataset, sets task to regression for the downloadable sets, gets the atom
        embedding file, and updates the data path.
    '''
    # Data hosting is done with google drive. You must create a publicly available link
    # using "get link", then convert it to a downloadable link here:
    # https://sites.google.com/site/gdocs2direct/
    if(dataset == 'lanthanides'):
        task = 'regression'
        target = ["formation_energy"] #TODO: Need to verify
        data_id = "1AB0XkOUdmOkxfKKjW_XkAxxGhAuA4F5k"
        csv_file_path = download_url(data_id, data_path, "/lanths.zip")
        csv_file_path = data_path + "/lanths/id_prop.csv"
        embedding_path = data_path + "/lanths/atom_init.json"
        data_path += "/lanths"
    elif(dataset == 'band_gap'):
        task = 'regression'
        target = ["band_gap"]
        data_id = "1pr8yxK3-BdfQwiVG_V8s_eCITY2l_ynC"
        csv_file_path = download_url(data_id, data_path, "/band.zip")
        csv_file_path = data_path + "/band/id_prop.csv"
        embedding_path = data_path + "/band/atom_init.json"
        data_path += "/band"
    elif(dataset == 'perovskites'):
        task = 'regression'
        target = ["energy"] #TODO: Need to verify
        data_id = "1y-SAMNBHrE9GUCuZofDuvLXc3MXGF7dG"
        csv_file_path = download_url(data_id, data_path, "/abx3_cifs.zip")
        csv_file_path = data_path + "/abx3_cifs/id_prop.csv"
        embedding_path = data_path + "/abx3_cifs/atom_init.json"
        data_path += "/abx3_cifs"
    elif(dataset == 'fermi_energy'):
        task = 'regression'
        target = ["fermi_energy"]
        data_id = "1m5I2MSfoT9WOlua4H4ZBYQS3dEzpmYid"
        csv_file_path = download_url(data_id, data_path, "/fermi.zip")
        csv_file_path = data_path + "/fermi/id_prop.csv"
        embedding_path = data_path + "/fermi/atom_init.json"
        data_path += "/fermi"
    elif(dataset == 'formation_energy'):
        task = 'regression'
        target = ["formation_energy"]
        data_id = "1NNmald3DfhHUr2gzN4V1ExsTp2cuYgvx"
        csv_file_path = download_url(data_id, data_path, "/FE.zip")
        csv_file_path = data_path + "/FE/id_prop.csv"
        embedding_path = data_path + "/FE/atom_init.json"
        data_path += "/FE"
    else:
        raise ValueError("Please select one of the following datasets: lanthanides, band_gap, perovskites, fermi_energy, formation_energy")
        

    return data_path, embedding_path, csv_file_path, target, task


def read_crystal(dataset, data_path):
    """
        Inputs:
        -------
        dataset (str): The dataset to be used, needs to be one of: lanthanides, band_gap,
                       perovskites, fermi_energy, or formation_energy
        data_path (str): The path to search for data. If the requested data set is not there,
                         the data is downloaded automatically and stored at data_path.

        Outputs:
        --------
        csv_file_path (str): The path of id_prop.csv in the data set directory.
        embedding_file_path (str): The path of [] in the data set directory.
        ari: (AtomCustomJSONInitializer): Object that initializes the vector representation
                                          of atoms.
        data_path (str): The relative path of the data.
        target (str): The name of the target variable.
        task (str): regression or classification. Currently only classification is supported.
    """

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

