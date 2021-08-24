import pandas as pd
import os
import pathlib
import urllib
from tqdm import tqdm
import csv
import gzip
from rdkit import Chem
import numpy as np

#TODO:
#  1) Add functionality to download csv.gz datasets (which looks like most of them)
#  2) Clean up code and add proper comments
#  3) Check integrity?

USER_AGENT = "auglichem"

def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
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


def _load_data(dataset=None, data_path='./data_download/'):
    if(dataset == 'BACE'):
        task = 'classification'
        target = ['Class']
        csv_file_path = download_url(
                    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
                    data_path)

    elif(dataset == 'ClinTox'):
        task = 'classification'
        target = ['CT_TOX', 'FDA_APPROVED']
        csv_file_path = download_url(
                    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
                    data_path)

    elif(dataset == 'BBBP'):
        task = 'classification'
        target = ['p_np']
        csv_file_path = download_url(
                   "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
                   data_path)

    elif(dataset == 'SIDER'):
        task = 'classification'
        target = [
            "Hepatobiliary disorders",
            "Metabolism and nutrition disorders",
            "Product issues",
            "Eye disorders",
            "Investigations",
            "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders",
            "Social circumstances",
            "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders",
            "Surgical and medical procedures",
            "Vascular disorders",
            "Blood and lymphatic system disorders",
            "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders",
            "Psychiatric disorders",
            "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders",
            "Cardiac disorders",
            "Nervous system disorders",
            "Injury, poisoning and procedural complications"
        ]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
                data_path)

    return csv_file_path, target, task


def _process_csv(csv_file, target, task):
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    smiles_data, labels = [], {}
    for t in target:
        labels[t] = []

    for i, row in enumerate(csv_reader):
        # Skip header
        if i == 0:
            continue

        # smiles = row[3]
        try:
            smiles = row['smiles']
        except KeyError:
            smiles = row['mol']
        for idx, t in enumerate(target):
            label = row[t]
            mol = Chem.MolFromSmiles(smiles)
            if mol != None and label != '':
                if(idx == 0):
                    smiles_data.append(smiles)
                if task == 'classification':
                    labels[t].append(int(label))
                elif task == 'regression':
                    labels[t].append(float(label))
                else:
                    ValueError('task must be either regression or classification')

    # Recast lables to numpy arrays
    for t in target:
        labels[t] = np.array(labels[t])

    return smiles_data, labels, task


def read_smiles(dataset, data_path):

    # Create data download directory if it does not exist
    if(data_path is None):
        data_path = './data_download'
    if(not os.path.exists(data_path)):
        os.mkdir(data_path)

    # Download files if not already there
    csv_file_path, target, task = _load_data(dataset, data_path)
    if(".gz" in csv_file_path):
        with gzip.open(csv_file_path, 'rt') as csv_file:
            return _process_csv(csv_file, target, task)
    else:
        with open(csv_file_path) as csv_file:
            return _process_csv(csv_file, target, task)

    return smiles_data, np.array(labels)

