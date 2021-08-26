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

    elif(dataset == 'Tox21'):
        task = 'classification'
        target = [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53"
        ]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
                data_path)

    elif(dataset == 'HIV'):
        task = 'classification'
        target = ["HIV_active"]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
                data_path)

    elif(dataset == 'MUV'): # Too large to run locally
        task = 'classification'
        target = [
                "MUV-466",
                "MUV-548",
                "MUV-600",
                "MUV-644",
                "MUV-652",
                "MUV-692",
                "MUV-712",
                "MUV-713",
                "MUV-733",
                "MUV-737",
                "MUV-810",
                "MUV-832",
                "MUV-846",
                "MUV-852",
                "MUV-858",
                "MUV-859"
        ]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
                data_path)

    elif(dataset == 'FreeSolv'): # SMILES is second column
        task = 'regression'
        target = ["expt"]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
                data_path)

    elif(dataset == 'ESOL'): # SMILES is last column
        task = 'regression'
        target = ["measured log solubility in mols per litre"]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
                data_path)

    elif(dataset == 'Lipophilicity'):
        task = 'regression'
        target = ["exp"]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
                data_path)

    elif(dataset == 'QM7'):
        task = 'regression'
        target = ["u0_atom"]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv",
                data_path)

    elif(dataset == 'QM7b'):
        task = 'regression'
        target = ["u0_atom"]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.csv.gz",
                data_path)

    elif(dataset == 'QM8'):
        task = 'regression'
        target = [
                "E1-CC2",
                "E2-CC2",
                "f1-CC2",
                "f2-CC2",
                "E1-PBE0",
                "E2-PBE0",
                "f1-PBE0",
                "f2-PBE0",
                "E1-CAM",
                "E2-CAM",
                "f1-CAM",
                "f2-CAM"
        ]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv",
                data_path)

    elif(dataset == 'QM9'):
        task = 'regression'
        target = [
                "mu",
                "alpha",
                "homo",
                "lumo",
                "gap",
                "r2",
                "ZPVE",
                "U0",
                "U",
                "H",
                "G",
                "Cv",
                "omega1",

        ]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
                data_path)

    elif(dataset == "PCBA"):
        task = 'classification'
        target = [
                "PCBA-1030",
                "PCBA-1379",
                "PCBA-1452",
                "PCBA-1454",
                "PCBA-1457",
                "PCBA-1458",
                "PCBA-1460",
                "PCBA-1461",
                "PCBA-1468",
                "PCBA-1469",
                "PCBA-1471",
                "PCBA-1479",
                "PCBA-1631",
                "PCBA-1634",
                "PCBA-1688",
                "PCBA-1721",
                "PCBA-2100",
                "PCBA-2101",
                "PCBA-2147",
                "PCBA-2242",
                "PCBA-2326",
                "PCBA-2451",
                "PCBA-2517",
                "PCBA-2528",
                "PCBA-2546",
                "PCBA-2549",
                "PCBA-2551",
                "PCBA-2662",
                "PCBA-2675",
                "PCBA-2676",
                "PCBA-411",
                "PCBA-463254",
                "PCBA-485281",
                "PCBA-485290",
                "PCBA-485294",
                "PCBA-485297",
                "PCBA-485313",
                "PCBA-485314",
                "PCBA-485341",
                "PCBA-485349",
                "PCBA-485353",
                "PCBA-485360",
                "PCBA-485364",
                "PCBA-485367",
                "PCBA-492947",
                "PCBA-493208",
                "PCBA-504327",
                "PCBA-504332",
                "PCBA-504333",
                "PCBA-504339",
                "PCBA-504444",
                "PCBA-504466",
                "PCBA-504467",
                "PCBA-504706",
                "PCBA-504842",
                "PCBA-504845",
                "PCBA-504847",
                "PCBA-504891",
                "PCBA-540276",
                "PCBA-540317",
                "PCBA-588342",
                "PCBA-588453",
                "PCBA-588456",
                "PCBA-588579",
                "PCBA-588590",
                "PCBA-588591",
                "PCBA-588795",
                "PCBA-588855",
                "PCBA-602179",
                "PCBA-602233",
                "PCBA-602310",
                "PCBA-602313",
                "PCBA-602332",
                "PCBA-624170",
                "PCBA-624171",
                "PCBA-624173",
                "PCBA-624202",
                "PCBA-624246",
                "PCBA-624287",
                "PCBA-624288",
                "PCBA-624291",
                "PCBA-624296",
                "PCBA-624297",
                "PCBA-624417",
                "PCBA-651635",
                "PCBA-651644",
                "PCBA-651768",
                "PCBA-651965",
                "PCBA-652025",
                "PCBA-652104",
                "PCBA-652105",
                "PCBA-652106",
                "PCBA-686970",
                "PCBA-686978",
                "PCBA-686979",
                "PCBA-720504",
                "PCBA-720532",
                "PCBA-720542",
                "PCBA-720551",
                "PCBA-720553",
                "PCBA-720579",
                "PCBA-720580",
                "PCBA-720707",
                "PCBA-720708",
                "PCBA-720709",
                "PCBA-720711",
                "PCBA-743255",
                "PCBA-743266",
                "PCBA-875",
                "PCBA-881",
                "PCBA-883",
                "PCBA-884",
                "PCBA-885",
                "PCBA-887",
                "PCBA-891",
                "PCBA-899",
                "PCBA-902",
                "PCBA-903",
                "PCBA-904",
                "PCBA-912",
                "PCBA-914",
                "PCBA-915",
                "PCBA-924",
                "PCBA-925",
                "PCBA-926",
                "PCBA-927",
                "PCBA-938",
                "PCBA-995",
                ]
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
                data_path)

    elif(dataset == "PDBbind"):
        task = 'regression'
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/PDBbind.csv.gz",
                data_path)
        target = []


    print("DATASET: {}".format(dataset))
    return csv_file_path, target, task


def _process_csv(csv_file, target, task):
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    smiles_data, labels = [], {}
    for t in target:
        labels[t] = []

    for i, row in tqdm(enumerate(csv_reader)):
        #for c in row.keys():
        #    print("\"{}\",".format(c))
        #raise
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
                    raise ValueError('task must be either regression or classification')

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

