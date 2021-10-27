# NOTE: Uses code from torchdrug.ai

import pandas as pd
import os
import pathlib
import urllib
from tqdm import tqdm
import csv
import gzip
from rdkit import Chem
import numpy as np
import hashlib

#TODO:
#  1) Clean up code and add proper documentation

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


def _make_csv(fpath, molecules):
    # Open csv from fpath, match SMILES with data, save as new csv
    data_csv = pd.read_csv(fpath)
    smiles = []
    idxs = []
    for idx, m in enumerate(molecules):
        if(m is not None):
            smiles.append(Chem.MolToSmiles(m))
            idxs.append(idx)

    data_csv = data_csv.iloc[idxs]
    data_csv['smiles'] = smiles

    combined_csv_path = fpath[:-8] + ".csv"
    data_csv.to_csv(fpath[:-8] + ".csv")

    return combined_csv_path


def _extract(fpath):
    if(".tar.gz" in fpath):
        import tarfile
        with tarfile.open(fpath, "r") as fin:
            if("gdb8" in fpath):
                fin.extractall(fpath[:-7] + "/")
                fpath = fpath[:-7] + "/qm8.sdf.csv"
                molecules = Chem.SDMolSupplier(fpath[:-4], True, True, False)
            elif("gdb9" in fpath):
                fin.extractall(fpath[:-7] + "/")
                fpath = fpath[:-7] + "/gdb9.sdf.csv"
                molecules = Chem.SDMolSupplier(fpath[:-4], True, True, False)

            # Need to put together smiles and data
            fpath = _make_csv(fpath, molecules)

    elif(".zip" in fpath):
        import zipfile
        with zipfile.ZipFile(fpath) as fin:
            if("FreeSolv" in fpath):
                fin.extractall(fpath[:-4])
                fpath = fpath[:-4] + "/SAMPL.csv"

    return fpath

def download_url(url, root, md5, filename=None):
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

    # Correct file path for extracted sets
    if("FreeSolv.zip" in fpath):
        if(os.path.isfile(fpath[:-4] + "/SAMPL.csv")):
            fpath = fpath[:-4] + "/SAMPL.csv"
    elif("gdb8.tar.gz" in fpath):
        if(os.path.isfile(fpath[:-7] + "/qm8.sdf.csv")):
            fpath = fpath[:-7] + "/qm8.csv"
    elif("gdb9.tar.gz" in fpath):
        if(os.path.isfile(fpath[:-7] + "/gdb9.sdf.csv")):
            fpath = fpath[:-7] + "/gdb9.csv"

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
    
    message = "Download failed or dataset corrupt. Please delete and try again."
    assert check_integrity(fpath, md5), message
    return _extract(fpath)


def _load_data(dataset=None, data_path='./data_download/'):
    if(dataset == 'BACE'):
        task = 'classification'
        target = ['Class']
        md5 = "ba7f8fa3fdf463a811fa7edea8c982c2"
        csv_file_path = download_url(
                    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
                    data_path, md5)

    elif(dataset == 'ClinTox'):
        task = 'classification'
        target = ['CT_TOX', 'FDA_APPROVED']
        md5 = "db4f2df08be8ae92814e9d6a2d015284"
        csv_file_path = download_url(
                    "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
                    data_path, md5)

    elif(dataset == 'BBBP'):
        task = 'classification'
        target = ['p_np']
        md5 = "66286cb9e6b148bd75d80c870df580fb"
        csv_file_path = download_url(
                   "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
                   data_path, md5)

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
        md5 = "77c0ef421f7cc8ce963c5836c8761fd2"
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
                data_path, md5)

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
        md5 = "2882d69e70bba0fec14995f26787cc25"
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
                data_path, md5)

    elif(dataset == 'HIV'):
        task = 'classification'
        target = ["HIV_active"]
        md5 = "9ad10c88f82f1dac7eb5c52b668c30a7"
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
                data_path, md5)

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
        md5 = "9c40bd41310991efd40f4d4868fa3ddf"
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
                data_path, md5)

    elif(dataset == 'FreeSolv'): # SMILES is second column
        task = 'regression'
        target = ["expt"]
        md5 = "8d681babd239b15e2f8b2d29f025577a"
        csv_file_path = download_url(
                "https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/FreeSolv.zip",
                data_path, md5)

    elif(dataset == 'ESOL'): # SMILES is last column
        task = 'regression'
        target = ["measured log solubility in mols per litre"]
        md5 = "0c90a51668d446b9e3ab77e67662bd1c"
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
                data_path, md5)

    elif(dataset == 'Lipophilicity'):
        task = 'regression'
        target = ["exp"]
        md5 = "85a0e1cb8b38b0dfc3f96ff47a57f0ab"
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
                data_path, md5)

    elif(dataset == 'QM7'):
        task = 'regression'
        target = ["u0_atom"]
        md5 = None # No md5 from torchdrug
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv",
                data_path, md5)

    #elif(dataset == 'QM7b'):
    #    task = 'regression'
    #    target = ["u0_atom"]
    #    csv_file_path = download_url(
    #            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat",
    #            data_path)

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
        md5 = "b7e2a2c823c75b35c596f3013319c86e"
        csv_file_path = download_url(
                "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb8.tar.gz",
                data_path, md5)

    elif(dataset == 'QM9'):
        task = 'regression'
        target = [
                "mu",
                "alpha",
                "homo",
                "lumo",
                "gap",
                "r2",
                "zpve",
                "u0",
                "u298", 
                "h298", 
                "g298", 
                "cv",
        ]
        md5 = "560f62d8e6c992ca0cf8ed8d013f9131"
        csv_file_path = download_url(
                "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz",
                data_path, md5)

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
        md5 = None # No md5 available from torchdrug
        csv_file_path = download_url(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
                data_path, md5)

    #elif(dataset == "PDBbind"):
    #    task = 'regression'
    #    csv_file_path = download_url(
    #"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2019_other_PL.tar.gz",
    #            data_path)
    #    target = ['-logKd/Ki']
    #elif(dataset == "PDBbind - refined"):
    #    task = 'regression'
    #    csv_file_path = download_url(
    #"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2019_refined.tar.gz",
    #            data_path)
    #    target = ['-logKd/Ki']
    #elif(dataset == "PDBbind - core"):
    #    task = 'regression'
    #    csv_file_path = download_url(
    #"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2013_core_set.tar.gz",
    #            data_path)
    #    target = ['-logKd/Ki']

    elif(dataset == 'ToxCast'):
        task = 'classification'
        target = [
                'ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive',
                'APR_HepG2_CellCycleArrest_24h_dn', 'APR_HepG2_CellCycleArrest_24h_up',
                'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn',
                'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_24h_dn',
                'APR_HepG2_MicrotubuleCSK_24h_up', 'APR_HepG2_MicrotubuleCSK_72h_dn',
                'APR_HepG2_MicrotubuleCSK_72h_up', 'APR_HepG2_MitoMass_24h_dn',
                'APR_HepG2_MitoMass_24h_up', 'APR_HepG2_MitoMass_72h_dn',
                'APR_HepG2_MitoMass_72h_up', 'APR_HepG2_MitoMembPot_1h_dn',
                'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
                'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up',
                'APR_HepG2_NuclearSize_24h_dn', 'APR_HepG2_NuclearSize_72h_dn',
                'APR_HepG2_NuclearSize_72h_up', 'APR_HepG2_OxidativeStress_24h_up',
                'APR_HepG2_OxidativeStress_72h_up', 'APR_HepG2_StressKinase_1h_up',
                'APR_HepG2_StressKinase_24h_up', 'APR_HepG2_StressKinase_72h_up',
                'APR_HepG2_p53Act_24h_up', 'APR_HepG2_p53Act_72h_up',
                'APR_Hepat_Apoptosis_24hr_up', 'APR_Hepat_Apoptosis_48hr_up',
                'APR_Hepat_CellLoss_24hr_dn', 'APR_Hepat_CellLoss_48hr_dn',
                'APR_Hepat_DNADamage_24hr_up', 'APR_Hepat_DNADamage_48hr_up',
                'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up',
                'APR_Hepat_MitoFxnI_1hr_dn', 'APR_Hepat_MitoFxnI_24hr_dn',
                'APR_Hepat_MitoFxnI_48hr_dn', 'APR_Hepat_NuclearSize_24hr_dn',
                'APR_Hepat_NuclearSize_48hr_dn', 'APR_Hepat_Steatosis_24hr_up',
                'APR_Hepat_Steatosis_48hr_up', 'ATG_AP_1_CIS_dn', 'ATG_AP_1_CIS_up',
                'ATG_AP_2_CIS_dn', 'ATG_AP_2_CIS_up', 'ATG_AR_TRANS_dn', 'ATG_AR_TRANS_up',
                'ATG_Ahr_CIS_dn', 'ATG_Ahr_CIS_up', 'ATG_BRE_CIS_dn', 'ATG_BRE_CIS_up',
                'ATG_CAR_TRANS_dn', 'ATG_CAR_TRANS_up', 'ATG_CMV_CIS_dn', 'ATG_CMV_CIS_up',
                'ATG_CRE_CIS_dn', 'ATG_CRE_CIS_up', 'ATG_C_EBP_CIS_dn', 'ATG_C_EBP_CIS_up',
                'ATG_DR4_LXR_CIS_dn', 'ATG_DR4_LXR_CIS_up', 'ATG_DR5_CIS_dn',
                'ATG_DR5_CIS_up', 'ATG_E2F_CIS_dn', 'ATG_E2F_CIS_up', 'ATG_EGR_CIS_up',
                'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up', 'ATG_ERRa_TRANS_dn',
                'ATG_ERRg_TRANS_dn', 'ATG_ERRg_TRANS_up', 'ATG_ERa_TRANS_up',
                'ATG_E_Box_CIS_dn', 'ATG_E_Box_CIS_up', 'ATG_Ets_CIS_dn', 'ATG_Ets_CIS_up',
                'ATG_FXR_TRANS_up', 'ATG_FoxA2_CIS_dn', 'ATG_FoxA2_CIS_up',
                'ATG_FoxO_CIS_dn', 'ATG_FoxO_CIS_up', 'ATG_GAL4_TRANS_dn',
                'ATG_GATA_CIS_dn', 'ATG_GATA_CIS_up', 'ATG_GLI_CIS_dn', 'ATG_GLI_CIS_up',
                'ATG_GRE_CIS_dn', 'ATG_GRE_CIS_up', 'ATG_GR_TRANS_dn', 'ATG_GR_TRANS_up',
                'ATG_HIF1a_CIS_dn', 'ATG_HIF1a_CIS_up', 'ATG_HNF4a_TRANS_dn',
                'ATG_HNF4a_TRANS_up', 'ATG_HNF6_CIS_dn', 'ATG_HNF6_CIS_up',
                'ATG_HSE_CIS_dn', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn', 'ATG_IR1_CIS_up',
                'ATG_ISRE_CIS_dn', 'ATG_ISRE_CIS_up', 'ATG_LXRa_TRANS_dn',
                'ATG_LXRa_TRANS_up', 'ATG_LXRb_TRANS_dn', 'ATG_LXRb_TRANS_up',
                'ATG_MRE_CIS_up', 'ATG_M_06_TRANS_up', 'ATG_M_19_CIS_dn',
                'ATG_M_19_TRANS_dn', 'ATG_M_19_TRANS_up', 'ATG_M_32_CIS_dn',
                'ATG_M_32_CIS_up', 'ATG_M_32_TRANS_dn', 'ATG_M_32_TRANS_up',
                'ATG_M_61_TRANS_up', 'ATG_Myb_CIS_dn', 'ATG_Myb_CIS_up', 'ATG_Myc_CIS_dn',
                'ATG_Myc_CIS_up', 'ATG_NFI_CIS_dn', 'ATG_NFI_CIS_up', 'ATG_NF_kB_CIS_dn',
                'ATG_NF_kB_CIS_up', 'ATG_NRF1_CIS_dn', 'ATG_NRF1_CIS_up',
                'ATG_NRF2_ARE_CIS_dn', 'ATG_NRF2_ARE_CIS_up', 'ATG_NURR1_TRANS_dn',
                'ATG_NURR1_TRANS_up', 'ATG_Oct_MLP_CIS_dn', 'ATG_Oct_MLP_CIS_up',
                'ATG_PBREM_CIS_dn', 'ATG_PBREM_CIS_up', 'ATG_PPARa_TRANS_dn',
                'ATG_PPARa_TRANS_up', 'ATG_PPARd_TRANS_up', 'ATG_PPARg_TRANS_up',
                'ATG_PPRE_CIS_dn', 'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up',
                'ATG_PXR_TRANS_dn', 'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up',
                'ATG_RARa_TRANS_dn', 'ATG_RARa_TRANS_up', 'ATG_RARb_TRANS_dn',
                'ATG_RARb_TRANS_up', 'ATG_RARg_TRANS_dn', 'ATG_RARg_TRANS_up',
                'ATG_RORE_CIS_dn', 'ATG_RORE_CIS_up', 'ATG_RORb_TRANS_dn',
                'ATG_RORg_TRANS_dn', 'ATG_RORg_TRANS_up', 'ATG_RXRa_TRANS_dn',
                'ATG_RXRa_TRANS_up', 'ATG_RXRb_TRANS_dn', 'ATG_RXRb_TRANS_up',
                'ATG_SREBP_CIS_dn', 'ATG_SREBP_CIS_up', 'ATG_STAT3_CIS_dn',
                'ATG_STAT3_CIS_up', 'ATG_Sox_CIS_dn', 'ATG_Sox_CIS_up', 'ATG_Sp1_CIS_dn',
                'ATG_Sp1_CIS_up', 'ATG_TAL_CIS_dn', 'ATG_TAL_CIS_up', 'ATG_TA_CIS_dn',
                'ATG_TA_CIS_up', 'ATG_TCF_b_cat_CIS_dn', 'ATG_TCF_b_cat_CIS_up',
                'ATG_TGFb_CIS_dn', 'ATG_TGFb_CIS_up', 'ATG_THRa1_TRANS_dn',
                'ATG_THRa1_TRANS_up', 'ATG_VDRE_CIS_dn', 'ATG_VDRE_CIS_up',
                'ATG_VDR_TRANS_dn', 'ATG_VDR_TRANS_up', 'ATG_XTT_Cytotoxicity_up',
                'ATG_Xbp1_CIS_dn', 'ATG_Xbp1_CIS_up', 'ATG_p53_CIS_dn', 'ATG_p53_CIS_up',
                'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down', 'BSK_3C_ICAM1_down',
                'BSK_3C_IL8_down', 'BSK_3C_MCP1_down', 'BSK_3C_MIG_down',
                'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down',
                'BSK_3C_Thrombomodulin_down', 'BSK_3C_Thrombomodulin_up',
                'BSK_3C_TissueFactor_down', 'BSK_3C_TissueFactor_up', 'BSK_3C_VCAM1_down',
                'BSK_3C_Vis_down', 'BSK_3C_uPAR_down', 'BSK_4H_Eotaxin3_down',
                'BSK_4H_MCP1_down', 'BSK_4H_Pselectin_down', 'BSK_4H_Pselectin_up',
                'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down', 'BSK_4H_VEGFRII_down',
                'BSK_4H_uPAR_down', 'BSK_4H_uPAR_up', 'BSK_BE3C_HLADR_down',
                'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down',
                'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down',
                'BSK_BE3C_SRB_down', 'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down',
                'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPAR_up', 'BSK_BE3C_uPA_down',
                'BSK_CASM3C_HLADR_down', 'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL6_up',
                'BSK_CASM3C_IL8_down', 'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_LDLR_up',
                'BSK_CASM3C_MCP1_down', 'BSK_CASM3C_MCP1_up', 'BSK_CASM3C_MCSF_down',
                'BSK_CASM3C_MCSF_up', 'BSK_CASM3C_MIG_down',
                'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_Proliferation_up',
                'BSK_CASM3C_SAA_down', 'BSK_CASM3C_SAA_up', 'BSK_CASM3C_SRB_down',
                'BSK_CASM3C_Thrombomodulin_down', 'BSK_CASM3C_Thrombomodulin_up',
                'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down',
                'BSK_CASM3C_VCAM1_up', 'BSK_CASM3C_uPAR_down', 'BSK_CASM3C_uPAR_up',
                'BSK_KF3CT_ICAM1_down', 'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down',
                'BSK_KF3CT_IP10_up', 'BSK_KF3CT_MCP1_down', 'BSK_KF3CT_MCP1_up',
                'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down', 'BSK_KF3CT_TGFb1_down',
                'BSK_KF3CT_TIMP2_down', 'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down',
                'BSK_LPS_Eselectin_down', 'BSK_LPS_Eselectin_up', 'BSK_LPS_IL1a_down',
                'BSK_LPS_IL1a_up', 'BSK_LPS_IL8_down', 'BSK_LPS_IL8_up',
                'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down',
                'BSK_LPS_PGE2_up', 'BSK_LPS_SRB_down', 'BSK_LPS_TNFa_down',
                'BSK_LPS_TNFa_up', 'BSK_LPS_TissueFactor_down', 'BSK_LPS_TissueFactor_up',
                'BSK_LPS_VCAM1_down', 'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down',
                'BSK_SAg_CD69_down', 'BSK_SAg_Eselectin_down', 'BSK_SAg_Eselectin_up',
                'BSK_SAg_IL8_down', 'BSK_SAg_IL8_up', 'BSK_SAg_MCP1_down',
                'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down',
                'BSK_SAg_PBMCCytotoxicity_up', 'BSK_SAg_Proliferation_down',
                'BSK_SAg_SRB_down', 'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_EGFR_down',
                'BSK_hDFCGF_EGFR_up', 'BSK_hDFCGF_IL8_down', 'BSK_hDFCGF_IP10_down',
                'BSK_hDFCGF_MCSF_down', 'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down',
                'BSK_hDFCGF_MMP1_up', 'BSK_hDFCGF_PAI1_down',
                'BSK_hDFCGF_Proliferation_down', 'BSK_hDFCGF_SRB_down',
                'BSK_hDFCGF_TIMP1_down', 'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn',
                'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn', 'CEETOX_H295R_DOC_dn',
                'CEETOX_H295R_DOC_up', 'CEETOX_H295R_ESTRADIOL_dn',
                'CEETOX_H295R_ESTRADIOL_up', 'CEETOX_H295R_ESTRONE_dn',
                'CEETOX_H295R_ESTRONE_up', 'CEETOX_H295R_OHPREG_up',
                'CEETOX_H295R_OHPROG_dn', 'CEETOX_H295R_OHPROG_up', 'CEETOX_H295R_PROG_up',
                'CEETOX_H295R_TESTO_dn', 'CLD_ABCB1_48hr', 'CLD_ABCG2_48hr',
                'CLD_CYP1A1_24hr', 'CLD_CYP1A1_48hr', 'CLD_CYP1A1_6hr', 'CLD_CYP1A2_24hr',
                'CLD_CYP1A2_48hr', 'CLD_CYP1A2_6hr', 'CLD_CYP2B6_24hr', 'CLD_CYP2B6_48hr',
                'CLD_CYP2B6_6hr', 'CLD_CYP3A4_24hr', 'CLD_CYP3A4_48hr', 'CLD_CYP3A4_6hr',
                'CLD_GSTA2_48hr', 'CLD_SULT2A_24hr', 'CLD_SULT2A_48hr', 'CLD_UGT1A1_24hr',
                'CLD_UGT1A1_48hr', 'NCCT_HEK293T_CellTiterGLO', 'NCCT_QuantiLum_inhib_2_dn',
                'NCCT_QuantiLum_inhib_dn', 'NCCT_TPO_AUR_dn', 'NCCT_TPO_GUA_dn',
                'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_ADME_hCYP19A1', 'NVS_ADME_hCYP1A1',
                'NVS_ADME_hCYP1A2', 'NVS_ADME_hCYP2A6', 'NVS_ADME_hCYP2B6',
                'NVS_ADME_hCYP2C19', 'NVS_ADME_hCYP2C9', 'NVS_ADME_hCYP2D6',
                'NVS_ADME_hCYP3A4', 'NVS_ADME_hCYP4F12', 'NVS_ADME_rCYP2C12',
                'NVS_ENZ_hAChE', 'NVS_ENZ_hAMPKa1', 'NVS_ENZ_hAurA', 'NVS_ENZ_hBACE',
                'NVS_ENZ_hCASP5', 'NVS_ENZ_hCK1D', 'NVS_ENZ_hDUSP3', 'NVS_ENZ_hES',
                'NVS_ENZ_hElastase', 'NVS_ENZ_hFGFR1', 'NVS_ENZ_hGSK3b', 'NVS_ENZ_hMMP1',
                'NVS_ENZ_hMMP13', 'NVS_ENZ_hMMP2', 'NVS_ENZ_hMMP3', 'NVS_ENZ_hMMP7',
                'NVS_ENZ_hMMP9', 'NVS_ENZ_hPDE10', 'NVS_ENZ_hPDE4A1', 'NVS_ENZ_hPDE5',
                'NVS_ENZ_hPI3Ka', 'NVS_ENZ_hPTEN', 'NVS_ENZ_hPTPN11', 'NVS_ENZ_hPTPN12',
                'NVS_ENZ_hPTPN13', 'NVS_ENZ_hPTPN9', 'NVS_ENZ_hPTPRC', 'NVS_ENZ_hSIRT1',
                'NVS_ENZ_hSIRT2', 'NVS_ENZ_hTrkA', 'NVS_ENZ_hVEGFR2', 'NVS_ENZ_oCOX1',
                'NVS_ENZ_oCOX2', 'NVS_ENZ_rAChE', 'NVS_ENZ_rCNOS', 'NVS_ENZ_rMAOAC',
                'NVS_ENZ_rMAOAP', 'NVS_ENZ_rMAOBC', 'NVS_ENZ_rMAOBP', 'NVS_ENZ_rabI2C',
                'NVS_GPCR_bAdoR_NonSelective', 'NVS_GPCR_bDR_NonSelective',
                'NVS_GPCR_g5HT4', 'NVS_GPCR_gH2', 'NVS_GPCR_gLTB4', 'NVS_GPCR_gLTD4',
                'NVS_GPCR_gMPeripheral_NonSelective', 'NVS_GPCR_gOpiateK',
                'NVS_GPCR_h5HT2A', 'NVS_GPCR_h5HT5A', 'NVS_GPCR_h5HT6', 'NVS_GPCR_h5HT7',
                'NVS_GPCR_hAT1', 'NVS_GPCR_hAdoRA1', 'NVS_GPCR_hAdoRA2a',
                'NVS_GPCR_hAdra2A', 'NVS_GPCR_hAdra2C', 'NVS_GPCR_hAdrb1',
                'NVS_GPCR_hAdrb2', 'NVS_GPCR_hAdrb3', 'NVS_GPCR_hDRD1', 'NVS_GPCR_hDRD2s',
                'NVS_GPCR_hDRD4.4', 'NVS_GPCR_hH1', 'NVS_GPCR_hLTB4_BLT1', 'NVS_GPCR_hM1',
                'NVS_GPCR_hM2', 'NVS_GPCR_hM3', 'NVS_GPCR_hM4', 'NVS_GPCR_hNK2',
                'NVS_GPCR_hOpiate_D1', 'NVS_GPCR_hOpiate_mu', 'NVS_GPCR_hTXA2',
                'NVS_GPCR_p5HT2C', 'NVS_GPCR_r5HT1_NonSelective',
                'NVS_GPCR_r5HT_NonSelective', 'NVS_GPCR_rAdra1B',
                'NVS_GPCR_rAdra1_NonSelective', 'NVS_GPCR_rAdra2_NonSelective',
                'NVS_GPCR_rAdrb_NonSelective', 'NVS_GPCR_rNK1', 'NVS_GPCR_rNK3',
                'NVS_GPCR_rOpiate_NonSelective', 'NVS_GPCR_rOpiate_NonSelectiveNa',
                'NVS_GPCR_rSST', 'NVS_GPCR_rTRH', 'NVS_GPCR_rV1', 'NVS_GPCR_rabPAF',
                'NVS_GPCR_rmAdra2B', 'NVS_IC_hKhERGCh', 'NVS_IC_rCaBTZCHL',
                'NVS_IC_rCaDHPRCh_L', 'NVS_IC_rNaCh_site2', 'NVS_LGIC_bGABARa1',
                'NVS_LGIC_h5HT3', 'NVS_LGIC_hNNR_NBungSens', 'NVS_LGIC_rGABAR_NonSelective',
                'NVS_LGIC_rNNR_BungSens', 'NVS_MP_hPBR', 'NVS_MP_rPBR', 'NVS_NR_bER',
                'NVS_NR_bPR', 'NVS_NR_cAR', 'NVS_NR_hAR', 'NVS_NR_hCAR_Antagonist',
                'NVS_NR_hER', 'NVS_NR_hFXR_Agonist', 'NVS_NR_hFXR_Antagonist', 'NVS_NR_hGR',
                'NVS_NR_hPPARa', 'NVS_NR_hPPARg', 'NVS_NR_hPR', 'NVS_NR_hPXR',
                'NVS_NR_hRAR_Antagonist', 'NVS_NR_hRARa_Agonist', 'NVS_NR_hTRa_Antagonist',
                'NVS_NR_mERa', 'NVS_NR_rAR', 'NVS_NR_rMR', 'NVS_OR_gSIGMA_NonSelective',
                'NVS_TR_gDAT', 'NVS_TR_hAdoT', 'NVS_TR_hDAT', 'NVS_TR_hNET', 'NVS_TR_hSERT',
                'NVS_TR_rNET', 'NVS_TR_rSERT', 'NVS_TR_rVMAT2', 'OT_AR_ARELUC_AG_1440',
                'OT_AR_ARSRC1_0480', 'OT_AR_ARSRC1_0960', 'OT_ER_ERaERa_0480',
                'OT_ER_ERaERa_1440', 'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440',
                'OT_ER_ERbERb_0480', 'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120',
                'OT_ERa_EREGFP_0480', 'OT_FXR_FXRSRC1_0480', 'OT_FXR_FXRSRC1_1440',
                'OT_NURR1_NURR1RXRa_0480', 'OT_NURR1_NURR1RXRa_1440',
                'TOX21_ARE_BLA_Agonist_ch1', 'TOX21_ARE_BLA_Agonist_ch2',
                'TOX21_ARE_BLA_agonist_ratio', 'TOX21_ARE_BLA_agonist_viability',
                'TOX21_AR_BLA_Agonist_ch1', 'TOX21_AR_BLA_Agonist_ch2',
                'TOX21_AR_BLA_Agonist_ratio', 'TOX21_AR_BLA_Antagonist_ch1',
                'TOX21_AR_BLA_Antagonist_ch2', 'TOX21_AR_BLA_Antagonist_ratio',
                'TOX21_AR_BLA_Antagonist_viability', 'TOX21_AR_LUC_MDAKB2_Agonist',
                'TOX21_AR_LUC_MDAKB2_Antagonist', 'TOX21_AR_LUC_MDAKB2_Antagonist2',
                'TOX21_AhR_LUC_Agonist', 'TOX21_Aromatase_Inhibition',
                'TOX21_AutoFluor_HEK293_Cell_blue', 'TOX21_AutoFluor_HEK293_Media_blue',
                'TOX21_AutoFluor_HEPG2_Cell_blue', 'TOX21_AutoFluor_HEPG2_Cell_green',
                'TOX21_AutoFluor_HEPG2_Media_blue', 'TOX21_AutoFluor_HEPG2_Media_green',
                'TOX21_ELG1_LUC_Agonist', 'TOX21_ERa_BLA_Agonist_ch1',
                'TOX21_ERa_BLA_Agonist_ch2', 'TOX21_ERa_BLA_Agonist_ratio',
                'TOX21_ERa_BLA_Antagonist_ch1', 'TOX21_ERa_BLA_Antagonist_ch2',
                'TOX21_ERa_BLA_Antagonist_ratio', 'TOX21_ERa_BLA_Antagonist_viability',
                'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_ERa_LUC_BG1_Antagonist',
                'TOX21_ESRE_BLA_ch1', 'TOX21_ESRE_BLA_ch2', 'TOX21_ESRE_BLA_ratio',
                'TOX21_ESRE_BLA_viability', 'TOX21_FXR_BLA_Antagonist_ch1',
                'TOX21_FXR_BLA_Antagonist_ch2', 'TOX21_FXR_BLA_agonist_ch2',
                'TOX21_FXR_BLA_agonist_ratio', 'TOX21_FXR_BLA_antagonist_ratio',
                'TOX21_FXR_BLA_antagonist_viability', 'TOX21_GR_BLA_Agonist_ch1',
                'TOX21_GR_BLA_Agonist_ch2', 'TOX21_GR_BLA_Agonist_ratio',
                'TOX21_GR_BLA_Antagonist_ch2', 'TOX21_GR_BLA_Antagonist_ratio',
                'TOX21_GR_BLA_Antagonist_viability', 'TOX21_HSE_BLA_agonist_ch1',
                'TOX21_HSE_BLA_agonist_ch2', 'TOX21_HSE_BLA_agonist_ratio',
                'TOX21_HSE_BLA_agonist_viability', 'TOX21_MMP_ratio_down',
                'TOX21_MMP_ratio_up', 'TOX21_MMP_viability', 'TOX21_NFkB_BLA_agonist_ch1',
                'TOX21_NFkB_BLA_agonist_ch2', 'TOX21_NFkB_BLA_agonist_ratio',
                'TOX21_NFkB_BLA_agonist_viability', 'TOX21_PPARd_BLA_Agonist_viability',
                'TOX21_PPARd_BLA_Antagonist_ch1', 'TOX21_PPARd_BLA_agonist_ch1',
                'TOX21_PPARd_BLA_agonist_ch2', 'TOX21_PPARd_BLA_agonist_ratio',
                'TOX21_PPARd_BLA_antagonist_ratio', 'TOX21_PPARd_BLA_antagonist_viability',
                'TOX21_PPARg_BLA_Agonist_ch1', 'TOX21_PPARg_BLA_Agonist_ch2',
                'TOX21_PPARg_BLA_Agonist_ratio', 'TOX21_PPARg_BLA_Antagonist_ch1',
                'TOX21_PPARg_BLA_antagonist_ratio', 'TOX21_PPARg_BLA_antagonist_viability',
                'TOX21_TR_LUC_GH3_Agonist', 'TOX21_TR_LUC_GH3_Antagonist',
                'TOX21_VDR_BLA_Agonist_viability', 'TOX21_VDR_BLA_Antagonist_ch1',
                'TOX21_VDR_BLA_agonist_ch2', 'TOX21_VDR_BLA_agonist_ratio',
                'TOX21_VDR_BLA_antagonist_ratio', 'TOX21_VDR_BLA_antagonist_viability',
                'TOX21_p53_BLA_p1_ch1', 'TOX21_p53_BLA_p1_ch2', 'TOX21_p53_BLA_p1_ratio',
                'TOX21_p53_BLA_p1_viability', 'TOX21_p53_BLA_p2_ch1', 'TOX21_p53_BLA_p2_ch2',
                'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability',
                'TOX21_p53_BLA_p3_ch1', 'TOX21_p53_BLA_p3_ch2', 'TOX21_p53_BLA_p3_ratio',
                'TOX21_p53_BLA_p3_viability', 'TOX21_p53_BLA_p4_ch1', 'TOX21_p53_BLA_p4_ch2',
                'TOX21_p53_BLA_p4_ratio', 'TOX21_p53_BLA_p4_viability',
                'TOX21_p53_BLA_p5_ch1', 'TOX21_p53_BLA_p5_ch2', 'TOX21_p53_BLA_p5_ratio',
                'TOX21_p53_BLA_p5_viability', 'Tanguay_ZF_120hpf_AXIS_up',
                'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_BRAI_up',
                'Tanguay_ZF_120hpf_CFIN_up', 'Tanguay_ZF_120hpf_CIRC_up',
                'Tanguay_ZF_120hpf_EYE_up', 'Tanguay_ZF_120hpf_JAW_up',
                'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_OTIC_up',
                'Tanguay_ZF_120hpf_PE_up', 'Tanguay_ZF_120hpf_PFIN_up',
                'Tanguay_ZF_120hpf_PIG_up', 'Tanguay_ZF_120hpf_SNOU_up',
                'Tanguay_ZF_120hpf_SOMI_up', 'Tanguay_ZF_120hpf_SWIM_up',
                'Tanguay_ZF_120hpf_TRUN_up', 'Tanguay_ZF_120hpf_TR_up',
                'Tanguay_ZF_120hpf_YSE_up'
            ]
        md5 = "92911bbf9c1e2ad85231014859388cd6"
        csv_file_path = download_url(
                "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
                data_path, md5)

    else:
        raise RuntimeError("{} is not supported.".format(dataset))
    print("DATASET: {}".format(dataset))
    return csv_file_path, target, task


def _process_csv(csv_file, target, task):
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    smiles_data, labels = [], {}
    for t in target:
        labels[t] = []

    for i, row in tqdm(enumerate(csv_reader)):
        # Skip header
        if i == 0:
            continue

        # Hold on to all rows that contain any targets
        try:
            smiles = row['smiles']
        except KeyError:
            smiles = row['mol']

        # Hold on to good smiles representations
        mol = Chem.MolFromSmiles(smiles) 
        if(mol != None):
            smiles_data.append(smiles)
        else:
            continue

        # Check values for every target
        for idx, t in enumerate(target):
            label = row[t]
            if label != '':
                if task == 'classification':
                    try:
                        labels[t].append(int(label))
                    except ValueError:
                        labels[t].append(int(label[0]))
                elif task == 'regression':
                    labels[t].append(float(label))
                else:
                    raise ValueError('task must be either regression or classification')
            elif mol != None and label == '':
                labels[t].append(-999999999)

    # Recast lables to numpy arrays
    for t in target:
        labels[t] = np.array(labels[t])

    #for key, val in labels.items():
    #    print(key, len(val))

    return smiles_data, labels, task


def read_smiles(dataset, data_path):

    # Create data download directory if it does not exist
    if(data_path is None):
        data_path = './data_download'
    if(not os.path.exists(data_path)):
        os.mkdir(data_path)

    # Download files if not already there
    csv_file_path, target, task = _load_data(dataset, data_path)
    if("csv.gz" in csv_file_path):
        with gzip.open(csv_file_path, 'rt') as csv_file:
            return _process_csv(csv_file, target, task)
    else:
        with open(csv_file_path) as csv_file:
            return _process_csv(csv_file, target, task)

    return smiles_data, np.array(labels)

