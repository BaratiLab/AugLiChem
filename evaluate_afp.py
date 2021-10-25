import sys
from tqdm import tqdm
#sys.path.append(sys.path[0][:-8])

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder

from rdkit import Chem

# suppress warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from auglichem.molecule.data import MoleculeDatasetWrapper
from auglichem.molecule import RandomAtomMask, RandomBondDelete, Compose
from auglichem.molecule.models import AttentiveFP

from auglichem.molecule.models import AttentiveFP as AFP
#from torch_geometric.nn.models import AttentiveFP as AFP
from matplotlib import pyplot as plt



def get_model(radius, T, p_dropout, fingerprint_dim, output_units_num, seed):
    model = AFP(num_layers=radius,
                num_timesteps=T,
                dropout=p_dropout,
                hidden_channels=fingerprint_dim,
                out_channels=output_units_num,
                edge_dim=2,
                in_channels=2,
                seed=seed
    )

    return model


def evaluate(model, test_loader, device, target_list, save_string):
    all_targets = {target: [] for target in target_list}
    scores = {target: None for target in target_list}
    for t in target_list:
        print("\nEvaluating using model with highest validation from {}".format(save_string(t)))
        model.load_state_dict(torch.load(save_string(t)))
        with torch.no_grad():
            model.eval()
            all_preds = {target: [] for target in target_list}
            all_targets = {target: [] for target in target_list}
            for data in test_loader:
                
                # Get data
                x = data.x.float().to(device=device)
                edge_index = data.edge_index.to(device=device)
                edge_attr = data.edge_attr.float().to(device=device)
                
                # Predict
                pred = model(x, edge_index, edge_attr, data.batch.to(device=device))

                for i, target in enumerate(target_list):
                    if(t != target):
                        continue

                    # Get indices where there's no data and exclude those from pred and data.y
                    #good_idx = np.where((data.y[:,i]==0) | (data.y[:,i]==1))
                    good_idx = np.where(data.y[:,i]!=-999999999)

                    # Hold on to predictions and targets
                    if(test_loader.dataset.task == 'classification'):
                        current_preds = pred[:,2*i:2*(i+1)][good_idx]
                        current_targets = data.y[:,i][good_idx].to(device=device)
                        if(val_loader.dataset.dataset == 'MUV'):
                            all_preds[target].append(list(current_preds.detach().cpu().numpy()))
                        else:
                            all_preds[target].append(list(current_preds.detach().cpu().numpy()[:,1]))
                        all_targets[target].append(list(current_targets.cpu().flatten().numpy()))

                    elif(test_loader.dataset.task == 'regression'):
                        current_preds = pred[good_idx]
                        current_targets = data.y[:,i][good_idx].to(device=device)
                        all_preds[target].append(current_preds.detach().cpu().numpy()[:,0])
                        all_targets[target].append(current_targets.cpu().numpy().flatten())

        for target in target_list:
            if(t != target):
                continue
            if(test_loader.dataset.task == 'classification'):
                if(test_loader.dataset.dataset == 'MUV'):
                    scores[target] = accuracy_score(all_targets[target][0],
                                                    np.argmax(all_preds[target][0], axis=1))
                    print("{0} VALIDATION ACCURACY: {1:.5f}".format(target, scores[target]))
                else:
                    scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
                    print("{0} VALIDATION ROC: {1:.5f}".format(target, scores[target]))
            elif(test_loader.dataset.task == 'regression'):
                scores[target] = mean_squared_error(all_targets[target], all_preds[target],
                                                squared=False)
                print("{0} TEST RMSE: {1:.5f}".format(target, scores[target]))
        model.train()

    return scores

def print_scores(all_scores):
    print("\n\nALL SCORES:")
    for key, val in all_scores.items():
        print(key)
        val_str = ''.join([str(v) + "\t" for v in val])
        print(val_str)
        print()


if __name__ == '__main__':

    task_name = 'ClinTox'
    aug_time = 0

    # Select GPU
    if(len(sys.argv) > 1):
        device = torch.device(sys.argv[1])
    else:
        device = torch.device('cpu')

    # Set run parameters
    batch_size = 100
    epochs = 100
    
    print("Instantiating model...")
    #radius = 3
    #T = 2
    #fingerprint_dim = 150
    #p_dropout = 0.1
    
    print("Getting Dataset...")
    # Not transforming yet
    transform = Compose([
        RandomAtomMask(0.1),
        RandomBondDelete(0.1)
    ])
    dataset = MoleculeDatasetWrapper(task_name, transform=transform, aug_time=aug_time,
                                     batch_size=batch_size, split='scaffold')

    target_list = dataset.labels.keys()
    all_scores = {target: [] for target in target_list}
    for seed in [10, 20, 30]:

        # Model saving
        save_string_pre = './saved_models/multi_sample_' + task_name + "/"
        save_string_pre += 'no_aug_' if dataset.aug_time == 0 else ''
        save_string_post = '_afp'
        save_string_post += '_' + str(seed)
        save_string = lambda x: save_string_pre + x + save_string_post

        # Get parameters
        params = np.load(save_string("parameters")+".npy", allow_pickle=True).item()
        for key, val in params.items():
            print("{}:\t{}".format(key, val))
        radius = params['radius']
        T = params['T']
        fingerprint_dim = params['fingerprint_dim']
        splitting = params['splitting']
        p_dropout = params['dropout']
    
        dataset = MoleculeDatasetWrapper(task_name, transform=transform, aug_time=aug_time,
                                 batch_size=batch_size, seed=seed, split=splitting)

        # Would need to change things here for multiclass
        train_loader, val_loader, test_loader = \
                dataset.get_data_loaders(list(dataset.labels.keys()))

        # Set up training
        if(dataset.task == 'classification'):
            output_units_num = 2 * len(dataset.labels.keys())
        elif(dataset.task == 'regression'):
            output_units_num = 1

        model = get_model(radius, T, p_dropout, fingerprint_dim, output_units_num, seed).to(
                          device=device)

        #print("RESULTS FOR: {}".format(task_name))
        results = evaluate(model, test_loader, device, target_list, save_string)
        #print("RESULTS AFTER EVALUATE:\n{}".format(results))
        for t in target_list:
            all_scores[t].append(results[t])

    print_scores(all_scores)

