import sys
from tqdm import tqdm
#sys.path.append(sys.path[0][:-8])

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, mean_squared_error
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
            for data in val_loader:
                
                # Get data
                x = data.x.float().to(device=device)
                edge_index = data.edge_index.to(device=device)
                edge_attr = data.edge_attr.float().to(device=device)
                
                # Predict
                pred = model(x, edge_index, edge_attr, data.batch.to(device=device))

                for i, target in enumerate(target_list):
                    if(t != target):
                        continue
                    # Hold on to predictions and targets
                    #all_preds[target].append(list(pred[:,2*i:2*(i+1)].detach().cpu().numpy()[:,1]))
                    #all_targets[target].append(list(data.y[:,i].flatten().numpy()))
                    if(train_loader.dataset.task == 'classification'):
                        all_preds[target].append(list(pred[:,2*i:2*(i+1)].detach().cpu().numpy()[:,1]))
                        all_targets[target].append(list(data.y[:,i].flatten().numpy()))
                    elif(train_loader.dataset.task == 'regression'):
                        all_preds[target].append(pred.detach().cpu().numpy()[:,0])
                        all_targets[target].append(data.y.numpy().flatten())

        for target in target_list:
            if(t != target):
                continue
            #scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
            if(val_loader.dataset.task == 'classification'):
                scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
                print("{0} TEST ROC: {1:.5f}".format(target, scores[target]))
            elif(val_loader.dataset.task == 'regression'):
                scores[target] = mean_squared_error(all_targets[target], all_preds[target],
                                                    squared=False)
                print("{0} TEST RMSE: {1:.5f}".format(target, scores[target]))
            model.train()
            print("SCORES: {}".format(scores))
    return scores

def print_scores(all_scores):
    print("\n\nALL SCORES:")
    for key, val in all_scores.items():
        print(key)
        val_str = ''.join([str(v) + "\t" for v in val])
        print(val_str)
        print()


if __name__ == '__main__':

    task_name = 'HIV'
    splitting = 'scaffold'
    aug_time = 0

    # Select GPU
    if(len(sys.argv) > 1):
        device = torch.device(sys.argv[1])
    else:
        device = torch.device('cpu')

    # Set run parameters
    batch_size = 200
    epochs = 100
    
    print("Instantiating model...")
    radius = 2
    T = 2
    fingerprint_dim = 200
    p_dropout = 0.3
    weight_decay = 5. # also known as l2_regularization_lambda
    learning_rate = 2.5
    
    print("Getting Dataset...")
    # Not transforming yet
    transform = Compose([
        RandomAtomMask(0.2),
        RandomBondDelete(0.2)
    ])
    dataset = MoleculeDatasetWrapper(task_name, transform=transform, aug_time=aug_time,
                                     batch_size=batch_size, split=splitting)

    target_list = dataset.labels.keys()
    all_scores = {target: [] for target in target_list}
    for seed in [10, 20, 30]:

        # Model saving
        save_string_pre = './saved_models/' + task_name + "/"
        save_string_pre += 'no_aug_' if dataset.aug_time == 0 else ''
        save_string_post = '_afp'
        save_string_post += '_' + str(seed)
        save_string = lambda x: save_string_pre + x + save_string_post

    
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

        print("RESULTS FOR: {}".format(task_name))
        results = evaluate(model, test_loader, device, target_list, save_string)
        print("RESULTS AFTER EVALUATE:\n{}".format(results))
        for t in target_list:
            all_scores[t].append(results[t])

    print_scores(all_scores)

