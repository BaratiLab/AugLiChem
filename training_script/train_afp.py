import sys
from tqdm import tqdm
#sys.path.append(sys.path[0][:-8])

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score
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
    print("\nEvaluating using model at end of training...")
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
                # Hold on to predictions and targets
                all_preds[target].append(list(pred[:,2*i:2*(i+1)].detach().cpu().numpy()[:,1]))
                all_targets[target].append(list(data.y[:,i].flatten().numpy()))

    scores = {target: None for target in target_list}
    for target in target_list:
        scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
        print("{0} TEST ROC: {1:.3f}".format(target, scores[target]))

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
                    all_preds[target].append(list(pred[:,2*i:2*(i+1)].detach().cpu().numpy()[:,1]))
                    all_targets[target].append(list(data.y[:,i].flatten().numpy()))

        scores = {target: None for target in target_list}
        for target in target_list:
            if(t != target):
                continue
            scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
            print("{0} TEST ROC: {1:.3f}".format(target, scores[target]))
        model.train()


def validate(model, val_loader, device, target_list, train=False):
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
                # Hold on to predictions and targets
                all_preds[target].append(list(pred[:,2*i:2*(i+1)].detach().cpu().numpy()[:,1]))
                all_targets[target].append(list(data.y[:,i].flatten().numpy()))

    scores = {target: None for target in target_list}
    for target in target_list:
        scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
        print("{0} VALIDATION ROC: {1:.3f}".format(target, scores[target]))

    model.train()
    return scores


def train(model, train_loader, val_loader, optimizer, criterion, device, save_string):
    epochs = 400//(train_loader.dataset.aug_time + 1)
    target_list = train_loader.dataset.target
    best_score = {target: 0 for target in target_list}
    for epoch in range(epochs):

        # Hold on to predictions and targets for ROC calculation
        all_preds = {target: [] for target in target_list}
        all_targets = {target: [] for target in target_list}
        total_loss = 0

        for bn, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            # Get data
            x = data.x.float().to(device=device)
            edge_index = data.edge_index.to(device=device)
            edge_attr = data.edge_attr.float().to(device=device)

            # Predict
            pred = model(x, edge_index, edge_attr, data.batch.to(device=device))
            
            # Calculate loss in multitask way
            loss = 0.
            for i, target in enumerate(target_list):
                loss += criterion(pred[:,2*i:2*(i+1)], data.y[:,i].to(device=device))

                # Hold on to predictions and targets
                all_preds[target].append(list(pred[:,2*i:2*(i+1)].detach().cpu().numpy()[:,1]))
                all_targets[target].append(list(data.y[:,i].flatten().numpy()))

            # Add batch loss
            total_loss += loss.detach()

            # Update
            loss.backward()
            optimizer.step()

        print("EPOCH:\t{0}, LOSS: {1:.3f}".format(epoch, total_loss))
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()

        # Need to do per task also, and keep track per task...
        val_score = validate(model, val_loader, device, target_list)

        for target in target_list:
            if(val_score[target] > best_score[target]):
                print("\nNEW BEST VALIDATION: {0:.3f}".format(val_score[target]))
                best_score[target] = val_score[target]
                print("SAVING TO: {}".format(save_string(target)))
                torch.save(model.state_dict(), save_string(target))
        print()

    return model


if __name__ == '__main__':

    task_name = 'SIDER'
    splitting = 'scaffold'

    # Select GPU
    if(len(sys.argv) > 1):
        device = torch.device(sys.argv[1])
    else:
        device = torch.device('cpu')

    # Set run parameters
    batch_size = 400
    weight_decay = 2.9 # also known as l2_regularization_lambda
    learning_rate = 3.5
    
    print("Instantiating model...")
    radius = 3
    T = 2
    p_dropout = 0.1
    fingerprint_dim = 150
    
    print("Getting Dataset...")
    # Not transforming yet
    transform = Compose([
        RandomAtomMask([0.0, 0.2]),
        #RandomBondDelete([0.0, 0.2])
    ])
    dataset = MoleculeDatasetWrapper(task_name, transform=transform, aug_time=1,
                                     batch_size=batch_size, split=splitting)

    target_list = dataset.labels.keys()
    for seed in [10, 20, 30]:
    
        dataset = MoleculeDatasetWrapper(task_name, transform=transform, aug_time=0,
                                 batch_size=batch_size, seed=seed, split=splitting)

        # Would need to change things here for multiclass
        train_loader, val_loader, test_loader = \
                dataset.get_data_loaders(list(dataset.labels.keys()))
    
        # Model saving
        save_string_pre = './saved_models/' + task_name + "/"
        save_string_pre += 'no_aug_' if dataset.aug_time == 0 else ''
        save_string_post = '_afp'
        save_string_post += '_' + str(seed)
        save_string = lambda x: save_string_pre + x + save_string_post
    
        # Set up training
        output_units_num = 2 * len(dataset.labels.keys())
        model = get_model(radius, T, p_dropout, fingerprint_dim, output_units_num, seed).to(
                          device=device)

        # Multi class?
        optimizer = optim.Adam(model.parameters(), 10**-learning_rate,
                               weight_decay=10**-weight_decay)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        model = train(model, train_loader, val_loader, optimizer, criterion, device, save_string)

        print("RESULTS FOR: {}".format(task_name))
        evaluate(model, test_loader, device, target_list, save_string)
    
