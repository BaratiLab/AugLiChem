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



def get_model(task, radius, T, p_dropout, fingerprint_dim, output_units_num, seed):
    #model = AFP(num_layers=radius,
    #            num_timesteps=T,
    #            dropout=p_dropout,
    #            hidden_channels=fingerprint_dim,
    #            out_channels=output_units_num,
    #            edge_dim=2,
    #            in_channels=2,
    #            seed=seed
    #)
    model = AFP(
            task=task,
            emb_dim=fingerprint_dim,
            num_layers=radius,
            num_timesteps=T,
            drop_ratio=p_dropout,
            out_dim=output_units_num
    )

    return model


def evaluate(model, test_loader, device, target_list, save_string):
    print("\nEvaluating using model at end of training...")
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
            _, pred = model(data.to(device=device))
            #pred = model(x, edge_index, edge_attr)#, data.batch.to(device=device))

            for i, target in enumerate(target_list):
                # Get indices where there's no data and exclude those from pred and data.y
                good_idx = np.where(data.cpu().y[:,i]!=-999999999)

                # Hold on to predictions and targets
                if(test_loader.dataset.task == 'classification'):
                    current_preds = pred[:,2*i:2*(i+1)][good_idx]
                    current_targets = data.y[:,i][good_idx].to(device=device)
                    if(val_loader.dataset.dataset == 'MUV'):
                        all_preds[target].append(list(current_preds.detach().cpu().numpy()))
                    else:
                        all_preds[target].append(list(current_preds.detach().cpu().numpy()[:,1]))
                    all_preds[target].append(list(current_preds.detach().cpu().numpy()[:,1]))
                    all_targets[target].append(list(current_targets.cpu().flatten().numpy()))

                elif(test_loader.dataset.task == 'regression'):
                    current_preds = pred[good_idx]
                    current_targets = data.y[:,i][good_idx].to(device=device)
                    all_preds[target].append(current_preds.detach().cpu().numpy()[:,0])
                    all_targets[target].append(current_targets.cpu().numpy().flatten())

    scores = {target: None for target in target_list}
    for target in target_list:
        if(test_loader.dataset.task == 'classification'):
            if(val_loader.dataset.dataset == 'MUV'):
                scores[target] = accuracy_score(all_targets[target][0],
                                                np.argmax(all_preds[target][0], axis=1))
                print("{0} TEST ACCURACY: {1:.5f}".format(target, scores[target]))
            else:
                scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
                print("{0} TEST ROC: {1:.5f}".format(target, scores[target]))
        elif(test_loader.dataset.task == 'regression'):
            scores[target] = mean_squared_error(all_targets[target], all_preds[target],
                                                squared=False)
            print("{0} TEST RMSE: {1:.5f}".format(target, scores[target]))

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
                _, pred = model(data.to(device=device))
                #pred = model(x, edge_index, edge_attr)#, data.batch.to(device=device))

                for i, target in enumerate(target_list):
                    if(t != target):
                        continue

                    # Get indices where there's no data and exclude those from pred and data.y
                    good_idx = np.where(data.cpu().y[:,i]!=-999999999)

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

        scores = {target: None for target in target_list}
        for target in target_list:
            if(t != target):
                continue
            if(test_loader.dataset.task == 'classification'):
                if(val_loader.dataset.dataset == 'MUV'):
                    scores[target] = accuracy_score(all_targets[target][0],
                                                    np.argmax(all_preds[target][0], axis=1))
                    print("{0} TEST ACCURACY: {1:.5f}".format(target, scores[target]))
                else:
                    scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
                    print("{0} TEST ROC: {1:.5f}".format(target, scores[target]))
            elif(test_loader.dataset.task == 'regression'):
                scores[target] = mean_squared_error(all_targets[target], all_preds[target],
                                                squared=False)
                print("{0} TEST RMSE: {1:.5f}".format(target, scores[target]))
        model.train()


def validate(model, val_loader, device, target_list, train=False):
    with torch.no_grad():
        model.eval()
        all_preds = {target: [] for target in target_list}
        all_targets = {target: [] for target in target_list}
        #for bn, data in enumerate(val_loader):
        data = next(iter(val_loader))
        
        # Get data
        x = data.x.float().to(device=device)
        edge_index = data.edge_index.to(device=device)
        edge_attr = data.edge_attr.float().to(device=device)
        
        # Predict
        _, pred = model(data.to(device=device))
        #pred = model(x, edge_index, edge_attr)#, data.batch.to(device=device))

        for i, target in enumerate(target_list):
            # Get indices where there's no data and exclude those from pred and data.y
            good_idx = np.where(data.cpu().y[:,i]!=-999999999)

            # Hold on to predictions and targets
            if(train_loader.dataset.task == 'classification'):
                current_preds = pred[:,2*i:2*(i+1)][good_idx]
                current_targets = data.y[:,i][good_idx].to(device=device)

                # Save predictions
                if(val_loader.dataset.dataset == 'MUV'):
                    all_preds[target].append(list(current_preds.detach().cpu().numpy()))
                else:
                    all_preds[target].append(list(current_preds.detach().cpu().numpy()[:,1]))

                # Save targets
                all_targets[target].append(list(current_targets.cpu().flatten().numpy()))

            elif(train_loader.dataset.task == 'regression'):
                current_preds = pred[good_idx]
                current_targets = data.y[:,i][good_idx].to(device=device)
                all_preds[target].append(current_preds.detach().cpu().numpy()[:,0])
                all_targets[target].append(current_targets.cpu().numpy().flatten())

    scores = {target: None for target in target_list}
    for target in target_list:
        if(val_loader.dataset.task == 'classification'):
            if(val_loader.dataset.dataset == 'MUV'):
                scores[target] = accuracy_score(all_targets[target][0],
                                                np.argmax(all_preds[target][0], axis=1))
                print("{0} VALIDATION ACCURACY: {1:.5f}".format(target, scores[target]))
            else:
                #print(all_preds[target])
                #print(all_preds[target][0])
                #print(all_targets[target][0])
                #raise
                scores[target] = roc_auc_score(all_targets[target][0], all_preds[target][0])
                print("{0} VALIDATION ROC: {1:.5f}".format(target, scores[target]))
        elif(val_loader.dataset.task == 'regression'):
            scores[target] = mean_squared_error(all_targets[target], all_preds[target],
                                                squared=False)
            print("{0} VALIDATION RMSE: {1:.5f}".format(target, scores[target]))

    model.train()
    return scores


def train(model, train_loader, val_loader, optimizer, criterion, device, save_string, epochs=100):
    target_list = train_loader.dataset.target
    if(train_loader.dataset.task == 'classification'):
        best_score = {target: 0 for target in target_list}
    elif(train_loader.dataset.task == 'regression'):
        best_score = {target: 99999 for target in target_list}

    # Plot validation if we're doing regression
    if(val_loader.dataset.task == 'regression'):
        fig, ax = plt.subplots()
        val_scores = []
        train_scores = []

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
            _, pred = model(data.to(device=device))
            #pred = model(x, edge_index, edge_attr)#, data.batch.to(device=device))
            
            # Calculate loss in multitask way
            loss = 0.
            for i, target in enumerate(target_list):

                # Get indices where there's no data and exclude those from pred and data.y
                #good_idx = np.where((data.y[:,i]==0) | (data.y[:,i]==1))
                good_idx = np.where(data.cpu().y[:,i]!=-999999999)

                # Hold on to predictions and targets
                if(train_loader.dataset.task == 'classification'):
                    current_preds = pred[:,2*i:2*(i+1)][good_idx]
                    current_targets = data.y[:,i][good_idx].to(device=device)
                    loss += criterion(current_preds, current_targets)
                    all_preds[target].append(list(current_preds.detach().cpu().numpy()[:,1]))
                    all_targets[target].append(list(current_targets.cpu().flatten().numpy()))

                elif(train_loader.dataset.task == 'regression'):
                    current_preds = pred[good_idx]
                    current_targets = data.y[:,i][good_idx].to(device=device)
                    loss += criterion(current_preds[:,0], current_targets)
                    all_preds[target].append(current_preds.detach().cpu().numpy()[:,0])
                    all_targets[target].append(current_targets.cpu().numpy().flatten())

            # Add batch loss
            total_loss += loss.detach()

            # Update
            loss.backward()
            optimizer.step()

        print("EPOCH:\t{0}, LOSS: {1:.5f}".format(epoch, total_loss))
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()

        # Need to do per task also, and keep track per task...
        val_score = validate(model, val_loader, device, target_list)
        if(val_loader.dataset.task == 'regression'):
            val_scores.append(val_score[target_list[0]])
            flat_preds = []
            for array in all_preds[0][target_list[0]]:
                for val in array:
                    flat_preds.append(val)
            flat_targets = []
            for array in all_targets[0][target_list[0]]:
                for val in array:
                    flat_targets.append(val)

            train_scores.append(mean_squared_error(flat_preds, flat_targets, squared=False))

        for target in target_list:
            condition = val_score[target] > best_score[target] \
                        if(train_loader.dataset.task == 'classification') else \
                        val_score[target] < best_score[target]
            #if(val_score[target] > best_score[target]):
            if(condition):
                print("\nNEW BEST VALIDATION: {0:.5f}".format(val_score[target]))
                best_score[target] = val_score[target]
                print("SAVING TO: {}".format(save_string(target)))
                torch.save(model.state_dict(), save_string(target))
        print()
    if(val_loader.dataset.task == 'regression'):
        plt.plot(train_scores, label="Train Loss")
        plt.plot(val_scores, label="Validation Loss")
        plt.savefig(save_string(target_list[0]) + ".png")

    return model


if __name__ == '__main__':

    task_name = 'BBBP'
    splitting = 'random'
    aug_time = 2

    # Select GPU
    if(len(sys.argv) > 1):
        device = torch.device(sys.argv[1])
    else:
        device = torch.device('cpu')

    # Set run parameters
    batch_size = 100
    epochs = 100
    radius = 3
    T = 2
    fingerprint_dim = 150
    p_dropout = 0.1
    weight_decay = 2.9 # also known as l2_regularization_lambda
    learning_rate = 3.5
    
    # Set up transformation
    transform = Compose([
        RandomAtomMask([0.0, 0.2]),#[0.1, 0.2]),
        RandomBondDelete([0.0, 0.2])
    ])

    for seed in [10, 20, 30]:
    
        print("Getting Dataset...")
        dataset = MoleculeDatasetWrapper(task_name, transform=transform, aug_time=aug_time,
                                 batch_size=batch_size, seed=seed, split=splitting)
        target_list = dataset.labels.keys()


        # Would need to change things here for multiclass
        train_loader, val_loader, test_loader = \
                dataset.get_data_loaders('all')
                #dataset.get_data_loaders(list(dataset.labels.keys()))
    
        # Model saving
        save_string_pre = './saved_models/new_afp_' + task_name + "/"
        save_string_pre += 'no_aug_' if dataset.aug_time == 0 else ''
        save_string_post = '_afp'
        save_string_post += '_' + str(seed)
        save_string = lambda x: save_string_pre + x + save_string_post
    
        # Set up training
        if(dataset.task == 'classification'):
            output_units_num = 2 * len(dataset.labels.keys())
        elif(dataset.task == 'regression'):
            output_units_num = 1

        # Get model
        print("Instantiating model...")
        model = get_model(dataset.task, radius, T, p_dropout, fingerprint_dim, output_units_num, seed).to(
                          device=device)
                  
        # Get optimizer and loss function
        optimizer = optim.Adam(model.parameters(), 10**-learning_rate,
                               weight_decay=10**-weight_decay)
        if(dataset.task == 'classification'):
            criterion = nn.CrossEntropyLoss(reduction='mean')
        elif(dataset.task == 'regression'):
            criterion = nn.MSELoss()

        # Save training parameters
        params = {'weight_decay': weight_decay,
                  'learning_rate': learning_rate, 
                  'atom_mask': transform.transforms[0].prob,
                  'bond_delete': transform.transforms[1].prob,
                  'T': T, 'radius': radius, 'dropout': p_dropout,
                  'fingerprint_dim': fingerprint_dim,
                  'splitting': splitting,
                  'epochs': epochs
        }
        np.save(save_string("parameters"), params)

        # Train our model
        model = train(model, train_loader, val_loader, optimizer, criterion, device, save_string,
                      epochs)

        print("RESULTS FOR: {}".format(task_name))

        # Evaluate our model
        evaluate(model, test_loader, device, target_list, save_string)
    
