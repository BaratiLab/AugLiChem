import os
import yaml
import pandas as pd
import shutil
import sys
sys.path.append(sys.path[0][:-8])
print(sys.path[-1])

import socket
import torch
import numpy as np
from datetime import datetime

from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, mean_squared_error

from auglichem.molecule.data import MoleculeData
from auglichem.models import get_model


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_sup.yaml', os.path.join(model_checkpoints_folder, 'config_sup.yaml'))


class Train(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['run']['dataset'] + '_' + config['dataset']['target']
        log_dir = os.path.join('runs_sup', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        #if config['dataset']['task'] == 'classification':
        if dataset.task == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        #elif config['dataset']['task'] == 'regression':
        elif dataset.task == 'regression':
            # self.criterion = nn.MSELoss()
            self.criterion = nn.L1Loss()

    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]

        if self.dataset.task == 'classification': 
            loss = self.criterion(pred, data.y.flatten())
        elif self.dataset.task == 'regression': 
            loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        # model = GINE(self.config['dataset']['task'], **self.config["model"]).to(self.device)
        model = get_model(self.dataset.task, self.config["model"]).to(self.device)
        # model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config['run']['init_lr'], 
            weight_decay=eval(self.config['run']['weight_decay'])
        )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rmse = np.inf
        best_valid_roc_auc = 0

        for epoch_counter in range(self.config['run']['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)

                loss = self._step(model, data, n_iter)

                if n_iter % self.config['run']['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    # self.writer.add_scalar('current_lr', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['run']['eval_every_n_epochs'] == 0:
                if self.dataset.task == 'classification': 
                    valid_loss, valid_roc_auc = self._validate(model, valid_loader)
                    if valid_roc_auc > best_valid_roc_auc:
                        # save the model weights
                        best_valid_roc_auc = valid_roc_auc
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.dataset.task == 'regression': 
                    valid_loss, valid_rmse = self._validate(model, valid_loader)
                    if valid_rmse < best_valid_rmse:
                        # save the model weights
                        best_valid_rmse = valid_rmse
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        self._test(model, test_loader)

    # def _load_pre_trained_weights(self, model):
    #     try:
    #         checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
    #         state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
    #         # model.load_state_dict(state_dict)
    #         model.load_my_state_dict(state_dict)
    #         print("Loaded pre-trained model with success.")
    #     except FileNotFoundError:
    #         print("Pre-trained weights not found. Training from scratch.")

    #     return model

    def _validate(self, model, valid_loader):
        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        if self.dataset.task == 'regression': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions, squared=False)
            print('Validation loss:', valid_loss)
            print('RMSE:', rmse)
            return valid_loss, rmse

        elif self.dataset.task == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Validation loss:', valid_loss)
            print('ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):        
        # model = GINE(**self.config["model"]).to(self.device)
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        #if self.config['dataset']['task'] == 'regression':
        if self.dataset.task == 'regression': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.rmse = mean_squared_error(labels, predictions, squared=False)
            print('Test loss:', test_loss)
            print('Test RMSE:', self.rmse)

        #elif self.config['dataset']['task'] == 'classification': 
        elif self.dataset.task == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Test loss:', test_loss)
            print('Test ROC AUC:', self.roc_auc)


def main(config):
    #dataset = MolDatasetWrapper(config['batch_size'], **config['dataset'])
    dataset = MoleculeData(**config['dataset'], **config['run'])
    roc_list = []
    for key in dataset.labels.keys():
        print(key)
        config['dataset']['target'] = key
        dataset.target = key
        results = []
        for i in range(config['run']['restarts']):
            trainer = Train(dataset, config)
            trainer.train()

            if dataset.task == 'classification':
                results.append(trainer.roc_auc)
            if dataset.task == 'regression':
                results.append(trainer.rmse)

        print("{} RESULTS: {}".format(key, results))
        roc_list.append([key, *results])
        print(roc_list)

    df = pd.DataFrame(roc_list)
    fn = os.path.join(save_dir, '{}_list.csv'.format(config['run']['dataset']))
    df.to_csv(fn, index=False)

if __name__ == "__main__":
    config = yaml.load(open("config_sup.yaml", "r"), Loader=yaml.FullLoader)

    save_dir = 'experiments_sup'
    os.makedirs(save_dir, exist_ok=True)

    main(config)
