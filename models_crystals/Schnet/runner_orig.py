import os
import yaml
import shutil
import sys
import socket
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import warnings
from dataset.data_wrapper_orig import CrystalDatasetWrapper
warnings.simplefilter("ignore",UserWarning)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class Runner(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # task_name = config['dataset']['data_dir'][5:-5]
        dir_name = current_time
        log_dir = os.path.join('runs', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
        elif config['dataset']['task'] == 'regression':
            self.criterion = nn.MSELoss()
            # self.criterion = nn.L1Loss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        pred = model(data.z,data.pos,data.batch)
        loss = self.criterion(pred, data.y)
        
        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        # model = GINet(**self.config["model"]).to(self.device)
        model = GNNModel(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=eval(self.config['weight_decay'])
        )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_ame = np.inf
        best_valid_roc_auc = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, (data) in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)

        #####        aug_data = aug_data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    # self.writer.add_scalar('current_lr', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_roc_auc = self._validate(model, valid_loader)
                    if valid_roc_auc > best_valid_roc_auc:
                        # save the model weights
                        best_valid_roc_auc = valid_roc_auc
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_ame = self._validate(model, valid_loader)
                    if valid_ame < best_valid_ame:
                        # save the model weights
                        best_valid_ame = valid_ame
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        loss, metric = self._test(model, test_loader)
        return loss, metric

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, (data) in enumerate(valid_loader):
                data = data.to(self.device)
                #print(dir(data))
                pred = model(data.z,data.pos,data.batch)
                #print("Model_predict",pred.shape)
                #print("data lab",data.y.shape)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(labels.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            predictions = valid_loader.dataset.scaler.inverse_transform(predictions)
            labels = valid_loader.dataset.scaler.inverse_transform(labels)
            # rmse = mean_squared_error(labels, predictions, squared=False)
            mae = mean_absolute_error(labels, predictions)
            print('Validation loss:', valid_loss)
            # print('RMSE:', rmse)
            print('MAE:', mae)
            return valid_loss, mae

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            predictions = 1 / (1 + np.exp(-predictions)) 
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions)
            print('Validation loss:', valid_loss)
            print('ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        # test steps
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
            for bn, (data) in enumerate(test_loader):
                data = data.to(self.device)
                #print(dir(data))
                pred = model(data.z,data.pos,data.batch)
                #print("Model_predict",pred.shape)
                #print("data lab",data.y.shape)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(labels.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            predictions = test_loader.dataset.scaler.inverse_transform(predictions)
            labels = test_loader.dataset.scaler.inverse_transform(labels)
            # self.rmse = mean_squared_error(labels, predictions, squared=False)
            self.mae = mean_absolute_error(labels, predictions)
            print('Test loss:', test_loss)
            # print('Test RMSE:', self.rmse)
            print('Test MAE:', self.mae)

            return test_loss, self.mae

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            predictions = 1 / (1 + np.exp(-predictions)) 
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions)
            print('Test loss:', test_loss)
            print('Test ROC AUC:', self.roc_auc)
            
            return test_loss, self.roc_auc


if __name__ == "__main__":
    
    config = yaml.load(open("config_orig.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if config['model_type'] == 'gin':
        from model.gin import GINet as GNNModel
    elif config['model_type'] == 'dimenet':
        from model.dimenet import DimeNet as GNNModel
    elif config['model_type'] == 'schnet':
        from model.schnet import SchNet as GNNModel
    else:
        raise ValueError('Undefined GNN model')

    if 'Augmented_lanths' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'lanths'
    elif 'Augmented_abx3_cifs' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'perovskites'
    elif 'Augmented_band' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'band'
    elif 'Augmented_FE' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'FE'
    elif 'Augmented_fermi' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'fermi'

    if 'lanths' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'lanths'
    elif 'abx3_cifs' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'perovskites'
    elif 'band' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'band'
    elif 'FE' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'FE'
    elif 'fermi' in config['dataset']['data_dir']:
        config['dataset']['task'] = 'regression'
        task_name = 'fermi'
    else:
        raise ValueError('Undefined dataset')

    fold_num = 5

    for i in range(0,fold_num):
        curr_fold = i
        dataset = CrystalDatasetWrapper(config['batch_size'], **config['dataset'],fold = curr_fold)

        runner = Runner(dataset, config)
        loss, metric = runner.train()
        
        df = pd.DataFrame([[loss, metric]])
        os.makedirs('experiments', exist_ok=True)
        df.to_csv('experiments/{}_{}_{}.csv'.format(config['model_type'], task_name,fold_num), index=False, mode='a', header=False)