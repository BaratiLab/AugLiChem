import os
import yaml
import pandas as pd
import shutil
import sys
sys.path.append(sys.path[0][:-8])

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
        shutil.copy('./config_aug.yaml', os.path.join(model_checkpoints_folder, 'config_aug.yaml'))


class Train(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['run']['dataset'] + '_' + config['dataset']['target']
        log_dir = os.path.join('runs_aug', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if dataset.task == 'classification':
            self.criterion = nn.CrossEntropyLoss()
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
        loss_trace = []
        valid_rmse_trace = []
        valid_rmse_trace_x = []

        for epoch_counter in range(self.config['run']['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)

                loss = self._step(model, data, n_iter)
                loss_trace.append(loss.detach())

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
                    valid_loss, valid_roc_auc = self._validate(model, test_loader)
                    if valid_roc_auc > best_valid_roc_auc:
                        # save the model weights
                        best_valid_roc_auc = valid_roc_auc
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.dataset.task == 'regression': 
                    valid_loss, valid_rmse = self._validate(model, test_loader)
                    valid_rmse_trace.append(valid_rmse)
                    valid_rmse_trace_x.append((epoch_counter+1)*bn)
                    if valid_rmse < best_valid_rmse:
                        # save the model weights
                        best_valid_rmse = valid_rmse
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        self._test(model, test_loader)
        #fig, ax = plt.subplots()
        #ax.plot(loss_trace)
        #ax.plot(valid_rmse_trace_x, valid_rmse_trace)
        #plt.show()

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

        if self.dataset.task == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.rmse = mean_squared_error(labels, predictions, squared=False)
            print('Test loss:', test_loss)
            print('Test RMSE:', self.rmse)

        elif self.dataset.task == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Test loss:', test_loss)
            print('Test ROC AUC:', self.roc_auc)


#def main(config):
#    dataset = MolDatasetWrapper(config['batch_size'], **config['dataset'])
#
#    trainer = Train(dataset, config)
#    trainer.train()
#
#    if config['dataset']['task'] == 'classification':
#        return trainer.roc_auc
#    if config['dataset']['task'] == 'regression':
#        return trainer.rmse
#
#
#if __name__ == "__main__":
#    config = yaml.load(open("config_aug.yaml", "r"), Loader=yaml.FullLoader)
#    print(config)
#
#    if config['task_name'] == 'BBBP':
#        config['dataset']['task'] = 'classification'
#        config['dataset']['data_path'] = './data/bbbp/raw/BBBP.csv'
#        target_list = ["p_np"]
#
#    elif config['task_name'] == 'Tox21':
#        config['dataset']['task'] = 'classification'
#        config['dataset']['data_path'] = './data/tox21/raw/tox21.csv'
#        target_list = [
#            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
#            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
#        ]
#
#    elif config['task_name'] == 'ClinTox':
#        config['dataset']['task'] = 'classification'
#        config['dataset']['data_path'] = './data/clintox/raw/clintox.csv'
#        target_list = ['CT_TOX', 'FDA_APPROVED']
#
#    elif config['task_name'] == 'HIV':
#        config['dataset']['task'] = 'classification'
#        config['dataset']['data_path'] = './data/hiv/raw/HIV.csv'
#        target_list = ["HIV_active"]
#
#    elif config['task_name'] == 'BACE':
#        config['dataset']['task'] = 'classification'
#        config['dataset']['data_path'] = './data/bace/raw/bace.csv'
#        target_list = ["Class"]
#
#    elif config['task_name'] == 'SIDER':
#        config['dataset']['task'] = 'classification'
#        config['dataset']['data_path'] = './data/sider/raw/sider.csv'
#        target_list = [
#            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders", "Investigations", 
#            "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances", 
#            "Immune system disorders", "Reproductive system and breast disorders", 
#            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
#            "General disorders and administration site conditions", 
#            "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders", "Blood and lymphatic system disorders", 
#            "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders", "Infections and infestations", 
#            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders", 
#            "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders", 
#            "Nervous system disorders", "Injury, poisoning and procedural complications"
#        ]
#    
#    elif config['task_name'] == 'MUV':
#        config['dataset']['task'] = 'classification'
#        config['dataset']['data_path'] = './data/sider/raw/muv.csv'
#        target_list = [
#            "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-692", "MUV-712", "MUV-713", 
#            "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
#        ]
#    
#    elif config['task_name'] == 'FreeSolv':
#        config['dataset']['task'] = 'regression'
#        config['dataset']['data_path'] = './data/freesolv/raw/SAMPL.csv'
#        target_list = ["expt"]
#    
#    elif config["task_name"] == 'ESOL':
#        config['dataset']['task'] = 'regression'
#        config['dataset']['data_path'] = './data/esol/raw/delaney-processed.csv'
#        target_list = ["measured log solubility in mols per litre"]
#
#    elif config["task_name"] == 'Lipo':
#        config['dataset']['task'] = 'regression'
#        config['dataset']['data_path'] = './data/lipophilicity/raw/Lipophilicity.csv'
#        target_list = ["exp"]
#
#    save_dir = 'experiments_aug'
#    os.makedirs(save_dir, exist_ok=True)
#
#    roc_list = []
#    total_start = time.time()
#    for target in target_list:
#        config['dataset']['target'] = target
#        start = time.time()
#        result1 = main(config)
#        print("TRANING TIME: {}".format(time.time() - start))
#        start = time.time()
#        result2 = main(config)
#        print("TRANING TIME: {}".format(time.time() - start))
#        start = time.time()
#        result3 = main(config)
#        print("TRANING TIME: {}".format(time.time() - start))
#        roc_list.append([target, result1, result2, result3])
#
#    print("\n\nTOTAL TRAINING TIME: {}".format(time.time() - total_start))
#    df = pd.DataFrame(roc_list)
#    fn = os.path.join(save_dir, '{}_list.csv'.format(config['task_name']))
#    df.to_csv(fn, index=False)

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
    config = yaml.load(open("config_aug.yaml", "r"), Loader=yaml.FullLoader)

    save_dir = 'experiments_aug'
    os.makedirs(save_dir, exist_ok=True)

    main(config)
