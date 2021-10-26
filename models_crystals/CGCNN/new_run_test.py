import os
import numpy as np
import json
import time
from string import Template
import glob
import re
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import csv

def running(dataset = "abx3_cifs", test_data = 'abx3_cifs'):
    h = open("results_prediction_{}.csv".format(test_data), "a")
    h.write("Mean, Max, Min, Max_fold, Min Fold")
    h.write("\n")
    mae = []
    fold = 0
    model_filename = 'model_best_{}_{}.pth.tar'.format(fold,dataset)
    results_file = ''
    while fold < 5:          
        print("at fold:",fold)
        os.system('python predict.py'+' '+ './models/'+ model_filename +' '+'data/'+ test_data +' '+ '--fold'+" "+ str(fold)+" "+'--dataset_run'+" "+ test_data)
        results_fname = './results_predict/test_results_{}_{}.csv'.format(test_data,fold)
        results_data = np.genfromtxt(results_fname, delimiter=',')
        target = (results_data[:,1])
        pred = results_data[:,2]
        mae.append(mean_absolute_error(target,pred))
        mae_arr = np.asarray(mae)
        max_fold = np.argmax(mae)
        mae_max = np.max(mae_arr)
        mae_mean = np.mean(mae_arr)
        mae_min = np.min(mae_arr)
        min_fold = np.argmin(mae)
        fold+=1
    h.write("%s,%s,%s,%s,%s" %(mae_mean,mae_max,mae_min,max_fold,min_fold))
    h.write("\n")



dataset_list = ['abx3_cifs']#['band','fermi','FE','abx3_cifs']

for i in range(len(dataset_list)):
    dataset_run = "Augmented_{}".format(dataset_list[i])
    dataset_test = dataset_list[i]
    running(dataset = dataset_run, test_data = dataset_test)