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

def running(dataset = "abx3_cifs"):
    mae = []
    fold = 0
    random_seed = [1,101,10,11,2]
    while fold < 5:          
        print("at fold:",fold)
        seed = random_seed[fold]
        os.system('python -W ignore main_automate_original.py'+' '+' '+ '--fold'+" "+ str(fold)+" "+'--dataset_run'+" "+ dataset+ ' '+ '--random_seed'+ " "+str(seed))
        fold+=1


dataset_list = ['lanths']
for i in range(len(dataset_list)):
    dataset_run = "{}".format(dataset_list[i])
    running(dataset = dataset_run)