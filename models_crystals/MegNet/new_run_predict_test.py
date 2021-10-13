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
    while fold < 5:          
        print("at fold:",fold)
        os.system('python -W ignore main_automate_test.py'+' '+' '+ '--fold'+" "+ str(fold)+" "+'--dataset_run'+" "+ dataset+ ' ')
        fold+=1

dataset_list = ['FE']#['abx3_cifs','fermi','lanths','band']
for i in range(len(dataset_list)):
    dataset_run = "{}".format(dataset_list[i])
    running(dataset = dataset_run)