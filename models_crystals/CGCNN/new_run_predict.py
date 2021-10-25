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
    fold = 1
    while fold < 5:          
        print("at fold:",fold)
        os.system('python main.py'+' '+ 'data/'+ dataset +' '+ '--fold'+" "+ str(fold)+" "+'--dataset_run'+" "+ dataset+ ' ' + '--train-ratio'+" "+ str(0.8)+ " " +'--val-ratio'+" "+ str(0.2))
        fold+=1

dataset_run = "Augmented_lanths"
running(dataset = dataset_run)