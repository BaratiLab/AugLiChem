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
        os.system('python -W ignore main_automate.py'+' '+' '+ '--fold'+" "+ str(fold)+" "+'--dataset_run'+" "+ dataset+ ' ')
        fold+=1

running(dataset = "Augmented_FE")