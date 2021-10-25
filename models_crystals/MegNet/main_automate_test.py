from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
import os
import csv
import random
import sys
import warnings
from pymatgen.core.structure import Structure
from sklearn.metrics import mean_absolute_error
import argparse

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='MEGNET')

# parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
#                     help='dataset options, started with the path to root dir, '
#                          'then other options')
# parser.add_argument('--epochs', default=50, type=int, metavar='N',
#                     help='number of total epochs to run (default: 30)')

parser.add_argument('--fold', default= 0, type=int, metavar='N',
                    help='Fold number')
parser.add_argument('--dataset_run', default= "lanths", type=str, metavar='N',
                    help='Dataset run')
# parser.add_argument('--random_seed', default= 0, type=int, metavar='N',
#                     help='seed number')

args = parser.parse_args(sys.argv[1:])



nfeat_bond = 10
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)

model_filename = "./models/Augmented_{}_{}".format(args.dataset_run,args.fold)+".hdf5"

model = MEGNetModel.from_file(model_filename)

root_dir = "./data/" + args.dataset_run

id_prop_test_file = os.path.join(root_dir, 'id_prop_test_{}.csv'.format(args.fold))
assert os.path.exists(id_prop_test_file), 'id_prop_test_{}.csv does not exist!'.format(args.fold)

with open(id_prop_test_file) as f:
    reader = csv.reader(f)
    id_prop_test_data = [row for row in reader]

print("TEST",len(id_prop_test_data))

total_size = len(id_prop_test_data)
indices = list(range(total_size))
test_idx = indices

test_cifs = []
test_labels = []

for t_idx in (test_idx):
  cif_id, target = id_prop_test_data[t_idx]
  crystal = Structure.from_file(os.path.join(root_dir,cif_id+ '.cif'))
  test_cifs.append(crystal)
  test_labels.append(target)

predict_test = []

for i in range(len(test_cifs)):
  pred_target_test  = model.predict_structure(test_cifs[i])
  predict_test.append(pred_target_test)

mae_test = mean_absolute_error(test_labels, predict_test)

f = open("./results_test/error_record_{}.csv".format(args.dataset_run), "a")
f.write("Fold,best_mae_error_test")
f.write("\n")
f.write("%s,%s"%(args.fold,mae_test))
f.write("\n")




