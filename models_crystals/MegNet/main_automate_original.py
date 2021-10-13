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
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--fold', default= 0, type=int, metavar='N',
                    help='Fold number')
parser.add_argument('--dataset_run', default= "lanths", type=str, metavar='N',
                    help='Dataset run')
parser.add_argument('--random_seed', default= 0, type=int, metavar='N',
                    help='seed number')

args = parser.parse_args(sys.argv[1:])



nfeat_bond = 10
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)

model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)

root_dir = "./data/" + args.dataset_run

id_prop_train_file = os.path.join(root_dir, 'id_prop.csv')#.format(args.fold))
assert os.path.exists(id_prop_train_file), 'id_prop.csv does not exist!'#.format(args.fold)

with open(id_prop_train_file) as f:
    reader = csv.reader(f)
    id_prop_train_data = [row for row in reader]

print("TRA",len(id_prop_train_data))

total_size = len(id_prop_train_data)
indices = list(range(total_size))
train_size = int(0.64*len(id_prop_train_data))
val_size = int(0.16*len(id_prop_train_data))
test_size = total_size - (val_size+train_size)
random.seed(args.random_seed)
random.shuffle(indices)

train_idx = indices[:train_size]
val_idx = indices[-(val_size + test_size):-test_size]
test_idx = indices[-test_size:]
 # print(type(crystal))

train_cifs = []
train_labels = []

val_cifs = []
val_labels = []
# tes_size = len(id_prop_data) - train_size
test_cifs = []
test_labels = []

for idx in (train_idx):
  cif_id, target = id_prop_train_data[idx]
  crystal = Structure.from_file(os.path.join(root_dir,cif_id+ '.cif'))
  train_cifs.append(crystal)
  train_labels.append(target)

for v_idx in (val_idx):
  cif_id, target = id_prop_train_data[v_idx]
  crystal = Structure.from_file(os.path.join(root_dir,cif_id+ '.cif'))
  val_cifs.append(crystal)
  val_labels.append(target)

for t_idx in (test_idx):
  cif_id, target = id_prop_train_data[t_idx]
  crystal = Structure.from_file(os.path.join(root_dir,cif_id+ '.cif'))
  test_cifs.append(crystal)
  test_labels.append(target)

model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width, batch_size = 16)
graphs_valid = []
targets_valid = []
structures_invalid = []
for s, p in zip(train_cifs, train_labels):
    try:
        graph = model.graph_converter.convert(s)
        #print(type(p))
        graphs_valid.append(graph)
        targets_valid.append(float(p))
    except:
        structures_invalid.append(s)
model.train_from_graphs(graphs_valid, targets_valid,epochs=args.epochs)

predict_val = []
predict_test  = []
#best_val_error = 10e10

for i in range(len(val_cifs)):
  pred_target_val  = model.predict_structure(val_cifs[i])
  predict_val.append(pred_target_val)

for j in range(len(test_cifs)):
  pred_target_test = model.predict_structure(test_cifs[i]) 
  predict_test.append(pred_target_test)


mae_val = mean_absolute_error(val_labels, predict_val)
mae_test = mean_absolute_error(test_labels, predict_test)

f = open("./results_original/error_record_{}.csv".format(args.dataset_run), "a")
f.write("Fold,best_mae_error_val,best_mae_error_test")
f.write("\n")
f.write("%s,%s,%s"%(args.fold,mae_val,mae_test))
f.write("\n")

filename_model = args.dataset_run+'_'+str(args.fold)+'.hdf5'
model.save_model('./models_original/{}'.format(filename_model))



