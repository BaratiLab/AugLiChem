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
parser.add_argument('--dataset_run', default= "Augmented_lanths", type=str, metavar='N',
                    help='Dataset run')

args = parser.parse_args(sys.argv[1:])



nfeat_bond = 10
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)

model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)

root_dir = "./data/" + args.dataset_run

id_prop_train_file = os.path.join(root_dir, 'id_prop_train_{}.csv'.format(args.fold))
assert os.path.exists(id_prop_train_file), 'id_prop_train_{}.csv does not exist!'.format(args.fold)

id_prop_augment_file = os.path.join(root_dir, 'id_prop_augment_{}.csv'.format(args.fold))
assert os.path.exists(id_prop_augment_file), 'id_prop_augment_{}.csv does not exist!'.format(args.fold)

with open(id_prop_train_file) as f:
    reader = csv.reader(f)
    id_prop_train_data = [row for row in reader]

print("TRA",len(id_prop_train_data))

total_size = len(id_prop_train_data)
indices = list(range(total_size))
train_size = int(0.8*len(id_prop_train_data))
random.shuffle(indices)
train_idx = indices[:train_size]
train_idx_augment = []
val_idx = indices[train_size:]

num_aug = 4
for i in range (len(train_idx)):
    idx_correction = num_aug*train_idx[i]
    add_1 = idx_correction + 1
    add_2 = idx_correction + 2
    add_3 = idx_correction + 3
    add_  = idx_correction
    train_idx_augment.append(add_1)
    train_idx_augment.append(add_2)
    train_idx_augment.append(add_3)
    train_idx_augment.append(add_)

print("MAX_AUG",max(train_idx_augment))
print("MAX_TRA",max(train_idx))
#for i in range(len())
train_cifs = []
train_labels = []

with open(id_prop_augment_file) as f:
    reader = csv.reader(f)
    id_prop_augment_data = [row for row in reader]

print("LEN",len(id_prop_augment_data))
for idx in (train_idx_augment):
  cif_id, target = id_prop_augment_data[idx]
  crystal = Structure.from_file(os.path.join(root_dir,cif_id+ '.cif'))
  train_cifs.append(crystal)
  train_labels.append(target)
 # print(type(crystal))

val_cifs = []
val_labels = []
# tes_size = len(id_prop_data) - train_size

for v_idx in (val_idx):
  cif_id, target = id_prop_train_data[v_idx]
  crystal = Structure.from_file(os.path.join(root_dir,cif_id+ '.cif'))
  val_cifs.append(crystal)
  val_labels.append(target)


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

#print(len(graphs_valid))
# train the model using valid graphs and targets
model.train_from_graphs(graphs_valid, targets_valid,epochs=args.epochs)

#model2 = MEGNetModel.from_file('./models/{}'.format(filename_model))
predict = []
for i in range(len(val_cifs)):
  pred_target  = model.predict_structure(val_cifs[i])
  predict.append(pred_target)

mae = mean_absolute_error(val_labels, predict)

f = open("./results_val/val_error_record_{}.csv".format(args.dataset_run), "a")
f.write("Fold,best_mae_error")
f.write("\n")
f.write("%s,%s"%(args.fold,mae))
f.write("\n")

filename_model = args.dataset_run+'_'+str(args.fold)+'.hdf5'
model.save_model('./models/{}'.format(filename_model))



