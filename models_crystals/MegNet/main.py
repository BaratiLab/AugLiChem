from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
import os
import csv
import random
from pymatgen.core.structure import Structure
from sklearn.metrics import mean_absolute_error



nfeat_bond = 10
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)

model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)

root_dir = "./data/abx3_cifs/"
id_prop_file = os.path.join(root_dir, 'id_prop.csv')
assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
with open(id_prop_file) as f:
    reader = csv.reader(f)
    id_prop_data = [row for row in reader]
random_seed=123
random.seed(random_seed)
random.shuffle(id_prop_data)

train_size = int(0.8*len(id_prop_data))

train_cifs = []
train_labels = []

for idx in range(train_size):
  cif_id, target = id_prop_data[idx]
  crystal = Structure.from_file(os.path.join(root_dir,cif_id+ '.cif'))
  train_cifs.append(crystal)
  train_labels.append(target)
 # print(type(crystal))

test_cifs = []
test_labels = []
test_size = len(id_prop_data) - train_size

for idx in range(test_size):
  cif_id, target = id_prop_data[train_size + idx]
  crystal = Structure.from_file(os.path.join(root_dir,cif_id+ '.cif'))
  test_cifs.append(crystal)
  test_labels.append(target)


model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
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
model.train_from_graphs(graphs_valid, targets_valid,epochs=50)

predict = []
for i in range(len(test_cifs)):
  pred_target  = model.predict_structure(test_cifs[i])
  predict.append(pred_target)

print(mean_absolute_error(test_labels, predict))



