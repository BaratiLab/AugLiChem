---
layout: page
title: Molecule Usage
permalink: /molecule_usage/
---

AugLiChem has been designed from the ground up with ease-of-use in mind.
Fully functional notebooks are available at our github in the `examples/` directory [here](https://github.com/BaratiLab/AugLiChem/tree/main/examples).
In-depth documentation of each function is given in the docstrings and can be printed out using python's built-in `help()` function.
Using PyTorch's CUDA support, all models and data sets can be used with GPUs.


This guide explains all of the features of the package.
We have also provided jupyter notebooks that are ready to run after installation.
Links to notebooks that demonstrate each type of training are provided below:

- Molecule:
  - [Single target](https://github.com/BaratiLab/AugLiChem/blob/main/examples/molecule_dataset.ipynb)
  - [Multitarget](https://github.com/BaratiLab/AugLiChem/blob/main/examples/molecule_multitarget_dataset.ipynb)
- Crystal:
  - [Standard training](https://github.com/BaratiLab/AugLiChem/blob/main/examples/crystal_dataset.ipynb)
  - [Automatic k-fold cross validation](https://github.com/BaratiLab/AugLiChem/blob/main/examples/crystal_kfold_dataset.ipynb)
  - [CGCNN standard training](https://github.com/BaratiLab/AugLiChem/blob/main/examples/crystal_cgcnn_dataset.ipynb)
  - [CGCNN k-fold cross validation](https://github.com/BaratiLab/AugLiChem/blob/main/examples/crystal_cgcnn_kfold_dataset.ipynb).


## Molecule Usage

The first step is to import the relevant modules.
AugLiChem is largely self-contained, and so we import transformations, data wrapper, and models.

### Setup
```python
from auglichem.molecule import RandomAtomMask, RandomBondDelete, MotifRemoval
from auglichem.molecule.data import MoleculeDatasetWrapper
from auglichem.molecule.models import AttentiveFP, GCN, DeepGCN, GINE
```

Next, we set up our transformations.
Transformations can be set up as a list or single transformation.
When using a list, each molecule is transformed by all transformations passed in.


### Creating augmentations
```python
transforms = [
      RandomAtomMask([0.1,0.3]),
      RandomBondDelete([0.1, 0.4]),
      MotifRemoval(0.6)
]
```
RandomAtomMask and RandomBondDelete take in either a list of two number, or a single number, which represents the fraction of atoms to mask and bonds to delete, respectively.
When a list is passed in, a number is sampled uniformly between the two values for each molecule and used for the masking/deletion fraction.
MotifRemoval is deterministic, and uses a similarity score between motifs and the original molecule.
Motifs that are above the similarity score threshold, the passed in parameter, are retained for the augmented molecule data.

### Data Loading
After initializing our transformations, we are ready to initialize our data set.
Data sets are selected with a string, and are automatically downloaded to `./data_download` by default.
This directory is created if it is not present, and does not download the data again if it is already present.
Batch size, validation size, and test size for training and evaluation are set here.
The transforms are passed in here and scaffold splitting is supported.

```python
dataset = MoleculeDatasetWrapper(
             dataset="ClinTox",
             transform=transforms,
             split="scaffold",
             batch_size=128,
             num_workers=0
             valid_size=0.1,
             test_size=0.1,
             aug_time=0,
             data_path="./data_download",
             seed=None
)
```
`MoleculeDatasetWrapper` arguments:
- `dataset` (str): One of the datasets available from MoleculeNet
               (http://moleculenet.ai/datasets-1)
- `transform` (Compose, OneOf, RandomAtomMask, RandomBondDelete object): transormations
               to apply to the data at call time.
- `split` (str, optional default=scaffold): random or scaffold. The splitting strategy
                                        used for train/test/validation set creation.
- `batch_size` (int, optional default=64): Batch size used in training
- `num_workers` (int, optional default=0): Number of workers used in loading data
- `valid_size` (float in [0,1], optional default=0.1): 
- `test_size` (float in [0,1],  optional default=0.1): 
- `aug_time` (int, optional default=0): Number of times to call each augmentation
- `data_path` (str, optional default=None): specify path to save/lookup data. Default
            creates `data_download` directory and stores data there
- `seed` (int, optional, default=None): Random seed to use for reproducibility

After loading our data, our `dataset` object has additional information from the parent class, `MoleculeDataset` that may be useful to look at.
We can look at the SMILES representation of each molecule in the data, as well as the targets:

```python
>>> print(dataset.smiles_data)
['[C@@H]1([C@@H]([C@@H]([C@H]([C@@H]([C@@H]1Cl)Cl)Cl)Cl)Cl)Cl'
 '[C@H]([C@@H]([C@@H](C(=O)[O-])O)O)([C@H](C(=O)[O-])O)O'
 '[H]/[NH+]=C(/C1=CC(=O)/C(=C\\C=c2ccc(=C([NH3+])N)cc2)/C=C1)\\N' ...
 'O=[Zn]' 'OCl(=O)(=O)=O' 'S=[Se]=S']
```

and the labels can be viewed with:

```python
>>> print(dataset.labels)
{'CT_TOX': array([0, 0, 0, ..., 0, 0, 0]), 'FDA_APPROVED': array([1, 1, 1, ..., 1, 1, 1])}
```


### Data Splitting
Using the wrapper class is preferred for easy training because of the data loader function, which creates pytorch-geometric data loaders that are easy to iterate over.
With multi-target data sets, such as ClinTox, we specify the target we want here.
If no target is selected, the first target in the downloaded data file is used.
Multiple targets can be selected for multi-target training by passing in a list of targets, or 'all' to use all of them.

```python
train_loader, valid_loader, test_loader = dataset.get_data_loaders("FDA_APPROVED")
```
`MoleculeDatasetWrapper.get_data_loaders()` argument:
- `target` (str, list of str, optional): Target name to get data loaders for. If None,
     	                           returns the loaders for the first target. If 'all'
                                   returns data for all targets at once, ideal for
                                   multitarget trainimg.

Returns:
- `train/valid/test_loader` (DataLoader): Data loaders containing the train, validation
                                        and test splits of our data.


Now that our data is ready for training and evalutaion, we initialize our model.
Task, either regression or classification needs to be passed in.
Our dataset object stores this in the `task` attribute.

### Model Initialization
```python
model = AttentiveFP(
            task=dataset.task,
            emb_dim=300,
            num_layers=5,
            num_timesteps=3,
            drop_ratio=0,
            output_dim=None
)
```
`AttentiveFP` arguments:
- `task` (str): 'classification' or 'regression'
- `edge_dim` (int): Edge feature dimensionality.
- `num_layers` (int): Number of GNN layers.
- `num_timesteps` (int): Number of iterative refinement steps for global readout.
- `dropout` (float, optional): Dropout probability. (default: :obj:`0.0`)
- `output_dim` (int, optional): Output dimension. Defaults to 1 if task='regression', 2 if task='classification'. Pass in the number of targets if doing multi-target classification.


```python
model = DeepGCN(
	[THIS NEEDS TO BE FILLED IN]
)
```
`DeepGCN` arguments:
- **I DON'T KNOW YET**

```python
model = GCN(
            task=dataset.task,
            emb_dim=300,
            feat_dim=256
            num_layers=5,
            pool='mean'
            drop_ratio=0,
            output_dim=None
)
```
`GCN` arguments:
- `task` (str): 'classification' or 'regression'
- `edge_dim` (int): Edge feature dimensionality.
- `feature_dim` (int): Feature dimensionality before final prediction layers.
- `num_layers` (int): Number of GNN layers.
- 'pool' (str): Pooling function to be used. One of 'mean', 'add', 'max'.
- `drop_ratio` (float, optional): Dropout probability. (default: :obj:`0.0`)
- `output_dim` (int, optional): Output dimension. Defaults to 1 if task='regression', 2 if task='classification'. Pass in the number of targets if doing multi-target classification.
After initializing one of the models as seen above, we are ready to train using standard PyTorch training procedure.

```python
model = GINE(
            task=dataset.task,
            emb_dim=300,
            feat_dim=256
            num_layers=5,
            pool='mean'
            drop_ratio=0,
            output_dim=None
)
```
`GINE` arguments:
- `task` (str): 'classification' or 'regression'
- `edge_dim` (int): Edge feature dimensionality.
- `feature_dim` (int): Feature dimensionality before final prediction layers.
- `num_layers` (int): Number of GNN layers.
- 'pool' (str): Pooling function to be used. One of 'mean', 'add', 'max'.
- `drop_ratio` (float, optional): Dropout probability. (default: :obj:`0.0`)
- `output_dim` (int, optional): Output dimension. Defaults to 1 if task='regression', 2 if task='classification'. Pass in the number of targets if doing multi-target classification.
After initializing one of the models as seen above, we are ready to train using standard PyTorch training procedure.


## Single Target Training
```python
import torch
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

Now we have our training loop.

```python
for epoch in range(100):
    for bn, data in tqdm(enumerate(train_loader)):

        optimizer.zero_grad()
        
        _, pred = model(data)
        loss = criterion(pred, data.y.flatten())

        loss.backward()
        optimizer.step()
```

### Evaluation
Evaluation requires storing all predections and labels for each batch, and so we have

```python
from sklearn.metrics import roc_auc_score

with torch.no_grad():
    model.eval()
        
   all_preds = []
    all_labels = []
    for data in test_loader:
        _, pred = model(data)

        # Hold on to all predictions and labels
        all_preds.extend(pred[:,1])
        all_labels.extend(data.y)
    
    metric = roc_auc_score(data.y.cpu(), pred.cpu().detach()[:,1])
    print("TEST ROC: {1:.3f}".format(metric))
```

## Multi-target Training

AugLiChem supports multi-target training as well.
When working with a data set that has multiple targets, we can pass in a list of targets we want, or use all targets at once.
In this example, we use QM8, a multi-target regression set.

```python
dataset = MoleculeDatasetWrapper("QM8", data_path="./data_download", transform=transform, batch_size=5)
train_loader, valid_loader, test_loader = dataset.get_data_loaders("all")
```

Because many of these data sets often have labels for some, but not all targets, empty label values have been filled with a placeholder that we skip during training.
Our training setup is the same as in single-target training:


```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

In our training loop, we see that we only compute the loss when we have a label corresponding to a molecule.

```python
for epoch in range(2):
    for bn, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        loss = 0.
        
        # Get prediction for all data
        _, pred = model(data)
        
        for idx, t in enumerate(train_loader.dataset.target):
            # Get indices where target has a value
            good_idx = np.where(data.y[:,idx]!=-999999999)
            
            current_preds = pred[:,idx][good_idx]
            current_labels = data.y[:,idx][good_idx]
            
            loss += criterion(current_preds, current_labels)
        
        loss.backward()
        optimizer.step()
```

When evaluating, we need to iterate over all targets, and also skip data when there is no label

```python
with torch.no_grad():
    
    # All targets we're evaluating
    target_list = test_loader.dataset.target
    
    # Dictionaries to keep track of predictions and labels for all targets
    all_preds = {target: [] for target in target_list}
    all_labels = {target: [] for target in target_list}
    
    model.eval()
    for data in test_loader:
        # Get prediction for all data
        _, pred = model(data)

        for idx, target in enumerate(target_list):
            # Get indices where target has a value
            good_idx = np.where(data.y[:,idx]!=-999999999)
            
            current_preds = pred[:,idx][good_idx]
            current_labels = data.y[:,idx][good_idx]
            
            # Save predictions and targets
            all_preds[target].extend(list(current_preds.detach().cpu().numpy()))
            all_labels[target].extend(list(current_labels.detach().cpu().numpy()))

    scores = {target: None for target in target_list}
    for target in target_list:
        scores[target] = mean_squared_error(all_labels[target], all_preds[target],
                                            squared=False)
        print("{0} TEST RMSE: {1:.5f}".format(target, scores[target]))
```


## Training with CUDA

AugLiChem takes advantage of PyTorch's CUDA support to leverage GPUs for faster training and evaluation.
To initialize a model on our GPU, we call the `.cuda()` function.

```python
model = GCN(task=dataset.task)
model.cuda()
```
Our training setup is the same as before:

```python
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

The only difference in our training loop is putting our data on the GPU as we train:

```python
for epoch in range(100):
    for bn, data in tqdm(enumerate(train_loader)):

        optimizer.zero_grad()
        
        # data -> GPU
        _, pred = model(data.cuda())
        
        loss = criterion(pred[:,0], data.y.flatten())

        loss.backward()
        optimizer.step()
```

Which we also do for evaluation:

```python
    task = test_loader.dataset.task
    set_str = "VALIDATION" if validation else "TEST"
    with torch.no_grad():
        model.eval()
        
        all_preds = []
        all_labels = []
        for data in test_loader:

            # data -> GPU
            _, pred = model(data.cuda())
            
            # Hold on to all predictions and labels
            all_preds.extend(pred)
            all_labels.extend(data.y)
        
        metric = mse(data.y.cpu(), pred.cpu().detach(), squared=False)
        print("TEST RMSE: {0:.3f}".format(metric))
```
