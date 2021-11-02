---
layout: page
title: Crystal Usage
permalink: /crystal_usage/
---

## Crystal Usage

The first step is to import relevant modules.
Auglichem is largely self-contained, and so we import transforamtions, data wrapper, and models.

### Setup
The first step is to import the relevant modules.
AugLiChem is largely self-contained, and so we import transformations, data wrapper, and models.

###  Creating Augmentations
```python
from auglichem.crystal import (PerturbStructureTransformation,
                               RotationTransformation,
                               SwapAxesTransformation,
                               TranslateSitesTransformation,
                               SupercellTransformation,
)
from auglichem.crystal.data import CrystalDatasetWrapper
from auglichem.crystal.models import SchNet, GINet, CrystalGraphConvNet
```

Next, we set up our transformations.
Transformations can be set up as a list or single transformation.
When using a list, each molecule is transformed by all transformations passed in.

```python
transforms = [
        PerturbStructureTransformation(distance=0.1, min_distance=0.01),
	RotationTransformation(axis=[0,0,1], angle=90),
        SwapAxesTransformation(),
        TranslateSitesTransformation(indices_to_move=[0], translation_vector=[1,0,0],
                                     vector_in_frac_coords=True),
        SupercellTransformation(scaling_matrix=[[1,0,0],[0,1,0],[0,0,1]]),
]
```
`PertubStructureTransformation` arguments:
- `distance` (float, optional, default=0.01): Distance of perturbation in angstroms. All sites will be perturbed by exactly that distance in a random direction. Units in Angstroms.
- `min_distance` (float, optional, default=None): if None, all displacements will be equidistant. If int or float, perturb each site a distance drawn from the uniform distribution between 'min_distance' and 'distance'. Units in Angstroms.

`RotationTransformation` arguments:
- `axis` (list or np.array of ints with shape=(3,1)): Axis of rotation, e.g., [1, 0, 0]
- `angle` (float): Angle to rotate in degrees

`SwapAxesTransformation` arugments:
- None. Axes are randomly selected and swapped.

`TranslateSitesTransformation` arguments:
- `indices_to_move` (list of ints): The indices of the sites to move.
- `translation_vector` (list or np.array of floats, shape=(len(indices_to_move), 3)): Vector to move the sites. Each translation vector is applied to the corresponding site in the indices_to_move.
- `vector_in_frac_coords` (bool, default=True): Set to True if the translation vector is in fractional coordinates, and False if it is in cartesian coordinations.

`SupercellTransformation` arguments:
- `scaling_matrix` (list or np.array of ints with shape=(3,3), default=identity matrix): A matrix of transforming the lattice vectors. Defaults to the identity matrix. Has to be all integers. e.g., [[2,1,0],[0,3,0],[0,0,1]] generates a new structure with lattice vectors a" = 2a + b, b" = 3b, c" = c where a, b, and c are the lattice vectors of the original structure.

### Data Loading 

After initializing our transformations, we are ready to initialize our data set.
Data sets are selected with a string, and are automatically downloaded to `./data_download` by default.
This directory is created if it is not present, and does not download the data again if it is already present.
Batch size, validation size, and test size for training and evaluation are set here.
Unlike molecule, CrystalDatasetWrapper handles transformations when splitting the data.
Random splitting, and k-fold cross validation are supported.

```python
dataset = CrystalDatasetWrapper("lanthanides", batch_size=128, valid_size=0.1, test_size=0.1)
```


```python
dataset = CrystalDatasetWrapper(
             dataset="lanthanides",
             transform=transforms,
             split="scaffold",
             batch_size=128,
             num_workers=0
             valid_size=0.1,
             test_size=0.1,
             data_path="./data_download",
             target=None,
	     kfolds=0,
             seed=None,
             cgcnn=False
)
```
`CrystalDatasetWrapper` arguments:
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
- `target` (str, optional, default=None): Target variable
- `data_path` (str, optional default=None): specify path to save/lookup data. Default
            creates `data_download` directory and stores data there
- `kfolds` (int, default=0, folds > 1): Number of folds to use in k-fold cross
                         validation. kfolds > 1 for data to be split
- `seed` (int, optional, default=None): Random seed to use for reproducibility

Using the wrapper class is necessary for easy training in the crystal sets because of the data loader function, which creates pytorch-geometric data loaders that are easy to iterate over.
Crystal data sets are augmented when getting the data loaders and augmented CIF files are stored next to the originals.
Loading original or augmented CIF files is handled automatically by the loaders, where only the training set uses the augmented files.
Note: this function call may take a while the first time due to loading, transforming, and saving thousands of files. However, if augmented files are present, they will not be augmented again, and running the code again will be significantly faster.

After loading our data, our `dataset` object has additional information from the parent class, `CrystalDataset` that may be useful to look at. We can look at the CIF ids and their correspongind label:

```python
>>> print(dataset.id_prop_augment)
[['0' '0.099463']
 ['1' '-0.63467']
 ['2' '-0.725799']
 ...
 ['4189' '-1.004029']
 ['4190' '-3.7094099999999997']
 ['4191' '-3.6372910000000003']]
```

as well as the updated data path:

```python
>>> print(dataset.data_path)
./data_download/lanths
```

### Data Splitting

```python
train_loader, valid_loader, test_loader = dataset.get_data_loaders(
                                                     target=None,
                                                     transform=transforms,
                                                     fold=None
)
```

`CsytalDatasetWrapper.get_data_loaders()` argument:
- `target` (str, optional, default=None): The target label for training. Currently all
                            crystal datasets are single-target, and so this parameter
                            is truly optional.
- `transform` (AbstractTransformation, optional, default=None): The data transformation
                            we will use for data augmentation.
- `fold` (int, optiona, default=None): Which of k folds to use for training. Will
                            throw an error if specified and k-fold CV is not
                            done in the class instantiaion. This overrides
                            valid_size and test_size

Returns:                           
- `train/valid/test_loader` (DataLoader): Data loaders containing the train, validation
                                        and test splits of our data.

Now that our data is ready for training and evalutaion, we initialize our model.
Task, either regression or classification needs to be passed in.
Our dataset object stores this in the `task` attribute.
Note: CGCNN uses a different implementation and needs to be initialized and trained differently.
This is covered at the end of this guide and a working example of this is given in `examples/`.

After splitting the data, we can see which CIF files are in each data loader.
Augmented files only appear in the training set:

```python
>>> print(train_loader.dataset.id_prop_augment)
[['1762' '-3.933583']
 ['1762_perturbed' '-3.933583']
 ['1762_rotated' '-3.933583']
 ...
 ['826_swapaxes' '-1.640507']
 ['826_translate' '-1.640507']
 ['826_supercell' '-1.640507']]
```

```python
>>> print(valid_loader.dataset.id_prop_augment)
[['2183' '-0.266351']
 ['3898' '0.542863']
 ['167' '-0.22298400000000002']
 ...
 ['1660' '-3.824347']
 ['644' '-2.436586']
 ['2777' '0.173244']]
```

```python
>>> print(test_set.dataset.id_prop_augment)
[['778' '-0.403455']
 ['1381' '-3.686927']
 ['3728' '-0.459214']
 ...
 ['699' '-3.096345']
 ['528' '-2.411031']
 ['2355' '-0.660257']]
```


### Model Initialization

```python
model = SchNet(
          hidden_channels=128,
          num_filters=128,
          num_interactions=6,
          num_gaussians=50,
          cutoff=10.0,
          max_num_neighbors=32,
          readout='add',
          dipole=False,
          mean=None,
          std=None,
          atomref=None,
)
```

`SchNet` arguments:

- `hidden_channels` (int, optional): Hidden embedding size.
    (default: :obj:`128`)
- `num_filters` (int, optional): The number of filters to use.
    (default: :obj:`128`)
- `num_interactions` (int, optional): The number of interaction blocks.
    (default: :obj:`6`)
- `num_gaussians` (int, optional): The number of gaussians :math:`\mu`.
    (default: :obj:`50`)
- `cutoff` (float, optional): Cutoff distance for interatomic interactions.
    (default: :obj:`10.0`)
- `max_num_neighbors` (int, optional): The maximum number of neighbors to
    collect for each node within the :attr:`cutoff` distance.
    (default: :obj:`32`)
- `readout` (string, optional): Whether to apply :obj:`"add"` or
    :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
- `dipole` (bool, optional): If set to :obj:`True`, will use the magnitude
    of the dipole moment to make the final prediction, *e.g.*, for
    target 0 of :class:`torch_geometric.datasets.QM9`.
    (default: :obj:`False`)
- `mean` (float, optional): The mean of the property to predict.
    (default: :obj:`None`)
- `std` (float, optional): The standard deviation of the property to
    predict. (default: :obj:`None`)
- `atomref` (torch.Tensor, optional): The reference of single-atom
    properties.
    Expects a vector of shape :obj:`(max_atomic_number, )`.

Our SchNet implementation comes from [here](http://www.quantum-machine.org/datasets/trained_schnet_models.zip)

```python
model = GINet(
          num_layer=5,
          emb_dim=256,
          feat_dim=512,
          drop_ratio=0,
          pool='mean'
)
```

`GINet` arguments:
- `num_layer` (int): the number of GNN layers
- `emb_dim` (int): dimensionality of embeddings
- `feat_dim` (int): dimensionality of feature
- `drop_ratio` (float): dropout rate
- `pool` (str): One of 'mean', 'add', 'max'


### Training

From here, we are ready to train using standard PyTorch training procedure.

```python
import torch
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

Now we have our training loop. Pymatgen throws a lot of warnings, which we suppress.

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for epoch in range(100):
        for bn, data in tqdm(enumerate(train_loader)):        
            optimizer.zero_grad()

            # Comment out the following line and uncomment the line after for cuda
            pred = model(data)
            #pred = model(data.cuda())
            
            loss = criterion(pred, data.y)

            loss.backward()
            optimizer.step()
```

Evaluation requires storing all predections and labels for each batch, and so we have:

```python
from sklearn.metrics import mean_absolute_error 

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with torch.no_grad():
        model.eval()
        preds = torch.Tensor([])
        targets = torch.Tensor([])
        for data in test_loader:
            pred = model(data)
            preds = torch.cat((preds, pred.cpu()))
            targets = torch.cat((targets, data.y.cpu()))

        mae = mean_absolute_error(preds, targets)   
    
    set_str = "VALIDATION" if(validation) else "TEST"
    print("{0} MAE: {1:.3f}".format(set_str, mae))
```

### K-Fold Cross Validation

`CrystalDatasetWrapper` additionally supports automatic k-fold cross validation with few changes to the code.
We first specify the number of folds we would like when initializing the dataset:

```python
dataset = CrystalDatasetWrapper("lanthanides", kfolds=5, batch_size=128,
                                valid_size=0.1, test_size=0.1)
```

From there, we simply specify which fold we would like when splitting the data:

```python
train_loader, valid_loader, test_loader = dataset.get_data_loaders(transform=transform, fold=0)
```

Note: This takes longer to split because every CIF file is augmented at this point.
Augmenting all CIF files now speeds up obtaining the other folds later.

From here, we are ready to initialize our model, train, and evaluate as before:

```python
model = GINet(
)

import torch
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for epoch in range(100):
        for bn, data in tqdm(enumerate(train_loader)):        
            optimizer.zero_grad()

            # Comment out the following line and uncomment the line after for cuda
            pred = model(data)
            
            loss = criterion(pred, data.y)

            loss.backward()
```

Evalutation is also done in the same way:

```python
from sklearn.metrics import mean_absolute_error 

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with torch.no_grad():
        model.eval()
        preds = torch.Tensor([])
        targets = torch.Tensor([])
        for data in test_loader:
            pred = model(data)
            preds = torch.cat((preds, pred.cpu()))
            targets = torch.cat((targets, data.y.cpu()))

        mae = mean_absolute_error(preds, targets)   
    
    set_str = "VALIDATION" if(validation) else "TEST"
```

Obtaining the next fold is as easy as splitting the data again, with a different fold number passed in:


```python
train_loader, valid_loader, test_loader = dataset.get_data_loaders(transform=transform, fold=1)
```

Training and evalutation are then done in the same way as before.

### Training with CUDA

AugLiChem takes advantage of PyTorch’s CUDA support to leverage GPUs for faster training and evaluation. To initialize a model on our GPU, we call the .cuda() function.

```python
model = SchNet(
)
model.cuda()
```

Our training setup is the same as before:

```python
import torch
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

The only difference in our training loop is putting our data on the GPU as we train:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for epoch in range(100):
        for bn, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

	    # data -> GPU
            pred = model(data.cuda())

            loss = criterion(pred, data.y)

            loss.backward()
            optimizer.step()
```

Which we also do for evaluation:

```python
from sklearn.metrics import mean_absolute_error

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with torch.no_grad():
        model.eval()
        preds = torch.Tensor([])
        targets = torch.Tensor([])
        for data in test_loader:
      
            # data -> GPU
            pred = model(data.cuda())
            preds = torch.cat((preds, pred.cpu()))
            targets = torch.cat((targets, data.y.cpu()))
            
        mae = mean_absolute_error(preds, targets)
        
    set_str = "VALIDATION" if(validation) else "TEST"
``` 

### Crystal Graph Convolutional Network

The built in CGCNN implementation is based on an older version of pytorch-geometric and requires slightly different model initialization, training, and data handling.
The data handling is done automatically with AugLiChem when the `cgcnn=True` flag is passed in when initializing a `CrystalDatasetWrapper` object.

Our setup and transformations are the same as before.
Initializing CGCNN requires information about our data:

```python
structures, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

model = CrystalGraphConvNet(
                orig_atom_fea_len=orig_atom_fea,
                nbr_fea_len=nbr_fea_len,
                atom_fea_len=64,
                n_conv=3,
                h_efa_len=128,
                n_h=1,
)
```

`CrystalGraphConvNet` arguments:
- `orig_atom_fea_len` (int): Number of atom features in the input.
- `nbr_fea_len` (int): Number of bond features.
- `atom_fea_len` (int, optional, default=64): Number of hidden atom features in the convolutional layers
- `n_conv` (int, optional, default=3): Number of convolutional layers
- `h_fea_len` (int, optional, default=128): Number of hidden features after pooling
- `n_h` (int, optional, default=1): Number of hidden layers after pooling

Our training setup is the same as before:

```python
import torch
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

In training, we build our data object from the data loader output

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for epoch in range(1):
        for bn, (data, target, _) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            
            input_var = (Variable(data[0]),
                         Variable(data[1]),
                         data[2],
                         data[3])

            pred = model(*input_var)
            loss = criterion(pred, target)
            #loss = criterion(pred, target.cuda())

            loss.backward()
            optimizer.step()
```

Which we also do for evaluation:

```python
with torch.no_grad():
    model.eval()
    preds = torch.Tensor([])
    targets = torch.Tensor([])
    for data, target, _ in test_loader:
        input_var = (Variable(data[0]),
                     Variable(data[1]),
                     data[2],
                     data[3])
        
        pred = model(*input_var)
        
        preds = torch.cat((preds, pred.cpu().detach()))
        targets = torch.cat((targets, target))
        
    mae = mean_absolute_error(preds, targets)   
set_str = "VALIDATION" if(validation) else "TEST"
print("TEST MAE: {0:.3f}".format(mae))
```

### Crystal Graph Convolutional Network with CUDA

AugLiChem takes advantage of PyTorch’s CUDA support to leverage GPUs for faster training and evaluation. To initialize a model on our GPU, we call the .cuda() function.

```python
structures, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len)
model.cuda()
```

In training, we build our data object from the data loader output as before:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for epoch in range(1):
        for bn, (data, target, _) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            input_var = (Variable(data[0].cuda()),
                         Variable(data[1].cuda()),
                         data[2].cuda(),
                         data[3])

            pred = model(*input_var)
            loss = criterion(pred, target.cuda())

            loss.backward()
            optimizer.step()
```

Which we also do for evaluation:

```python
with torch.no_grad():
    model.eval()
    preds = torch.Tensor([])
    targets = torch.Tensor([])
    for data, target, _ in test_loader:
    
        input_var = (Variable(data[0].cuda()),
                     Variable(data[1].cuda()),
                     data[2].cuda(),
                     data[3])

        pred = model(*input_var)
    
        preds = torch.cat((preds, pred.cpu().detach()))
        targets = torch.cat((targets, target))
    
    mae = mean_absolute_error(preds, targets)   
set_str = "VALIDATION" if(validation) else "TEST"
print("TEST MAE: {0:.3f}".format(mae))
```
