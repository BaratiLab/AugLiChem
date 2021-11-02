---
layout: page
title: Crystal Usage
permalink: /crystal_usage/
---

## Crystal Usage

Using the crystal submodule is very similar to molecule.

The first step is to import the relevant modules.
AugLiChem is largely self-contained, and so we import transformations, data wrapper, and models.

```python
from auglichem.crystal import RotationTransformation, PerturbStructureTransformation
from auglichem.crystal.data import CrystalDatasetWrapper
from auglichem.crystal.models import SchNet
```

Next, we set up our transformations.
Transformations can be set up as a list or single transformation.
When using a list, each molecule is transformed by all transformations passed in.

```python
transforms = [
	RotationTransformation(),
	PerturbStructureTransformation()
]
```
RandomAtomMask and RandomBondDelete take in either a list of two number, or a single number, which represents the fraction of atoms to mask and bonds to delete, respectively.
When a list is passed in, a number is sampled uniformly between the two values for each molecule and used for the masking/deletion fraction.
MotifRemoval is deterministic, and uses a similarity score between motifs and the original molecule.
Motifs that are above the similarity score threshold, the passed in parameter, are retained for the augmented molecule data.

After initializing our transformations, we are ready to initialize our data set.
Data sets are selected with a string, and are automatically downloaded to `./data_download` by default.
This directory is created if it is not present, and does not download the data again if it is already present.
Batch size, validation size, and test size for training and evaluation are set here.
Unlike molecule, CrystalDatasetWrapper handles transformations when splitting the data.
Random splitting, and k-fold cross validation are supported.

```python
dataset = CrystalDatasetWrapper("lanthanides", batch_size=128, valid_size=0.1, test_size=0.1)
```

Using the wrapper class is necessary for easy training in the crystal sets because of the data loader function, which creates pytorch-geometric data loaders that are easy to iterate over.
Crystal data sets are augmented when getting the data loaders and augmented CIF files are stored next to the originals.
Loading original or augmented CIF files is handled automatically by the loaders, where only the training set uses the augmented files.
Note: this function call may take a while the first time due to loading, transforming, and saving thousands of files. However, if augmented files are present, they will not be augmented again, and running the code again will be significantly faster.

```python
train_loader, valid_loader, test_loader = dataset.get_data_loaders(transform=transforms)
```

Now that our data is ready for training and evalutaion, we initialize our model.
Task, either regression or classification needs to be passed in.
Our dataset object stores this in the `task` attribute.
Note: CGCNN uses a different implementation and needs to be initialized and trained differently.
A working example of this is given in `examples/`.

```python
model = SchNet(task=dataset.task)
```

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
    for epoch in range(1):
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
            #pred = model(data)
            pred = model(data.cuda())
            preds = torch.cat((preds, pred.cpu()))
            targets = torch.cat((targets, data.y.cpu()))

        mae = mean_absolute_error(preds, targets)   
    
    set_str = "VALIDATION" if(validation) else "TEST"
    print("{0} MAE: {1:.3f}".format(set_str, mae))
```
