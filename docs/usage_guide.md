---
layout: page
title: Usage
permalink: /usage_guide/
---

AugLiChem has been designed from the ground up with ease-of-use in mind.
Fully functional notebooks are available at our github in the `examples/` directory [here](https://github.com/BaratiLab/AugLiChem/tree/main/examples).
In-depth documentation of each function is given in the docstrings and can be printed out using python's built-in `help()` function.
Using PyTorch's CUDA support, all models and data sets can be used with GPUs.

## Molecule Usage

The first step is to import the relevant modules.
AugLiChem is largely self-contained, and so we import transformations, data wrapper, and models.

```
from auglichem.molecule import RandomAtomMask, RandomBondDelete, MotifRemoval
from auglichem.molecule.data import MoleculeDatasetWrapper
from auglichem.molecule.models import AttentiveFP as afp
```

Next, we set up our transformations.
Transformations can be set up as a list or single transformation.
When using a list, each molecule is transformed by all transformations passed in.

```
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

After initializing our transformations, we are ready to initialize our data set.
Data sets are selected with a string, and are automatically downloaded to `./data_download` by default.
This directory is created if it is not present, and does not download the data again if it is already present.
Batch size, validation size, and test size for training and evaluation are set here.
The transforms are passed in here and scaffold splitting is supported.

```
dataset = MoleculeDatasetWrapper("ClinTox", transform=transforms, batch_size=128, valid_size=0.1, test_size=0.1)
```

Using the wrapper class is preferred for easy training because of the data loader function, which creates pytorch-geometric data loaders that are easy to iterate over.
With multi-target data sets, such as ClinTox, we specify the target we want here.
If no target is selected, the first target in the downloaded data file is used.
Multiple targets can be selected for multi-target training by passing in a list of targets, or 'all' to use all of them.

```
train_loader, valid_loader, test_loader = dataset.get_data_loaders("FDA_APPROVED")
```

Now that our data is ready for training and evalutaion, we initialize our model.
Task, either regression or classification needs to be passed in.
Our dataset object stores this in the `task` attribute.

```
model = afp(task=dataset.task)
```

From here, we are ready to train using standard PyTorch training procedure.

```
import torch
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

Now we have our training loop.

```
for epoch in range(100):
    for bn, data in tqdm(enumerate(train_loader)):

        optimizer.zero_grad()
        
        _, pred = model(data)
        loss = criterion(pred, data.y.flatten())

        loss.backward()
        optimizer.step()
```

Evaluation requires storing all predections and labels for each batch, and so we have

```
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

## Crystal Usage

Using the crystal submodule is very similar to molecule.

The first step is to import the relevant modules.
AugLiChem is largely self-contained, and so we import transformations, data wrapper, and models.

```
from auglichem.crystal import RotationTransformation, PerturbStructureTransformation
from auglichem.crystal.data import CrystalDatasetWrapper
from auglichem.crystal.models import SchNet
```

Next, we set up our transformations.
Transformations can be set up as a list or single transformation.
When using a list, each molecule is transformed by all transformations passed in.

```
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

```
dataset = CrystalDatasetWrapper("lanthanides", batch_size=128, valid_size=0.1, test_size=0.1)
```

Using the wrapper class is necessary for easy training in the crystal sets because of the data loader function, which creates pytorch-geometric data loaders that are easy to iterate over.
Crystal data sets are augmented when getting the data loaders and augmented CIF files are stored next to the originals.
Loading original or augmented CIF files is handled automatically by the loaders, where only the training set uses the augmented files.
Note: this function call may take a while the first time due to loading, transforming, and saving thousands of files. However, if augmented files are present, they will not be augmented again, and running the code again will be significantly faster.

```
train_loader, valid_loader, test_loader = dataset.get_data_loaders(transform=transforms)
```

Now that our data is ready for training and evalutaion, we initialize our model.
Task, either regression or classification needs to be passed in.
Our dataset object stores this in the `task` attribute.
Note: CGCNN uses a different implementation and needs to be initialized and trained differently.
A working example of this is given in `examples/`.

```
model = SchNet(task=dataset.task)
```

From here, we are ready to train using standard PyTorch training procedure.

```
import torch
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

Now we have our training loop. Pymatgen throws a lot of warnings, which we suppress.

```
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

```
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
