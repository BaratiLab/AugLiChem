---
layout: page
title: Molecule
permalink: /molecule/
---

Molecular augmentation is done at call-time. Graphs are built from rdkit Chem objects.

## Molecular Representation on Graphs

Give example here.

## Data Augmentation

Random atom masking replaces randomly selected node attributes with a mask token.
The fraction of atoms masked is passed in as an argument, and at least one atom is masked regardless of the fraction selected.
In the graph representation, the atoms are still there, but masked atoms are all represented by the same token.
This transformation can be imported and initilized as follows:

```
from auglichem.molecule import RandomAtomMask
transform = RandomAtomMask(p=0.25)
```

Random  bond deletion removes bonds from our graph representation.

```
from auglichem.molecule import RandomBondDelete
transform = RandomBondDelete(p=0.25)
```


Motif removal is different from the other augmentations in that it is deterministic.
A similarity score is calculated between motifs and the molecule, where motifs above the threshold are retained.
This threshold can be set by passing in an argument.

```
from auglichem.molecule import MotifRemoval
transform = MotifRemoval(0.6)
```

For a more detailed guide on using these augmentations, please read the usage guide [NEED LINK].

## Data Sets

Datasets are downloaded automatically

## Models

AttentiveFP
