---
layout: page
title: Crystal
permalink: /crystal/
---

Crystals are loaded from CIF files, and the augmentations given can be stored as CIF files.
We take advantage of this fact and save augmented files next to original files.
This has the added benefit of allowing easy inspection of augmentations in their native format.

## Crystal Representation on Graphs

Give example here.

## Data Augmentation

![crystal](../images/original_crystal.png)

Five augmentations are supported for crystals.
We take the original structure seen above

![perturb](../images/perturbation.png)

In the random perturbation augmentation all the sites in the crystalline systemare randomly perturbed by a distance between0 to 1 A. This augmentationis especially useful in breaking the symmetries that exist between the sites inthe crystals.
This transformation can be imported and initilized as follows:

```
from auglichem.crystal import PerturbStructureTransformation
transform = PerturbStructureTransformation()
```

![perturb](../images/rotation.png)

In the random rotation transform, we randomly rotate the sites in the crystalbetween-90 to 90degrees. To generate a strong augmentation we initiallyuse  the  random  perturbation  augmentation  to  generate  the  initial  strcuturewhich is then rotated randomly.
This transformation can be imported and initilized as follows:

```
from auglichem.crystal import RotationTransformation
transform = RotationTransformation()
```

![perturb](../images/swap_axes.png)

In the swap axes transformation of the crystal, we swap the coordinates of thesites in the crystal between two axes. For example, we may swap the x and they axes coordinates or the y and z axes coordinates. The swap axes transformgreatly displaces the locations of all the sites in the crystal.
This transformation can be imported and initilized as follows:

```
from auglichem.crystal import SwapAxesTransformation
transform = SwapAxesTransformation()
```

![perturb](../images/translation.png)

The random translate transform randomly displaces a site in the crystal bya distance between0 to 1 A. In this work, we randomly select 25% of thesites in the crystal and displace them. This creates an augmentation different from the random perturb augmentation as not all the sites in the crystal aredisplaced.
This transformation can be imported and initilized as follows:

```
from auglichem.crystal import TranslateSitesTransformation
transform = TranslateSitesTransformation()
```

![perturb](../images/supercell.png)

The supercell transformation produces a supercell of the crystalline system.The distinct feature of the supercell of the crystal is that after transformationthe supercell represents the same crystal with a larger volume. There exists alinear mapping between the basis vectors of crystal and the basis vectors ofthe supercell.
This transformation can be imported and initilized as follows:

```
from auglichem.crystal import SupercellTransformation
transform = SupercellTransformation()
```


For a more detailed guide on using these augmentations, please read the [usage guide](../usage_guide).

## Data sets

The crystalline data sets cover multiple different types of materials, as well as predicting different properties.
The data sets currently supported are: Lanthanides, Perovskites, Formation Energy, Band Gap, and Fermi Energy prediction.


## Models

In addition to data sets, AugLiChem has popular models implemented and ready to use with our graph data. CGCNN, GIN, and SchNet are all supported, and readers are referred to the respective papers for the model details.
