#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import Any, Dict, List, Optional

from pymatgen.transformations.transformation_abc import AbstractTransformation

from auglichem.crystal.data._crystal_dataset import CrystalDataset
from auglichem.crystal._transforms import (
        RandomRotationTransformation,
        RandomPerturbStructureTransformation,
        RandomRemoveSitesTransformation,
        SupercellTransformation,
        RandomTranslateSitesTransformation,
        CubicSupercellTransformation,
        PrimitiveCellTransformation
)

"""
Composition Operators:
Compose: the Compose operator was added here such that users
do not have to import `torchvision` in order to compose multiple
augmentations together. These operators work identically and either
can be used.
OneOf: the OneOf operator takes as input a list of transforms and
may apply (with probability p) one of the transforms in the list.
If a transform is applied, it is selected using the specified
probabilities of the individual transforms.
Example:
 >>> Compose([
 >>>     Blur(),
 >>>     ColorJitter(saturation_factor=1.5),
 >>>     OneOf([
 >>>         OverlayOntoScreenshot(),
 >>>         OverlayEmoji(),
 >>>         OverlayText(),
 >>>     ]),
 >>> ])
"""


class BaseComposition(object):
    def __init__(self, transforms: List[AbstractTransformation], p: float = 1.0):
        """
        @param transforms: a list of transforms
        @param p: the probability of the transform being applied; default value is 1.0
        """
        for transform in transforms:
            assert isinstance(
                transform, (AbstractTransformation, BaseComposition)
            ), "Expected instances of type `AbstractTransformation` or `BaseComposition` for variable `transforms`"  # noqa: B950
        assert 0 <= p <= 1.0, "p must be a value in the range [0, 1]"

        self.transforms = transforms
        self.p = p


class Compose(BaseComposition):
    def __call__(self, crystal: CrystalDataset, seed: Optional[int] = None) -> CrystalDataset:
        """
        Applies the list of transforms in order to the crystal
        @param crystal: PIL Image to be augmented
        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned
        @returns: Augmented PIL Image
        """
        if random.random() > self.p:
            return crystal

        for transform in self.transforms:
            crystal = transform(crystal, seed=seed)

        return crystal


class OneOf(BaseComposition):
    def __init__(self, transforms: List[AbstractTransformation], p: float = 1.0):
        """
        @param transforms: a list of transforms to select from; one of which
            will be chosen to be applied to the media
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(transforms, p)
        transform_probs = [t.p for t in transforms]
        probs_sum = sum(transform_probs)
        self.transform_probs = [t / probs_sum for t in transform_probs]

    def __call__(self, crystal: CrystalDataset, seed: Optional[int] = None) -> CrystalDataset:
        """
        Applies one of the transforms to the crystal
        @param crystal: PIL Image to be augmented
        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned
        @returns: Augmented PIL Image
        """
        if random.random() > self.p:
            return crystal

        transform = random.choices(self.transforms, self.transform_probs)[0]
        return transform(crystal, force=True, metadata=metadata)
