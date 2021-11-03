from __future__ import print_function, division

import sys
import csv
import functools
import json
import os
import random
import ase
from ase.io import read, write
import warnings
import numpy as np
from pymatgen.core.structure import Structure
import glob
from ase.io import  read, write
from ase import Atoms,Atom
from pymatgen.core.operations import SymmOp
from pymatgen.transformations.transformation_abc import AbstractTransformation
import math
from typing import Optional, Union
from pymatgen.core.structure import Molecule, Structure


class AseAtomsAdaptor:
    """
    Adaptor serves as a bridge between ASE Atoms and pymatgen objects.
    """

    @staticmethod
    def get_atoms(structure, **kwargs):
        """
        Returns ASE Atoms object from pymatgen structure or molecule.
        Args:
            structure: pymatgen.core.structure.Structure or pymatgen.core.structure.Molecule
            **kwargs: other keyword args to pass into the ASE Atoms constructor
        Returns:
            ASE Atoms object
        """
        if not structure.is_ordered:
            raise ValueError("ASE Atoms only supports ordered structures")
        #if not ase_loaded:
        if('ase' not in sys.modules):
            raise ImportError(
                "AseAtomsAdaptor requires ase package.\n" "Use `pip install ase` or `conda install ase -c conda-forge`"
            )
        symbols = [str(site.specie.symbol) for site in structure]
        positions = [site.coords for site in structure]
        if hasattr(structure, "lattice"):
            cell = structure.lattice.matrix
            pbc = True
        else:
            cell = None
            pbc = None
        return Atoms(symbols=symbols, positions=positions, pbc=pbc, cell=cell, **kwargs)

    @staticmethod
    def get_structure(atoms, cls=None):
        """
        Returns pymatgen structure from ASE Atoms.
        Args:
            atoms: ASE Atoms object
            cls: The Structure class to instantiate (defaults to pymatgen structure)
        Returns:
            Equivalent pymatgen.core.structure.Structure
        """
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        lattice = atoms.get_cell()

        cls = Structure if cls is None else cls
        return cls(lattice, symbols, positions, coords_are_cartesian=True)


class RotationTransformation(AbstractTransformation):
    """
    The RotationTransformation applies a rotation to a structure.
    """

    def __init__(self, axis=None, angle=None):
        """
        Args:
            axis (3x1 array): Axis of rotation, e.g., [1, 0, 0]
            angle (float): Angle to rotate
        """
        self.axis = axis
        self.angle = angle

    def apply_transformation(self, structure, axis=None, angle=None, angle_in_radians=False,
                             seed=None):
        """
        Apply the transformation.
        Args:
            structure (Structure): Input Structure
        Returns:
            Rotated Structure.
        """
        if(axis is not None):
            self.axis = axis
        if(angle is not None):
            self.angle = angle

        self.angle_in_radians = angle_in_radians
        self._symmop = SymmOp.from_axis_angle_and_translation(self.axis, self.angle, self.angle_in_radians)

        s = structure.copy()
        s.apply_operation(self._symmop, fractional=True)
        # s = self._symmop(s)
        return s

    def __str__(self):
        return "Rotation Transformation about axis " + "{} with angle = {:.4f} {}".format(
            self.axis, self.angle, "radians" if self.angle_in_radians else "degrees"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """
        Returns:
            Inverse Transformation.
        """
        return RotationTransformation(self.axis, -self.angle, self.angle_in_radians)

    @property
    def is_one_to_many(self):
        """Returns: False"""
        return False


class PerturbStructureTransformation(AbstractTransformation):
    """
    This transformation perturbs a structure by a specified distance in random
    directions. Used for breaking symmetries.
    """

    def __init__(
        self,
        distance: float = 0.01,
        min_distance: Optional[Union[int, float]] = None,
    ):
        """
        Args:
            distance: Distance of perturbation in angstroms. All sites
                will be perturbed by exactly that distance in a random
                direction.
            min_distance: if None, all displacements will be equidistant. If int
                or float, perturb each site a distance drawn from the uniform
                distribution between 'min_distance' and 'distance'.
        """
        self.distance = distance
        self.min_distance = min_distance

    def apply_transformation(self, structure: Structure, seed=None) -> Structure:
        """
        Apply the transformation.
        Args:
            structure: Input Structure
        Returns:
            Structure with sites perturbed.
        """
        s = structure.copy()
        if(seed is not None):
            np.random.seed(seed) # Pymatgen uses numpy random
        s.perturb(self.distance, min_distance=self.min_distance)
        return s

    def __str__(self):
        return "PerturbStructureTransformation : " + "Min_distance = {}".format(self.min_distance)

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """
        Returns: None
        """
        return None

    @property
    def is_one_to_many(self):
        """
        Returns: False
        """
        return False


class SwapAxesTransformation(object):
    """
    Swap coordinate axes in the crystal strucutre.
    """
    def __init__(self):
        #TODO: Shouldn't this always call the transformation?
        #TODO: Decide if its probabilistic or not, and if we should use random seeding.
        pass

    def apply_transformation(self, crys, seed=None, _test_choice=None):
        atoms = AseAtomsAdaptor().get_atoms(crys)

        if(seed is not None):
            np.random.seed(seed)
        choice = np.random.choice(3, 2, replace=False)
        if(_test_choice is not None):
            choice = _test_choice
        pos = (atoms.positions)
        pos[:,[choice[0], choice[1]]] = pos[:,[choice[1], choice[0]]]
        atoms.arrays["positions"] = pos

        return AseAtomsAdaptor.get_structure(atoms)

    def __str__(self):
        return "SwapAxesTransformation"

    def __repr__(self):
        return self.__str__()

class RemoveSitesTransformation(AbstractTransformation):
    """
    Remove certain sites in a structure.
    """

    def __init__(self, indices_to_remove):
        """
        Args:
            indices_to_remove: List of indices to remove. E.g., [0, 1, 2]
        """

        self.indices_to_remove = indices_to_remove

    def apply_transformation(self, structure, seed=None):
        """
        Apply the transformation.
        Arg:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.
        Return:
            Returns a copy of structure with sites removed.
        """
        s = structure.copy()
        s.remove_sites(self.indices_to_remove)
        return s

    def __str__(self):
        return "RemoveSitesTransformation :" + ", ".join(map(str, self.indices_to_remove))

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """Return: None"""
        return None

    @property
    def is_one_to_many(self):
        """Return: False"""
        return False


def _proj(b, a):
    """
    Returns vector projection (np.ndarray) of vector b (np.ndarray)
    onto vector a (np.ndarray)
    """
    return (b.T @ (a / np.linalg.norm(a))) * (a / np.linalg.norm(a))

def _round_and_make_arr_singular(arr: np.ndarray) -> np.ndarray:
    """
    This function rounds all elements of a matrix to the nearest integer,
    unless the rounding scheme causes the matrix to be singular, in which
    case elements of zero rows or columns in the rounded matrix with the
    largest absolute valued magnitude in the unrounded matrix will be
    rounded to the next integer away from zero rather than to the
    nearest integer.
    The transformation is as follows. First, all entries in 'arr' will be
    rounded to the nearest integer to yield 'arr_rounded'. If 'arr_rounded'
    has any zero rows, then one element in each zero row of 'arr_rounded'
    corresponding to the element in 'arr' of that row with the largest
    absolute valued magnitude will be rounded to the next integer away from
    zero (see the '_round_away_from_zero(x)' function) rather than the
    nearest integer. This process is then repeated for zero columns. Also
    note that if 'arr' already has zero rows or columns, then this function
    will not change those rows/columns.
    Args:
        arr: Input matrix
    Returns:
        Transformed matrix.
    """

    def round_away_from_zero(x):
        """
        Returns 'x' rounded to the next integer away from 0.
        If 'x' is zero, then returns zero.
        E.g. -1.2 rounds to -2.0. 1.2 rounds to 2.0.
        """
        abs_x = abs(x)
        return math.ceil(abs_x) * (abs_x / x) if x != 0 else 0

    arr_rounded = np.around(arr)

    # Zero rows in 'arr_rounded' make the array singular, so force zero rows to
    # be nonzero
    if (~arr_rounded.any(axis=1)).any():
        # Check for zero rows in T_rounded

        # indices of zero rows
        zero_row_idxs = np.where(~arr_rounded.any(axis=1))[0]

        for zero_row_idx in zero_row_idxs:  # loop over zero rows
            zero_row = arr[zero_row_idx, :]

            # Find the element of the zero row with the largest absolute
            # magnitude in the original (non-rounded) array (i.e. 'arr')
            matches = np.absolute(zero_row) == np.amax(np.absolute(zero_row))
            col_idx_to_fix = np.where(matches)[0]

            # Break ties for the largest absolute magnitude
            r_idx = np.random.randint(len(col_idx_to_fix))
            col_idx_to_fix = col_idx_to_fix[r_idx]

            # Round the chosen element away from zero
            arr_rounded[zero_row_idx, col_idx_to_fix] = round_away_from_zero(arr[zero_row_idx, col_idx_to_fix])

    # Repeat process for zero columns
    if (~arr_rounded.any(axis=0)).any():

        # Check for zero columns in T_rounded
        zero_col_idxs = np.where(~arr_rounded.any(axis=0))[0]
        for zero_col_idx in zero_col_idxs:
            zero_col = arr[:, zero_col_idx]
            matches = np.absolute(zero_col) == np.amax(np.absolute(zero_col))
            row_idx_to_fix = np.where(matches)[0]

            for i in row_idx_to_fix:
                arr_rounded[i, zero_col_idx] = round_away_from_zero(arr[i, zero_col_idx])
    return arr_rounded.astype(int)

class SupercellTransformation(AbstractTransformation):
    """
    The RotationTransformation applies a rotation to a structure.
    """

    def __init__(self, scaling_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        """
        Args:
            scaling_matrix: A matrix of transforming the lattice vectors.
                Defaults to the identity matrix. Has to be all integers. e.g.,
                [[2,1,0],[0,3,0],[0,0,1]] generates a new structure with
                lattice vectors a" = 2a + b, b" = 3b, c" = c where a, b, and c
                are the lattice vectors of the original structure.
        """
        self.scaling_matrix = scaling_matrix

    @staticmethod
    def from_scaling_factors(scale_a=1, scale_b=1, scale_c=1):
        """
        Convenience method to get a SupercellTransformation from a simple
        series of three numbers for scaling each lattice vector. Equivalent to
        calling the normal with [[scale_a, 0, 0], [0, scale_b, 0],
        [0, 0, scale_c]]
        Args:
            scale_a: Scaling factor for lattice direction a. Defaults to 1.
            scale_b: Scaling factor for lattice direction b. Defaults to 1.
            scale_c: Scaling factor for lattice direction c. Defaults to 1.
        Returns:
            SupercellTransformation.
        """
        return SupercellTransformation([[scale_a, 0, 0], [0, scale_b, 0], [0, 0, scale_c]])

    def apply_transformation(self, structure, seed=None):
        """
        Apply the transformation.
        Args:
            structure (Structure): Input Structure
        Returns:
            Supercell Structure.
        """
        return structure * self.scaling_matrix

    def __str__(self):
        return "Supercell Transformation with scaling matrix " + "{}".format(self.scaling_matrix)

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """
        Raises: NotImplementedError
        """
        raise NotImplementedError()

    @property
    def is_one_to_many(self):
        """
        Returns: False
        """
        return False

class TranslateSitesTransformation(AbstractTransformation):
    """
    This class translates a set of sites by a certain vector.
    """

    def __init__(self, indices_to_move, translation_vector, vector_in_frac_coords=True):
        """
        Args:
            indices_to_move: The indices of the sites to move
            translation_vector: Vector to move the sites. If a list of list or numpy
                array of shape, (len(indices_to_move), 3), is provided then each
                translation vector is applied to the corresponding site in the
                indices_to_move.
            vector_in_frac_coords: Set to True if the translation vector is in
                fractional coordinates, and False if it is in cartesian
                coordinations. Defaults to True.
        """
        self.indices_to_move = indices_to_move
        self.translation_vector = np.array(translation_vector)
        self.vector_in_frac_coords = vector_in_frac_coords


    def apply_transformation(self, 
        structure, indices_to_move=None, translation_vector=None, 
        vector_in_frac_coords=None, seed=None
    ):
        """
        Apply the transformation.
        Arg:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.
        Return:
            Returns a copy of structure with sites translated.
        """
        if(indices_to_move is not None):
            self.indices_to_move = indices_to_move
        if(translation_vector is not None):
            self.translation_vector = np.array(translation_vector)
        if(vector_in_frac_coords is not None):
            self.vector_in_frac_coords = vector_in_frac_coords

        s = structure.copy()
        if self.translation_vector.shape == (len(self.indices_to_move), 3):
            for i, idx in enumerate(self.indices_to_move):
                s.translate_sites(idx, self.translation_vector[i], self.vector_in_frac_coords)
        else:
            s.translate_sites(
                self.indices_to_move,
                self.translation_vector,
                self.vector_in_frac_coords,
            )
        return s

    def __str__(self):
        return "TranslateSitesTransformation for indices " + "{}, vect {} and vect_in_frac_coords = {}".format(
            self.indices_to_move,
            self.translation_vector,
            self.vector_in_frac_coords,
        )

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """
        Returns:
            TranslateSitesTranformation with the reverse translation.
        """
        return TranslateSitesTransformation(self.indices_to_move, -self.translation_vector, self.vector_in_frac_coords)

    @property
    def is_one_to_many(self):
        """Return: False"""
        return False

    def as_dict(self):
        """
        Json-serializable dict representation.
        """
        d = MSONable.as_dict(self)
        d["translation_vector"] = self.translation_vector.tolist()
        return d

class CubicSupercellTransformation(AbstractTransformation):
    """
    A transformation that aims to generate a nearly cubic supercell structure
    from a structure.
    The algorithm solves for a transformation matrix that makes the supercell
    cubic. The matrix must have integer entries, so entries are rounded (in such
    a way that forces the matrix to be nonsingular). From the supercell
    resulting from this transformation matrix, vector projections are used to
    determine the side length of the largest cube that can fit inside the
    supercell. The algorithm will iteratively increase the size of the supercell
    until the largest inscribed cube's side length is at least 'min_length'
    and the number of atoms in the supercell falls in the range
    ``min_atoms < n < max_atoms``.
    """

    def __init__(
        self,
        min_atoms: Optional[int] = None,
        max_atoms: Optional[int] = None,
        min_length: float = 15.0,
        force_diagonal: bool = False,
    ):
        """
        Args:
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal
                transformation matrix.
        """
        self.min_atoms = min_atoms if min_atoms else -np.Inf
        self.max_atoms = max_atoms if max_atoms else np.Inf
        self.min_length = min_length
        self.force_diagonal = force_diagonal
        self.transformation_matrix = None

    def apply_transformation(self, structure: Structure, seed=None) -> Structure:
        """
        The algorithm solves for a transformation matrix that makes the
        supercell cubic. The matrix must have integer entries, so entries are
        rounded (in such a way that forces the matrix to be nonsingular). From
        the supercell resulting from this transformation matrix, vector
        projections are used to determine the side length of the largest cube
        that can fit inside the supercell. The algorithm will iteratively
        increase the size of the supercell until the largest inscribed cube's
        side length is at least 'num_nn_dists' times the nearest neighbor
        distance and the number of atoms in the supercell falls in the range
        defined by min_atoms and max_atoms.
        Returns:
            supercell: Transformed supercell.
        """

        lat_vecs = structure.lattice.matrix

        # boolean for if a sufficiently large supercell has been created
        sc_not_found = True

        if self.force_diagonal:
            scale = self.min_length / np.array(structure.lattice.abc)
            self.transformation_matrix = np.diag(np.ceil(scale).astype(int))
            st = SupercellTransformation(self.transformation_matrix)
            return st.apply_transformation(structure)

        # target_threshold is used as the desired cubic side lengths
        target_sc_size = self.min_length
        while sc_not_found:
            target_sc_lat_vecs = np.eye(3, 3) * target_sc_size
            self.transformation_matrix = target_sc_lat_vecs @ np.linalg.inv(lat_vecs)

            # round the entries of T and force T to be nonsingular
            self.transformation_matrix = _round_and_make_arr_singular(self.transformation_matrix)  # type: ignore

            proposed_sc_lat_vecs = self.transformation_matrix @ lat_vecs  # type: ignore

            # Find the shortest dimension length and direction
            a = proposed_sc_lat_vecs[0]
            b = proposed_sc_lat_vecs[1]
            c = proposed_sc_lat_vecs[2]

            length1_vec = c - _proj(c, a)  # a-c plane
            length2_vec = a - _proj(a, c)
            length3_vec = b - _proj(b, a)  # b-a plane
            length4_vec = a - _proj(a, b)
            length5_vec = b - _proj(b, c)  # b-c plane
            length6_vec = c - _proj(c, b)
            length_vecs = np.array(
                [
                    length1_vec,
                    length2_vec,
                    length3_vec,
                    length4_vec,
                    length5_vec,
                    length6_vec,
                ]
            )

            # Get number of atoms
            st = SupercellTransformation(self.transformation_matrix)
            superstructure = st.apply_transformation(structure)
            num_at = superstructure.num_sites

            # Check if constraints are satisfied
            if (
                np.min(np.linalg.norm(length_vecs, axis=1)) >= self.min_length
                and self.min_atoms <= num_at <= self.max_atoms
            ):
                return superstructure
            # Increase threshold until proposed supercell meets requirements
            target_sc_size += 0.1
            if num_at > self.max_atoms:
                raise AttributeError(
                    "While trying to solve for the supercell, the max "
                    "number of atoms was exceeded. Try lowering the number"
                    "of nearest neighbor distances."
                )
        raise AttributeError("Unable to find cubic supercell")

    @property
    def inverse(self):
        """
        Returns:
            None
        """
        return None

    @property
    def is_one_to_many(self):
        """
        Returns:
            False
        """
        return False

class PrimitiveCellTransformation(AbstractTransformation):
    """
    This class finds the primitive cell of the input structure.
    It returns a structure that is not necessarily orthogonalized
    Author: Will Richards
    """

    def __init__(self, tolerance=0.5):
        """
        Args:
            tolerance (float): Tolerance for each coordinate of a particular
                site. For example, [0.5, 0, 0.5] in cartesian coordinates will be
                considered to be on the same coordinates as [0, 0, 0] for a
                tolerance of 0.5. Defaults to 0.5.
        """
        self.tolerance = tolerance

    def apply_transformation(self, structure, seed=None):
        """
        Returns most primitive cell for structure.
        Args:
            structure: A structure
        Returns:
            The most primitive structure found. The returned structure is
            guaranteed to have len(new structure) <= len(structure).
        """
        return structure.get_primitive_structure(tolerance=self.tolerance)

    def __str__(self):
        return "Primitive cell transformation"

    def __repr__(self):
        return self.__str__()

    @property
    def inverse(self):
        """
        Returns: None
        """
        return None

    @property
    def is_one_to_many(self):
        """
        Returns: False
        """
        return False

