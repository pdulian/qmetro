from __future__ import annotations

from collections import deque
from collections.abc import Hashable
import copy
from itertools import product, repeat, chain
from math import prod
from typing import Callable, cast, Union, TypeVar, Any
from uuid import uuid4
import warnings

import networkx as nx
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import unitary_group

from ncon import ncon

from ...qtools import ket_bra, krauses_from_choi, dkrauses_from_choi
from ...utils import (
    get_random_positive_matrix, enhance_hermiticity,
    is_perfect_square
)




Scalar = Union[int, float, complex]
T = TypeVar('T')




class SpaceDict():
    """
    Dictionary connecting spaces to their dimensions.

    Parameters
    ----------
    name : str, optional
        Name of the dictionary. If equal to '' then the name will be
        set to a number of previously existing quantum system. By
        default ''.
    
    Attributes
    ----------
    name : str
        Name of the dictionary.
    spaces : dict[Hashable, int]
        Dictionary connecting spaces to their dimensions.
    bond_spaces : set[Hashable]
        Set of bond spaces.
    prime : str
        Suffix for primed spaces.
    primed_spaces : dict[Hashable, Hashable]
        Dictionary connecting primed spaces to unprimed ones.
    """

    counter = 0


    def __init__(self, name: str = ''):
        """
        Dictionary connecting spaces to their dimensions.

        Parameters
        ----------
        name : str, optional
            Name of the dictionary. If equal to '' then the name will be
            set to a number of previously existing quantum system. By
            default ''.
        """
        if name != '':
            self.name = name
        else:
            self.name = str(self.counter)
            self.counter += 1

        self.spaces: dict[Hashable, int] = {}
        self.bond_spaces = set()

        self.prime = 'prime'
        self.primed_spaces: dict[Hashable, Hashable] = {}


    def __str__(self):
        return self.name


    def get_dimension(self, space: Hashable) -> int:
        """
        Get space dimension.

        Parameters
        ----------
        space : str
            Space name.

        Returns
        -------
        dim : int
            Space dimension.
        """
        return self.spaces[space]


    def __getitem__(self, space: Hashable) -> int:
        return self.get_dimension(space)


    def set_dimension(self, space: Hashable, dimension: int):
        """
        Set space dimension.

        Parameters
        ----------
        space : Hashable
            Space name.
        dimension : int
            Space dimension.
        """
        self.spaces[space] = dimension


    def __setitem__(self, space: Hashable, dimension: int):
        self.set_dimension(space, dimension)


    def __iter__(self):
        return iter(self.spaces)


    def set_bond(self, space: Hashable, dim: int):
        """
        Set space dimension and mark it as bond space.

        Parameters
        ----------
        space : Hashable
            Space name.
        dimension : int
            Space dimension.
        """
        self.set_dimension(space, dim)
        self.bond_spaces.add(space)


    @property
    def irange(self) -> dict[Hashable, int]:
        """
        Index range of spaces. Index range for physical space is
        its dimension squared and for bond space it is just its dimension.

        Returns
        -------
        irange : dict[Hashable, int]
            Dictionary of spaces and their ranges.
        """
        return {
            s: d if s in self.bond_spaces else d**2
            for s, d in self.spaces.items()
        }


    def arrange_spaces(self, shape: int | tuple[int, ...], dim: int,
        prefix: Hashable='SPACE') -> list:
        """
        Add spaces with names:
        
          (prefix, i_0, ..., i_r)
          
        for i_k = 0, ..., n_k.

        Parameters
        ----------
        shape : int | tuple[int, ...]
            Shape of the index tuple (n_0, ..., n_r).
        dim : int
            Dimension of added spaces.
        prefix : Hashable, optional
            Prefix, by default 'SPACE'.

        Returns
        -------
        list
            r-dimensional list of spaces names.
        """
        if isinstance(shape, int):
            shape = (shape,)
        spaces = np.full(shape, tuple([0]), tuple)

        for multindex in product(*(range(r) for r in shape)):
            if isinstance(prefix, tuple):
                space = (*prefix, *multindex)
            else:
                space = (prefix, *multindex)
            self[space] = dim
            spaces[multindex] = space
        return spaces.tolist()


    def arrange_bonds(self, shape: int | tuple[int, ...], dim: int,
        prefix: Hashable='BOND') -> list:
        """
        Add spaces with names:
        
          (prefix, i_0, ..., i_r)
          
        for i_k = 0, ..., n_k and mark them as bond spaces.

        Parameters
        ----------
        shape : int | tuple[int, ...]
            Shape of the index tuple (n_0, ..., n_r).
        dim : int
            Dimension of added spaces.
        prefix : Hashable, optional
            Prefix, by default 'BOND'.

        Returns
        -------
        list
            r-dimensional list of spaces names.
        """
        if isinstance(shape, int):
            shape = (shape,)
        spaces = np.full(shape, tuple([0]), tuple)

        for multindex in product(*(range(r) for r in shape)):
            if isinstance(prefix, tuple):
                space = (*prefix, *multindex)
            else:
                space = (prefix, *multindex)
            self[space] = dim
            spaces[multindex] = space
            self.bond_spaces.add(space)
        return spaces.tolist()


    def ctensor(self, spaces: list[Hashable], **kwargs) -> ConstTensor:
        """
        Creates constant tensor with sdict=self.

        Parameters
        ----------
        spaces : list[Hashable]
            Tensor spaces.
        **kwargs :
            Key-word arguments passed to ConstTensor constructor.

        Returns
        -------
        tensor : ConstTensor
            Added tensor.
        """
        return ConstTensor(spaces, sdict=self, **kwargs)


    def choi_identity(self, spaces: list[Hashable] | None = None, **kwargs
        ) -> ConstTensor:
        """
        Creates Choi-like constant tensor of identity matrix with
        sdict=self.

        Parameters
        ----------
        spaces : list[Hashable] | None
            Tensor spaces, by default empty list.
        **kwargs :
            Key-word arguments passed to ConstTensor constructor.

        Returns
        -------
        tensor : ConstTensor
            Choi-like constant tensor of identity matrix.
        """
        spaces = spaces or []
        if len(spaces) == 0:
            return ConstTensor([], choi=np.array([[1]]), sdict=self)

        dimension = np.prod([self[space] for space in spaces])
        return ConstTensor(
            spaces, choi=np.identity(dimension), sdict=self, **kwargs
        )


    def zero(self, spaces: list[Hashable] | None = None, **kwargs: Any
        ) -> ConstTensor:
        """
        Creates constant tensor filled with zeros with sdict=self.

        Parameters
        ----------
        spaces : list[Hashable]
            Tensor spaces, by default empty list.
        **kwargs :
            Key-word arguments passed to ConstTensor constructor.

        Returns
        -------
        tensor : ConstTensor
            Zero const tensor.
        """
        spaces = spaces or []
        shape = tuple(self.irange[s] for s in spaces) or (1,)
        arr = np.zeros(shape, dtype=my_complex)
        return ConstTensor(spaces, array=arr, sdict=self, **kwargs)


    def primed(self, space: Hashable) -> Hashable:
        """
        For spaces returns space' where space' = (space, self.prime).

        Parameters
        ----------
        space : Hashable
            Space name.

        Returns
        -------
        new_space : Hashable
            New space (space').
        """
        return space, self.prime


    def make_primed(self, *spaces: Hashable):
        """
        Adds given spaces to the dictionary of primed spaces
        (self.primed_spaces).
        """
        for space in spaces:
            _space = self.primed(space)
            self[_space] = self[space]
            if space in self.bond_spaces:
                self.bond_spaces.add(_space)
            self.primed_spaces[_space] = space


    def unprimed(self, space: Hashable) -> Hashable:
        """
        If space' is in self returns space else returns space.

        Parameters
        ----------
        space : Hashable
            Space name.

        Returns
        -------
        unprimed_space : Hashable
            Space name without a prime.
        """
        if space in self.primed_spaces:
            return self.primed_spaces[space]
        return space




DEFAULT_SDICT = SpaceDict('default')




class GeneralTensor():
    """
    Generalized tensor class.

    Parameters
    ----------
    spaces : list[Hashable]
        Tensor spaces.
    sdict : SpaceDict, optional
        Space dictionary, by default DEFAULT_SDICT.
    name : str | None, optional
        Tensor name, by default None.
    output_spaces : list[Hashable] | None, optional
        Tensor output spaces, by default None
    comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        Comb (causal) structure in a form of [(input_0, output_0),
        (input_1, output_1), ...] where input_i (output_i) is a list
        of input (output) spaces of the i-th tooth.

    Attributes
    ----------
    spaces : list[Hashable]
        List of all tensor spaces.
    sdict : SpaceDict
        Space dictionary.
    name : str
        Tensor name.
    physical_spaces : list[Hashable]
        List of physical spaces.
    bond_spaces : list[Hashable]
        List of bond spaces.
    dimension : int
        Product of dimensions of all tensor spaces.
    physical_dim : int
        Product of dimensions of physical spaces.
    """
    counter = 0
    name_prefix = 'GENERAL TENSOR '


    def __init__(self, spaces: list[Hashable],
        sdict: SpaceDict = DEFAULT_SDICT, name: str | None = None,
        output_spaces: list[Hashable] | None = None,
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        = None):
        """
        Generalized tensor class.

        Parameters
        ----------
        spaces : list[Hashable]
            Tensor spaces.
        sdict : SpaceDict, optional
            Space dictionary, by default DEFAULT_SDICT.
        name : str | None, optional
            Tensor name, by default None.
        output_spaces : list[Hashable] | None, optional
            Tensor output spaces, by default None
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
            Comb (causal) structure in a form of [(input_0, output_0),
            (input_1, output_1), ...] where input_i (output_i) is a list
            of input (output) spaces of the i-th tooth.
        """
        if len(spaces) != len(set(spaces)):
            raise ValueError(
                f"Spaces' names have to be unique but got {spaces}."
            )

        self.sdict = sdict

        if name:
            self.name = name
        else:
            self.name = f'{self.name_prefix}{self.counter}'
            GeneralTensor.counter += 1

        bond_spaces = []
        for space in spaces:
            if space not in self.sdict:
                raise ValueError(
                    f'Space {space} does not exist in dict {self.sdict}.'
                )
            if space in self.sdict.bond_spaces:
                bond_spaces.append(space)
        self.bond_spaces: list[Hashable] = bond_spaces
        self.physical_spaces: list[Hashable] = list(
            set(spaces).difference(bond_spaces)
        )
        self.physical_dim = prod(sdict[s] for s in self.physical_spaces)

        self.spaces = list(spaces)
        self.dimension = prod(self.dimensions)

        if output_spaces is None: output_spaces = []

        self._output_spaces: list[Hashable] = []
        self._input_spaces: list[Hashable] = []

        GeneralTensor.output_spaces.__set__(self, output_spaces)

        if comb_structure is not None:
            if not self.is_choi_like:
                raise ValueError(
                    'Comb tensors must be Choi-like but got '\
                    f'bond spaces: [{self.bond_spaces}].'
                )

            output_spaces = []
            spaces_set = set(self.spaces)
            for tooth_inp, tooth_out in comb_structure:
                for s in chain(tooth_inp, tooth_out):
                    if s not in sdict:
                        raise ValueError(
                            f'Space {s} not in space dictionary {sdict}.'
                        )
                    if s not in spaces_set:
                        raise ValueError(
                            f'Space {s} not in self.spaces ('\
                            f'{self.spaces}).'
                        )
                output_spaces += tooth_out

        self._comb_structure = comb_structure
        GeneralTensor.output_spaces.__set__(self, output_spaces)


    def _contr(self, *others: Tensor | Scalar) -> Tensor:
        raise NotImplementedError


    def contr(self, *others: Tensor | Scalar) -> Tensor:
        """
        Contract tensors.

        Parameters
        ----------
        others[0...*] : Tensor | Scalar
            Tensors to be contracted with self.

        Returns
        -------
        tensor : Tensor
            Contraction result.
        """
        specific_types = (
            TensorNetwork, VarTensor, ParamTensor, ConstTensor
        )
        for obj in (self, *others):
            if (
                not isinstance(obj, specific_types)
                and not isinstance(obj, (int, float, complex))
            ):
                raise ValueError(
                    'Cannot contract non-specific (generalized) tensors.'
                )

        # Filter all scalars and chceck for other space dictionaries.
        x = 1 # Product of all scalars on the list.
        tensors: list[GeneralTensor] = [self]
        for other in others:
            if isinstance(other, (int, float, complex)):
                x *= other
            elif self.sdict is not other.sdict:
                raise ValueError(
                    'Contraction is possible only for tensors with the '\
                    'same space dictionary.\n'\
                    f'Tried to contrat tensor with {self.sdict.name} '\
                    f'and tensor with {other.sdict.name}.')
            elif isinstance(other, ConstTensor) and len(other.spaces) == 0:
                x *= other.array[0]
            else:
                tensors.append(other)

        if any(
            isinstance(tensor, (TensorNetwork, VarTensor))
            for tensor in tensors
        ):
            return TensorNetwork(tensors=[x, *tensors], sdict=self.sdict)

        for i, tensor in enumerate(tensors):
            if isinstance(tensor, ParamTensor):
                new: ParamTensor = tensor._contr(
                    *(tensors[:i] + tensors[i + 1:])
                )
                new.array *= x
                new.dtensor.array *= x
                return new

        _new: ConstTensor = tensors[0]._contr(*tensors[1:])
        _new.array *= x
        return _new


    def kron(self, *others: Tensor | Scalar) -> Tensor:
        """
        For Choi-like tensors it computes their Kronecker product and for
        other tensors it is just contraction where there are no doubled
        indices (no index gets contracted).

        Parameters
        ----------
        others[0...*] : Tensor | Scalar
            Tensors Kronecker multiplied with self.

        Returns
        -------
        tensor : Tensor
            Kronecker product.
        """
        all_spaces = set(self.spaces)
        x = 1
        for other in others:
            if isinstance(other, (int, float, complex)):
                x *= other
            else:
                if self.sdict is not other.sdict:
                    raise ValueError(
                        'Kronecker product is possible only between '\
                        'tensors with the same space dictionary.\n'\
                        'Tried to multiply tensor with '\
                        f'{self.sdict.name} and {other.sdict.name}.'
                    )

                space_set = set(other.spaces)
                if all_spaces.intersection(space_set):
                    spacess = []
                    for t in chain([self], others):
                        try:
                            spacess.append(t.spaces)
                        except AttributeError:
                            continue

                    raise ValueError(
                        'Kronecker product is possible only between '\
                        'tensors acting on different spaces.\nTried to '\
                        'compute Kronecker product of tensors on spaces:'\
                        f' {spacess}.')
                all_spaces = all_spaces.union(space_set)

        # For tesnors acting on different spaces contraction is Kronecker
        # product.
        return self.contr(x, *others)


    def choi_trace(self, *spaces: Hashable, full: bool = False
        ) -> Tensor | complex:
        """
        For Choi-like tensor compute partial trace of the Choi matrix.

        Parameters
        ----------
        spaces[0...*] : Hashable
            Spaces to be traced out.
        full : bool, optional
            If True then computes the trace over all spaces and returns
            a complex number, by default False.

        Returns
        -------
        tensor : Tensor | complex
            Tensor of the result.

        """
        if full:
            spaces = tuple(self.spaces)
        else:
            spaces = tuple(set(spaces).intersection(set(self.spaces)))
        
        if set(spaces).intersection(self.bond_spaces):
            raise ValueError(
                'Choi trace can be performed only on physical spaces '\
                f'({self.physical_spaces}) but provided {spaces}.'
            )

        if not spaces:
            new = self.copy()
            new.sdict = self.sdict
            return new

        return self.contr(
            *(self.sdict.choi_identity([space]) for space in spaces)
        )


    def copy(self) -> Tensor:
        """
        Make a copy.

        Returns
        -------
        copy : Tensor
            Tensor copy.
        """
        return copy.copy(self)


    def __mul__(self, other: Tensor | Scalar) -> Tensor:
        return self.contr(other)


    def __rmul__(self, other: Tensor | Scalar) -> Tensor:
        return self.contr(other)


    def __str__(self):
        result = f'name: {self.name}'
        if self.sdict is not DEFAULT_SDICT:
            result += f'\nspace dictionary: {self.sdict.name}'
        result += f'\nspaces: {self.spaces}'
        return result


    def respace(self: GeneralTensor, spaces: list[Hashable] | None = None,
        space_map: Callable[[Hashable], Hashable] | None = None,
        sdict: SpaceDict | None = None, name: str | None = None
        ) -> GeneralTensor:
        """
        Make a copy of self but with renamed spaces.

        Parameters
        ----------
        spaces : list[Hashable] | None, optional
            The change of spaces will take the form
            self.spaces[i] -> spaces[i]. If None then change of spaces will
            be carried out using space_map, by default None.
        space_map : Callable[[Hashable], Hashable] | None, optional
            The change of spaces will take the form
            space -> space_map(space), by default None
        sdict : SpaceDict | None, optional
            Space dictionary of the copy. If None then it will be
            self.sdict, by default None.
        name : str | None, optional
            Name of the copy. If None then it will be self.name, by
            default None.

        Returns
        -------
        copy : GeneralTensor
            New tensor with renamed spaces.

        """
        if sdict is None:
            sdict = self.sdict

        new_spaces = spaces
        if new_spaces is None:
            if space_map is None:
                raise ValueError(
                    'One of the arguments spaces or space_map has to be'\
                    ' provided.'
                )
        else:
            to_new = dict(zip(self.spaces, new_spaces))
            space_map = lambda space: to_new[space]
        new_spaces = [space_map(space) for space in self.spaces]
        new_output_spaces = [
            space_map(space) for space in self.output_spaces
        ]

        comb_str = None
        if self.is_comb:
            comb_str = []
            for tooth_inps, tooth_outs in self.comb_structure:
                comb_str.append((
                    [space_map(s) for s in tooth_inps],
                    [space_map(s) for s in tooth_outs],
                ))

        return GeneralTensor(
            new_spaces, sdict=sdict, output_spaces=new_output_spaces,
            name=name if name is not None else self.name,
            comb_structure=comb_str
        )


    @property
    def dimensions(self) -> list[int]:
        """
        Dimensions of the tensor spaces.

        Returns
        -------
        dims : list[int]
            List of dimensions of self.spaces.
        """
        return [self.sdict[s] for s in self.spaces]


    @property
    def output_spaces(self) -> list[Hashable]:
        """
        List of the output spaces.

        Returns
        -------
        output_spaces : list[Hashable]
            List of output spaces.
        """
        return self._output_spaces


    @output_spaces.setter
    def output_spaces(self, output_spaces: list[Hashable]):
        all_set = set(self.spaces)
        bond_set = set(self.bond_spaces)
        for space in output_spaces:
            if space not in all_set:
                raise ValueError(
                    f'Space {space} provided in output spaces'\
                    f' is not in spaces: {self.spaces}.'
                )
            if space in bond_set:
                raise ValueError(
                    f'Space {space} provided in output spaces'\
                    f' is a bond space: {self.bond_spaces}.'
                )

        out_set = set(output_spaces)
        phys_set = set(self.physical_spaces)

        self._output_spaces = list(out_set)
        self._input_spaces = list(phys_set.difference(out_set))


    @property
    def output_dim(self) -> int:
        """
        Output dimension, that is the product of the all output spaces
        dimensions.

        Returns
        -------
        dim : int
            Output dimension.
        """
        return prod(self.sdict[s] for s in self.output_spaces)


    @property
    def input_spaces(self) -> list[Hashable]:
        """
        List of the input spaces.

        Returns
        -------
        input_spaces : list[Hashable]
            List of input spaces.
        """
        return self._input_spaces


    @input_spaces.setter
    def input_spaces(self, input_spaces: list[Hashable]):
        all_set = set(self.spaces)
        bond_set = set(self.bond_spaces)
        for space in input_spaces:
            if space not in all_set:
                raise ValueError(
                    f'Space {space} provided in input spaces'\
                    f' is not in spaces: {self.spaces}.'
                )
            if space in bond_set:
                raise ValueError(
                    f'Space {space} provided in input spaces'\
                    f' is a bond space: {self.bond_spaces}.'
                )

        inp_set = set(input_spaces)
        phys_set = set(self.physical_spaces)

        self._output_spaces = list(phys_set.difference(inp_set))
        self._input_spaces = list(inp_set)


    @property
    def input_dim(self) -> int:
        """
        Input dimension, that is the product of the all input spaces
        dimensions.

        Returns
        -------
        dim : int
            Input dimension.
        """
        return prod(self.sdict[s] for s in self.input_spaces)


    @property
    def shape(self) -> tuple[int, ...]:
        """
        Tensor shape, that is index ranges of self.spaces.

        Returns
        -------
        shape : tuple[int, ...]
            Tensor shape.
        """
        if self.spaces:
            return tuple(self.sdict.irange[s] for s in self.spaces)
        return (1,)


    @property
    def is_choi_like(self) -> bool:
        """
        Whether the tensor is Choi-like. The tensor is Choi-like if all its
        spaces are physical.        

        Returns
        -------
        is_choi_like : bool
            Whether the tensor is Choi-like.
        """
        return not self.bond_spaces


    @property
    def is_comb(self) -> bool:
        """
        Whether the tensor has comb (causal) structure.        

        Returns
        -------
        is_comb : bool
            Whether the tensor has comb structure.
        """
        cs = self.comb_structure
        return cs is not None and len(cs) > 1


    @property
    def comb_structure(self
        ) -> list[tuple[list[Hashable], list[Hashable]]] | None:
        """
        Comb (causal) structure in a form of
        
        ``[(input_0, output_0), (input_1, output_1), ...]``,

        where ``input_i`` (``output_i``) is a list of input (output)
        spaces of the i-th tooth.

        Returns
        -------
        list[tuple[list[Hashable], list[Hashable]]] | None
            Comb structure.
        """        
        return self._comb_structure


    @comb_structure.setter
    def comb_structure(self,
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        ):
        # TODO check correctness
        self._comb_structure = comb_structure




class ConstTensor(GeneralTensor):
    """
    Class of tensors that are constant during the optimization.

    Parameters
    ----------
    spaces : list[Hashable]
        Tensor spaces.
    choi : np.ndarray | None, optional
        Choi matrix which from which the tensor will be constructed. If
        None then it will be initialized from the array argument, by
        default None.
    sdict : SpaceDict, optional
        Space dictionary, by default DEFAULT_SDICT.
    array : np.ndarray | None, optional
        Tensor as numpy array, by default None.
    name : str | None, optional
        Tensor name, by default None.
    output_spaces : list[Hashable] | None, optional
        Tensor output spaces, by default None.
    comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        Comb (causal) structure in a form of [(input_0, output_0),
        (input_1, output_1), ...] where input_i (output_i) is a list
        of input (output) spaces of the i-th tooth.
    
    Attributes
    ----------
    array : np.ndarray
        Tensor as numpy array. The order of indices corresponds to
        the order of spaces in ``self.spaces``.
    """
    name_prefix = 'CONST TENSOR '
    contr_count = 0

    def __init__(self, spaces: list[Hashable],
        choi: np.ndarray | None = None,
        sdict: SpaceDict = DEFAULT_SDICT, array: np.ndarray | None = None,
        name: str | None = None,
        output_spaces: list[Hashable] | None = None,
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        = None):
        """
        Class of tensors that are constant during the optimization.

        Parameters
        ----------
        spaces[0...*] : Hashable
            Tensor spaces.
        choi : np.ndarray | None, optional
            Choi matrix which from which the tensor will be constructed. If
            None then it will be initialized from the array argument, by
            default None.
        sdict : SpaceDict, optional
            Space dictionary, by default DEFAULT_SDICT.
        array : np.ndarray | None, optional
            Tensor as numpy array, by default None.
        name : str | None, optional
            Tensor name, by default None.
        output_spaces : list[Hashable] | None, optional
            Tensor output spaces, by default None.
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
            Comb (causal) structure in a form of [(input_0, output_0),
            (input_1, output_1), ...] where input_i (output_i) is a list
            of input (output) spaces of the i-th tooth.
        """
        super().__init__(
            spaces, sdict=sdict, name=name, output_spaces=output_spaces,
            comb_structure=comb_structure
        )

        if choi is not None:
            if self.bond_spaces:
                raise ValueError(
                    'Constant tensor with bond spaces ('
                    f'{self.bond_spaces}) cannot be initialized from a'
                    ' Choi matrix.'
                )

            _matrix = np.array(choi, dtype=my_complex)
            if _matrix.shape != (self.dimension, self.dimension):
                raise ValueError(
                    'Matrix has to be of the appropiate dimension.\n'
                    f'Matrix of shape {_matrix.shape} given while the Choi'
                    f' matrix dimension is {self.dimension}.'
                )
            self.array = self._choi_to_tensor_arr(
                _matrix, self.dimensions
            )
        elif array is None:
            _matrix = np.zeros(
                (self.dimension, self.dimension), dtype=my_complex
            )
            self.array = self._choi_to_tensor_arr(
                _matrix, self.dimensions
            )
        else:
            if isinstance(array, (int, float, complex)):
                array = np.array([array])

            if self.shape != array.shape:
                raise ValueError(
                    'Tensor has to be of appropriate shape.\n'\
                    f'Tensor of shape {array.shape} given while the'\
                    f' expected shape was {self.shape}.'
                )

            if array.dtype == my_complex:
                self.array = array # Don't copy tensor if not necessary.
            else:
                self.array = array.astype(my_complex)


    @staticmethod
    def _choi_to_tensor_arr(matrix: np.ndarray, dims: list[int]
        ) -> np.ndarray:
        """
        Choi matrix array to tensor array.

        Parameters
        ----------
        matrix : np.ndarray
            Choi matrix.
        dims : list[int]
            Space dimensions.

        Returns
        -------
        array : np.ndarray
            Tensor array.
        """
        n = len(dims)
        order = [-i for i in range(1, 2 * n, 2)]
        order += [-i for i in range(2, 2 * n + 1, 2)]
        tensor = np.reshape(matrix, dims + dims)
        if order:
            tensor = ncon(tensor, order)
            return np.reshape(tensor, [d**2 for d in dims])
        return np.array([tensor])


    @staticmethod
    def _tensor_arr_to_choi(array: np.ndarray, dims: list[int]
        ) -> np.ndarray:
        """
        Tensor array to Choi matrix array.

        Parameters
        ----------
        array : np.ndarray
            Tensor array.
        dims : list[int]
            Space dimensions.

        Returns
        -------
        array : np.ndarray
            Choi matrix array.
        """
        if not dims:
            return np.array([[array[0]]])
        n = len(dims)
        array = np.reshape(array, np.concatenate([[d, d] for d in dims]))
        order = sum(([-i, -i - n] for i in range(1, n + 1)), [])
        array = ncon(array, order)
        return np.reshape(array, (np.prod(dims), np.prod(dims)))


    def choi(self: ConstTensor, spaces: list[Hashable]) -> np.ndarray:
        """
        Compute Choi matrix for the Choi-like tensor.

        Parameters
        ----------
        spaces : list[Hashable]
            List of spaces defining the order of spaces of the result.

        Returns
        -------
        matrix : np.ndarray
            Choi matrix.

        """
        if self.bond_spaces:
            raise ValueError(
                'Choi matrix is defined only for tensors without '\
                f'bond spaces ({self.bond_spaces}).'
            )
        if (set(spaces) != set(self.spaces)
            or len(spaces) != len(self.spaces)):
            raise ValueError(
                f'Tried to get matrix on {spaces} for tensor on '\
                f' {self.spaces}.'
            )
        if spaces:
            self.reorder(spaces)
        return self._tensor_arr_to_choi(self.array, list(self.dimensions))


    def __getitem__(self, nindex: dict[Hashable, int]) -> complex:
        indices = [nindex[s] for s in self.spaces]
        return self.array[tuple(indices)]


    def __setitem__(self, nindex: dict[Hashable, int], item: complex):
        indices = [nindex[s] for s in self.spaces]
        self.array[tuple(indices)] = item


    def reorder(self, new_spaces: list[Hashable]) -> ConstTensor:
        """
        Change the order of spaces.

        Parameters
        ----------
        new_spaces : list[Hashable]
            Spaces names in the new order.

        Returns
        -------
        self : ConstTensor
            self
        """
        if set(new_spaces) != set(self.spaces):
            raise ValueError(
                'When reordering new spaces can differ from old spaces '\
                f'only in order. Tried to reorder {self.spaces} into '\
                f'{new_spaces}.'
            )

        new_order = [-new_spaces.index(space) - 1 for space in self.spaces]
        self.array = ncon(self.array, new_order)
        self.spaces = new_spaces

        return self


    def choi_matmul(self, other: ConstTensor) -> ConstTensor:
        """
        For two Choi-like tensors compute the tensor of their Choi matrices
        multiplication.

        Parameters
        ----------
        other : ConstTensor
            Tensor to multiply.

        Returns
        -------
        product : ConstTensor
            Tensor of the product.
        """
        if self.sdict is not other.sdict:
            raise ValueError(
                'Matrix multiplication is possible only between Choi'\
                'matrices (tensors) with the same space dictionary.\n'\
                f'Tried to multiply matrix on {self.sdict.name} with'\
                f' matrix on {other.sdict.name}.'
            )

        if (
            set(other.spaces) != set(self.spaces)
            or len(other.spaces) != len(self.spaces)
        ):
            raise ValueError(
                'Matrix multiplication is possible only between Choi'\
                'matrices acting on the same spaces.\n'\
                f'Tried to multiply matrix on {self.spaces} with matrix'\
                f' on {other.spaces}.'
            )

        return ConstTensor(
            self.spaces,
            choi=self.choi(self.spaces) @ other.choi(self.spaces),
            sdict=self.sdict
        )


    def choi_transpose(self, *spaces: Hashable, full: bool = False
        ) -> ConstTensor:
        """
        Computes partial transposition of Choi-like tensor and for other
        tensors it does transpose-like reshuffling of entries on
        physcial spaces.

        Parameters
        ----------
        full : bool, optional
            If True tranposes all spaces, by default False.

        Returns
        -------
        transpostion: ConstTensor
            Tensor of the transposed matrix.
        """
        space_set = set(spaces)
        if space_set.intersection(self.bond_spaces):
            raise ValueError(
                'Choi transpose can be done only on physical spaces '\
                f'({self.physical_spaces}) but provided {spaces}.'
            )
        if full:
            tran_spaces = set(self.physical_spaces)
        elif spaces:
            tran_spaces = space_set.intersection(self.spaces)
        else:
            return self.copy()

        kept_spaces = set(self.spaces).difference(tran_spaces)
        self.reorder(list(kept_spaces) + list(tran_spaces))
        shape = self.shape
        new_array = self.array.copy()
        dims = self.dimensions[len(kept_spaces):]
        indicess = product(*(range(r) for r in shape[:len(kept_spaces)]))
        for indices in indicess:
            choi = self._tensor_arr_to_choi(self.array[indices], dims)
            new_array[indices] = self._choi_to_tensor_arr(choi.T, dims)

        return ConstTensor(
            self.spaces, array=new_array, sdict=self.sdict,
            name=self.name, output_spaces=self.output_spaces
        )


    def choi_T(self, *spaces: Hashable, full: bool = False) -> ConstTensor:
        """
        Computes partial transposition of Choi-like tensor and for other
        tensors it does transpose-like reshuffling of entries on
        physcial spaces.

        Parameters
        ----------
        full : bool, optional
            If True tranposes all spaces, by default False.

        Returns
        -------
        transpostion: ConstTensor
            Tensor of the transposed matrix.
        """
        return self.choi_transpose(*spaces, full=full)


    def choi_trace(self, *spaces: Hashable, full: bool = False
        ) -> Tensor | complex:
        """
        For Choi-like tensor compute partial trace of the Choi matrix
        and return its tensor form.

        Parameters
        ----------
        spaces[0...*] : Hashable
            Spaces to be traced out.
        full : bool, optional
            If True then computes the trace over all spaces and returns
            a complex number, by default False.

        Returns
        -------
        tensor : Tensor | complex
            Tensor form of the result.
        """
        if full:
            return np.trace(self.choi(self.spaces))

        return super().choi_trace(*spaces, full=False)


    def _contr(self, *others: ConstTensor) -> ConstTensor:
        """
        Contraction of two constant tensors.

        Returns
        -------
        tensor : ConstTensor
            Result of contraction.

        """
        ConstTensor.contr_count += 1

        x: complex = 1 # Contraction of all scalars on the list.
        tmp = []
        for other in others:
            if isinstance(other, (int, float, complex)):
                x *= other
            elif self.sdict is not other.sdict:
                raise ValueError(
                    'Contraction is possible only between tensors'\
                    'with the same space dictionary.\n'\
                    f'Tried to multiply tensor with {self.sdict.name}'\
                    f' and tensor with {other.sdict.name}.'
                )
            elif len(other.spaces) == 0:
                x *= other.array[0]
            else:
                tmp.append(other)
        _others: list[ConstTensor] = tmp

        if len(_others) == 0:
            new = self.copy()
            new.array *= x
            return new

        space_indices = {}
        i_same = 1
        i_different = -1

        if self.spaces:
            tensors = [self] + list(_others)
        else:
            x *= self.array[0]
            if not _others:
                new = _others[0].copy()
                new.array *= x
                return new

            tensors = list(_others)

        # space_sets[i] is a union of all spaces of tensors[i + 1:]
        space_sets = []
        space_set = set()
        for tensor in tensors[::-1]:
            space_sets.append(space_set)
            space_set = space_set.union(set(tensor.spaces))
        space_sets = space_sets[::-1]

        new_spaces = []
        tensor_inidicess = []

        for i, tensor in enumerate(tensors):
            space_set = space_sets[i]
            tensor_inidices = []
            for space in tensor.spaces:
                if space not in space_indices:
                    if space in space_set:
                        space_indices[space] = i_same
                        i_same += 1
                    else:
                        space_indices[space] = i_different
                        i_different -= 1
                        new_spaces.append(space)
                tensor_inidices.append(space_indices[space])
            tensor_inidicess.append(tensor_inidices)

        return ConstTensor(
            new_spaces, sdict=self.sdict,
            array=x * ncon(
                [tensor.array for tensor in tensors], tensor_inidicess
            )
        )


    def __str__(self):
        result = super().__str__()
        result += f'\ntensor:\n {self.array}'
        return result


    def __eq__(self, other: ConstTensor) -> bool:
        if self.sdict is not other.sdict:
            return False

        if not set(self.spaces) == set(other.spaces):
            return False

        other.reorder(self.spaces)
        return bool(np.all(self.array == other.array))


    def to_qchannel(self, input_spaces: list[Hashable] | None = None,
        output_spaces: list[Hashable] | None = None
        ) -> Callable[[np.ndarray], ConstTensor]:
        """
        Returns a quantum channel of the Choi matrix of the Choi-like
        tensor.

        Parameters
        ----------
        input_spaces : list[Hashable] | None, optional
            Input spaces. If None then self.input_spaces, by default None.
        output_spaces : list[Hashable] | None, optional
            Output spaces. If None then self.output_spaces, by default None.

        Returns
        -------
        channel : Callable[[np.ndarray], ConstTensor]
            Quantum channel of the Choi matrix.

        """
        if self.bond_spaces:
            raise ValueError(
                'Qunatum channel can be obtaine only from the choi-like '\
                'tensor, that is tensor without bond spaces ('\
                f'{self.bond_spaces}).'
            )
        inp = input_spaces if input_spaces else self.input_spaces
        out = output_spaces if output_spaces else self.output_spaces
        rest = set(self.spaces).difference(inp).difference(out)
        self_copy = copy.deepcopy(self)
        self_copy.sdict = self.sdict

        def fun(rho):
            rho = ConstTensor(
                inp, choi=np.copy(rho), sdict=self.sdict,
                output_spaces=inp
            )
            return (self_copy * rho).choi_trace(*rest).reorder(out)

        return fun


    def krauses(self, input_spaces: list[Hashable] | None = None,
        output_spaces: list[Hashable] | None = None) -> list[np.ndarray]:
        """
        Computes Kraus operators of the Choi matrix of the Choi-like
        tensor.

        Parameters
        ----------
        input_spaces : list[Hashable] | None, optional
            Input spaces. If None then self.input_spaces, by default None.
        output_spaces : list[Hashable] | None, optional
            Output spaces. If None then self.output_spaces, by default
            None.

        Returns
        -------
        krauses : list[np.ndarray]
            List of Kraus operators.
        """
        if not self.is_choi_like:
            raise ValueError(
                'Kraus operators can be optained only for Choi-like'\
                'tensors, that is tensors without bond spaces ('\
                f'{self.bond_spaces}).'
            )
        _input = input_spaces if input_spaces else self.input_spaces
        output = output_spaces if output_spaces else self.output_spaces
        rest = set(self.spaces).difference(_input).difference(output)
        chm = cast(ConstTensor, self.choi_trace(*rest))

        inp_dim = prod(self.sdict[space] for space in _input)
        out_dim = prod(self.sdict[space] for space in output)

        matrix = chm.choi(output + _input)

        return krauses_from_choi(matrix, (inp_dim, out_dim))


    def __add__(self, other: ConstTensor) -> ConstTensor:
        if not isinstance(other, ConstTensor):
            return NotImplemented
        
        if self.sdict is not other.sdict:
            raise ValueError(
                'Tensor addition is possible only between tensors with '\
                'the same space dictionary. Tried to add tensors with '\
                f' {other.sdict.name} and {self.sdict.name}.'
            )

        if set(other.spaces) != set(self.spaces):
            raise ValueError(
                'Tensor addition is possible only between tensors acting '\
                'on the same spaces. Tried to add tensors on '\
                f'{self.spaces} and {other.spaces}.'
            )

        if self.spaces:
            other.reorder(self.spaces)

        return ConstTensor(
            self.spaces, array=self.array + other.array, sdict=self.sdict
        )


    def __iadd__(self, other: ConstTensor) -> ConstTensor:
        if self.sdict is not other.sdict:
            raise ValueError(
                'Tensor addition is possible only between tensors with '\
                'the same space dictionary. Tried to add tensors with '\
                f' {other.sdict.name} and {self.sdict.name}.'
            )

        if set(other.spaces) != set(self.spaces):
            raise ValueError(
                'Tensor addition is possible only between tensors acting '\
                'on the same spaces. Tried to add tensors on '\
                f'{self.spaces} and {other.spaces}.'
            )

        if self.spaces:
            other.reorder(self.spaces)
        
        self.array += other.array

        return self


    def __sub__(self, other: ConstTensor) -> ConstTensor:
        return self + (-1) * other


    def __isub__(self, other: ConstTensor) -> ConstTensor:
        self += (-1) * other
        return self


    def update_choi(self, spaces: list[Hashable], matrix: np.ndarray):
        """
        Set tensor entries to make it Choi matrix equal to the given
        matrix.

        Parameters
        ----------
        spaces : list[Hashable]
            Spaces names,
        matrix : np.ndarray
            Choi matrix.
        """
        if self.bond_spaces:
            raise ValueError(
                'Choi update can be performed only for Choi-like'\
                'tensors, that is tensors without bond spaces ('\
                f'{self.bond_spaces}).'
            )

        new_array = self._choi_to_tensor_arr(
            matrix, [self.sdict[space] for space in spaces]
        )
        self.reorder(spaces)
        if new_array.shape != self.array.shape:
            raise ValueError(
                f'Matrix of shape {matrix.shape} is incompatible with'
                f' tensor of shape ({self.spaces}, {self.shape}).'
            )
        self.array = new_array


    def respace(self, spaces: list[Hashable] | None = None,
        space_map: Callable[[Hashable], Hashable] | None = None,
        sdict: SpaceDict | None = None, name: str | None = None,
        make_copy: bool = False) -> ConstTensor:
        """
        Make a copy of self but with renamed spaces.

        Parameters
        ----------
        spaces : list[Hashable] | None, optional
            The change of spaces will take the form
            self.spaces[i] -> spaces[i]. If None then change of spaces
            will be carried out using space_map, by default None.
        space_map : Callable[[Hashable], Hashable] | None, optional
            The change of spaces will take the form
            space -> space_map(space), by default None
        sdict : SpaceDict | None, optional
            Space dictionary of the copy. If None then it will be
            self.sdict, by default None.
        name : str | None, optional
            Name of the copy. If None then it will be self.name, by
            default None.
        make_copy : bool, optional
            Whether to make a copy of `self.array`, by default False.

        Returns
        -------
        copy : ConstTensor
            New tensor with renamed spaces.

        """
        templ = super().respace(spaces, space_map, sdict, name)

        array = self.array
        if make_copy:
            array = self.array.copy()

        new = ConstTensor(
            templ.spaces, sdict=templ.sdict, name=templ.name,
            output_spaces=templ.output_spaces, array=array,
            comb_structure=templ.comb_structure
        )

        return new


    def copy(self) -> ConstTensor:
        """
        Make a copy.

        Returns
        -------
        copy : ConstTensor
            Tensor copy.
        """
        new = copy.deepcopy(self)
        new.sdict = self.sdict
        return new


    def square_without(self, spaces: list[Hashable]) -> ConstTensor:
        """
        Computes Choi matrix square on spaces different than given spaces.
        
        For tensor T[a0, a1, a2, a3] and spaces [a0, a1] computes a new
        tensor U such that

            U[a0, a1].choi == T[a0, a1].choi @ T[a0, a1].choi.
        
        It can be used to compute matrix squares for tensor networks. For
        example for physcial spaces p0, p1 and bond space b0 and tensors:

            V[p0, p1] = T[p0, b0] * U[b0, p1],
        
        it satifies:

            V.choi([p0, p1]) @ V.choi([p0, p1]) == (T.square_wiyhout([b0])
            * U.square_wiyhout([b0])).choi([p0, p1]).
        
        Note that this requires T.square_wiyhout([b0]) to have two b0
        spaces. This is solved by adding b0' space using SpaceDictionar
        primed method.

        Only physical spaces can be squared.

        Parameters
        ----------
        spaces : list[Hashable]
            Spaces to be omitted.

        Returns
        -------
        tensor : ConstTensor
            Matrix square without given spaces.

        """
        tensor = copy.deepcopy(self.array)
        to_square = set(self.spaces).difference(spaces)
        # self.reorder(list(to_square) + spaces)

        if to_square.intersection(self.bond_spaces):
            raise ValueError(
                'Squaring can be done only on physical spaces ('\
                f'{self.physical_spaces}) but asked for {list(to_square)}.'
            )

        sd = self.sdict
        sd.make_primed(*spaces)

        tensor_new_dims = []
        contr_i = 1
        non_contr_i = -1
        tensor0_indices: list[int] = []
        tensor1_indices: list[int] = []
        new_tensor_new_dims = []
        new_spaces = []
        was_primed = set()
        for space in self.spaces:
            d = sd[space]
            r = sd.irange[space]
            if space in to_square:
                tensor_new_dims += [d, d] # It must be physical (r=d**2)
                tensor0_indices += [non_contr_i, contr_i]
                tensor1_indices += [contr_i, non_contr_i - 1]
                new_tensor_new_dims.append(d**2)
                new_spaces.append(space)
                contr_i += 1
            else:
                tensor_new_dims.append(r)
                tensor0_indices.append(non_contr_i)
                tensor1_indices.append(non_contr_i - 1)
                new_tensor_new_dims += [r, r]
                new_spaces += [space, sd.primed(space)]
                was_primed.add(space)
            non_contr_i -= 2
        tensor = tensor.reshape(tensor_new_dims)
        new_tensor = ncon(
            [tensor, tensor], [tensor0_indices, tensor1_indices]
        )
        new_tensor = new_tensor.reshape(new_tensor_new_dims)

        new_output_spaces = self.output_spaces.copy()
        for s in self.output_spaces:
            if s in was_primed:
                new_output_spaces.append(sd.primed(s))

        return ConstTensor(
            new_spaces, sdict=sd, array=new_tensor,
            output_spaces=new_output_spaces
        )


    @staticmethod
    def from_mps(array: np.ndarray, spaces: list[Hashable],
        sdict: SpaceDict = DEFAULT_SDICT,
        output_spaces: list[Hashable] | None = None,
        name: str | None = None) -> ConstTensor:
        """
        Computess a constant tensor of density MPO element from
        the array of MPS element.

        Parameters
        ----------
        array : np.ndarray
            Array of MPS element.
        spaces : list[Hashable]
            Array's order of spaces.
        sdict : SpaceDict, optional
            Space dictionary, by default DEFAULT_SDICT.
        output_spaces : list[Hashable] | None, optional
            Output spaces, by default None.
        name : str | None, optional
            New tensor name, by default None.

        Returns
        -------
        tensor : ConstTensor
            Density MPO element.
        """
        sd = sdict
        left_indices = [-1 - 2*i for i in range(len(spaces))]
        right_indices = [-2 - 2*i for i in range(len(spaces))]
        tensor_sq = ncon(
            [array.conjugate(), array],
            [left_indices, right_indices]
        )
        tensor_sq = tensor_sq.reshape([sd.irange[s] for s in spaces])
        return ConstTensor(
            spaces, sdict=sd, output_spaces=output_spaces,
            array=tensor_sq, name=name
        )


    def to_mps(self, spaces: list[Hashable], cutoff: float = 1e-10
        ) -> list[np.ndarray]:
        """
        Converts tensor form of density MPO component into MPS
        component(s).

        Parameters
        ----------
        spaces : list[Hashable]
            Order of spaces for the result.
        cutoff : float, optional
            Cutoff for small eigenvalues. The default is 1e-10.

        Returns
        -------
        list[np.ndarray]
            List of MPS components. Each component is a numpy array of
            shape (sqrt(range_s) for s in spaces). For pure states it
            should be a list of length 1.
        """
        if set(self.spaces) != set(spaces):
            raise ValueError(
                f'spaces ({spaces}) must contain the same elements as'
                f' self.spaces ({self.spaces}).'
            )
        
        sd = self.sdict
        irange = sd.irange
        if not all(is_perfect_square(irange[s]) for s in self.bond_spaces):
            raise ValueError(
                'Dimensions of bond spaces must be perfect squares but '
                f'got {[sd[s] for s in self.bond_spaces]}.'
            )
        
        array = self.reorder(spaces).array
        
        # Split indices.
        sqrt_irange = [int(np.sqrt(irange[s])) for s in spaces]
        array = array.reshape([x for r in sqrt_irange for x in (r, r)])

        # Express tensor as a matrix.
        n = 2 * len(spaces)
        order = list(range(0, n, 2)) + list(range(1, n, 2))
        array = np.transpose(array, order)
        mat = array.reshape(prod(sqrt_irange), prod(sqrt_irange))

        # Get states.
        eigvals, eigvecs = np.linalg.eigh(mat)

        result = []
        for e, v in zip(eigvals, eigvecs.T):
            if e > cutoff:
                result.append(np.sqrt(e) * v.reshape(sqrt_irange))
        return result




class VarTensor(GeneralTensor):
    """
    Class of tensors that are optimized by iss.

    Parameters
    ----------
    spaces : list[Hashable]
        Tensor spaces.
    sdict : SpaceDict, optional
        Space dictionary, by default DEFAULT_SDICT.
    name : str | None, optional
        Tensor name, by default None.
    output_spaces : list[Hashable] | None, optional
        Tensor output spaces, by default None
    is_unital : bool, optional
        Unitality constraint, by default False.
    is_measurement : bool, optional
        Whether it is a measurment-like (SLD-like) variable, by default
        False.
    comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        Comb (causal) structure in a form of [(input_0, output_0),
        (input_1, output_1), ...] where input_i (output_i) is a list
        of input (output) spaces of the i-th tooth.

    Attributes
    ----------
    is_unital : bool
        Whether the tensor is unital.
    is_measurement : bool
        Whether the tensor is measurement-like (SLD-like) variable.
    """
    name_prefix = 'VAR TENSOR '

    def __init__(self, spaces: list[Hashable],
        sdict: SpaceDict = DEFAULT_SDICT, name: str | None = None,
        output_spaces: list[Hashable] | None = None,
        is_unital: bool = False, is_measurement: bool = False,
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        = None):
        """
        Class of tensors that are optimized by iss.

        Parameters
        ----------
        spaces : list[Hashable]
            Tensor spaces.
        sdict : SpaceDict, optional
            Space dictionary, by default DEFAULT_SDICT.
        name : str | None, optional
            Tensor name, by default None.
        output_spaces : list[Hashable] | None, optional
            Tensor output spaces, by default None
        is_unital : bool, optional
            Unitality constraint, by default False.
        is_measurement : bool, optional
            Whether it is a measurment-like (SLD-like) variable, by
            default False.
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
            Comb (causal) structure in a form of [(input_0, output_0),
            (input_1, output_1), ...] where input_i (output_i) is a list
            of input (output) spaces of the i-th tooth.
        """
        super().__init__(
            spaces, sdict=sdict, name=name, output_spaces=output_spaces,
            comb_structure=comb_structure
        )
        if is_unital and self.output_dim != self.input_dim:
            raise ValueError(
                'Unital Choi matrix must have the same input and output'\
                f' dimensions but got {self.output_dim} and '\
                f'{self.input_dim}.'
            )
        self.is_unital = is_unital
        self.is_measurement = is_measurement

        if not is_measurement:
            for space in self.bond_spaces:
                d = self.sdict[space]
                if d != int(np.sqrt(d))**2:
                    raise ValueError(
                        'Bond spaces of input state density matrices MPO'\
                        f' must be squares but got {d}.\n'
                        'MPS elements are kept in pairs that is as MPO '\
                        'elements of density matrix thus their bond '\
                        'space is doubled.'
                    )

        if comb_structure is not None and is_measurement:
            raise ValueError(
                'Comb variable cannot be measurement variable.'
            )


    def _contr(self, *args, **kwargs):
        raise ValueError('This is never accessed.')


    def random_choi(self, name: str | None = None) -> ConstTensor:
        """
        Draw random tensor of CPTP Choi matrix.

        Parameters
        ----------
        name : str | None, optional
            Tensor name. If None then self.name, by default None.

        Returns
        -------
        tensor : ConstTensor
            Random tensor.

        """
        if self.bond_spaces:
            raise ValueError(
                'Cannot get random choi for tensor with bond spaces'\
                f' ({self.bond_spaces}). Use random_mps_element instead.'
            )

        if self.is_measurement:
            warnings.warn(
                'To get random measurement (SLD) use random_sld.'
            )

        if self.is_comb:
            warnings.warn(
                'To get random comb use random_comb.'
            )

        if self.is_unital:
            d = self.input_dim
            ps = np.random.rand(d)
            ps /= np.sum(ps)
            unitaries = [unitary_group.rvs(d) for _ in range(d)]
            krauses: list[np.ndarray] = [
                np.sqrt(p) * U for p, U in zip(ps, unitaries)
            ]
            k_vecs = [k.ravel() for k in krauses]
            m = sum(ket_bra(kv, kv) for kv in k_vecs)
        else:
            m = get_random_positive_matrix(
                self.dimension, self.input_dim
            ).astype(my_complex)
            id_in = np.identity(self.input_dim, dtype=my_complex)
            id_out = np.identity(self.output_dim, dtype=my_complex)

            m_in = np.zeros(
                (self.input_dim, self.input_dim), dtype=my_complex
            )
            for ei in id_out:
                tmp = np.kron(ei, id_in)
                m_in += tmp @ m @ tmp.T
            n = np.kron(id_out, np.linalg.inv(sqrtm(m_in)))
            m = n @ m @ n
            m, _ = enhance_hermiticity(m)

        name =  name if name else self.name
        return ConstTensor(
            self.output_spaces + self.input_spaces, choi=m,
            name=name if name else name, sdict=self.sdict,
            output_spaces=self.output_spaces
        )


    def random_comb(self, name: str | None = None) -> ConstTensor:
        """
        Draw random tensor of comb Choi matrix.

        Parameters
        ----------
        name : str | None, optional
            Tensor name. If None then self.name, by default None.

        Returns
        -------
        tensor : ConstTensor
            Random tensor.
        """
        if not self.is_comb:
            raise ValueError(
                'Cannot get random comb for tensor without comb '\
                'structure. Use random_choi instead.'
            )

        sd = self.sdict
        comb_str = self.comb_structure
        _id = uuid4().hex
        last_tooth_out: list[Hashable] = comb_str[-1][1]
        comb_out_dim: int = prod(sd[s] for s in last_tooth_out)
        result = sd.choi_identity()
        anc = [('ANCILLA', i, _id) for i in range(len(comb_str) + 1)]
        for i, (tooth_inp, tooth_out) in enumerate(comb_str):
            d_out = prod(sd[s] for s in tooth_out)
            d_anc = 2
            while d_out * d_anc < comb_out_dim:
                d_anc += 1

            anc_inp = anc[i]
            anc_out = anc[i + 1]
            sd[anc_out] = d_anc

            spaces = tooth_inp + tooth_out
            output_spaces = tooth_out.copy()

            if i > 0:
                spaces.append(anc_inp)
            if i < len(comb_str) - 1:
                spaces.append(anc_out)
                output_spaces.append(anc_out)

            tooth_t = VarTensor(
                spaces, sdict=sd, output_spaces=output_spaces
            ).random_choi()
            result = result * tooth_t

        for space in anc[1:]:
            del sd.spaces[space]

        result.comb_structure = self.comb_structure
        result.name = self.name if name is None else name
        return result


    def random_mps_element(self, name: str | None = None) -> ConstTensor:
        """
        Constant tensor of a density matrix MPO element from the random MPS
        element.

        Parameters
        ----------
        name : str | None, optional
            Tensor name. If None then self.name, by default None.

        Returns
        -------
        tensor : ConstTensor
            Random tensor.
        """
        if self.is_measurement:
            raise ValueError(
                'Cannot get random MPS element for measurement-like '\
                'tensor. Use random_sld instead.'
            )
        sd = self.sdict
        physical_dims = [sd[s] for s in self.physical_spaces]
        mps_bond_dims = [int(np.sqrt(sd[s])) for s in self.bond_spaces]
        m = np.random.rand(
            *physical_dims, *mps_bond_dims
        ).astype(my_complex)
        m += 1j * np.random.rand(*physical_dims, *mps_bond_dims)

        return ConstTensor.from_mps(
            m, self.physical_spaces + self.bond_spaces, sd,
            self.output_spaces, name if name else self.name
        )


    def respace(self: VarTensor, spaces: list[Hashable] | None = None,
        space_map: Callable[[Hashable], Hashable] | None = None,
        sdict: SpaceDict | None = None, name: str | None = None
        ) -> VarTensor:
        """
        Make a copy of self but with renamed spaces.

        Parameters
        ----------
        spaces : list[Hashable] | None, optional
            The change of spaces will take the form
            `self.spaces[i]` -> `spaces[i]`. If None then change of spaces
            will be carried out using `space_map`, by default None.
        space_map : Callable[[Hashable], Hashable] | None, optional
            The change of spaces will take the form
            `space` -> `space_map(space)`, by default None.
        sdict : SpaceDict | None, optional
            Space dictionary of the copy. If None then it will be
            `self.sdict`, by default None.
        name : str | None, optional
            Name of the copy. If None then it will be `self.name`, by
            default None.

        Returns
        -------
        copy : VarTensor
            New tensor with renamed spaces.
        """
        templ = super().respace(spaces, space_map, sdict, name)

        return VarTensor(
            templ.spaces, templ.sdict, templ.name, templ.output_spaces,
            self.is_unital, self.is_measurement, templ.comb_structure
        )


    def random_sld(self, name: str | None = None) -> ConstTensor:
        """
        Constant tensor of random SLD matrix.

        Parameters
        ----------
        name : str | None, optional
            Tensor name. If None then self.name, by default None.

        Returns
        -------
        tensor : ConstTensor
            Random SLD matrix.
        """
        if not self.is_measurement:
            warnings.warn(
                'To get random choi/mps use random_choi/'\
                'random_mps_element instead.'
            )
        arr = np.random.random(self.shape)
        sld = ConstTensor(
            self.spaces, array=arr, name=name if name else self.name,
            sdict=self.sdict
        )
        arr_hc = sld.choi_T(*self.physical_spaces).array.conjugate()
        sld.array = (sld.array + arr_hc) / 2
        return sld




class TensorNetwork(GeneralTensor):
    """
    Class of tensor network.

    Parameters
    ----------
    tensors : list[GeneralTensor | Scalar] | None, optional
        Nodes of the network (or other network), by default None.
    sdict : SpaceDict, optional
        Space dictionary, by default ``DEFAULT_SDICT``.
    name : str | None, optional
        Network name, by default None.

    Attributes
    ----------
    tensors : dict[str, ConstTensor | ParamTensor | VarTensor]
        Dictionary of tensors in the network where keys are tensor names
        and values are tensors.
    edges : dict[Hashable, list[str]]
        Dictionary of edges where keys are space names and values
        are lists of tensor names connected by the edge.
    contr_spaces : set[Hashable]
        Set of spaces that connect two tensors (i.e. are contracted) and
        as such are not visible outside the network.
    free_spaces : set[Hashable]
        Set of spaces such that only one tensor is connected to them
        (i.e. are not contracted).
    free_dimension : int
        Dimension of the free space (product of dimensions of free spaces).
    multiplier : complex
        Multiplier resulting from contraction with trivial tensors
        (scalars).
    """
    name_prefix = 'TENSOR NETWORK '


    def __init__(self, tensors: list[GeneralTensor | Scalar] | None = None,
        sdict: SpaceDict = DEFAULT_SDICT, name : str | None = None):
        """
        Class of tensor network.

        Parameters
        ----------
        tensors : list[GeneralTensor | Scalar] | None, optional
            Nodes of the network (or other network), by default None.
        sdict : SpaceDict, optional
            Space dictionary, by default ``DEFAULT_SDICT``.
        name : str | None, optional
            Network name, by default None.

        """
        self.edges: dict[Hashable, list[str]] = {}
        self._spaces = set() # All spaces used in this network.
        self.contr_spaces = set() # Contracted spaces.
        self.free_spaces = set() # Free spaces.

        super().__init__(
            list(self.free_spaces), sdict=sdict, name=name,
            output_spaces=[]
        )

        self.multiplier = 1 + 0j
        self.tensors: dict[str, ConstTensor | ParamTensor | VarTensor] = {}
        if tensors is None:
            tensors = []

        for tensor in tensors:
            if isinstance(tensor, (int, float, complex)):
                self.multiplier *= tensor
            elif tensor.sdict is not self.sdict:
                raise ValueError(
                    'Contraction is possible only between tensors with '\
                    'the same space dictionary.\n'\
                    f'Tried to multiply matrix on {sdict.name} '\
                    f'with matrix on {tensor.sdict.name}.')
            elif (
                isinstance(tensor, ConstTensor) 
                and len(tensor.spaces) == 0
            ):
                self.multiplier *= tensor.array[0]
            elif isinstance(tensor, TensorNetwork):
                x = self.contr_spaces.intersection(tensor._spaces)
                y = self._spaces.intersection(tensor.contr_spaces)
                if x or y:
                    raise ValueError(
                        f'Too many contractions of {list(x.union(y))}.'
                    )

                self._spaces.update(tensor.spaces)
                contr = self.free_spaces.intersection(tensor.free_spaces)
                self.contr_spaces.update(contr, tensor.contr_spaces)
                self.free_spaces.symmetric_difference_update(
                    tensor.free_spaces
                )

                for _tensor in tensor.tensors.values():
                    if _tensor.name in self.tensors:
                        raise ValueError(
                            f'Repeated tensor identifier {_tensor.name}.'
                        )

                    self.tensors[_tensor.name] = _tensor

                for space in tensor.edges:
                    if space not in self.edges:
                        self.edges[space] = []
                    self.edges[space] += tensor.edges[space]
            else:
                if tensor.name in self.tensors:
                    raise ValueError(
                        f'Repeated tensor identifier {tensor.name}.'
                    )
                self.tensors[tensor.name] = tensor
                self._spaces.update(tensor.spaces)
                contr = self.free_spaces.intersection(tensor.spaces)
                self.contr_spaces.update(contr)
                self.free_spaces.symmetric_difference_update(tensor.spaces)

                for space in tensor.spaces:
                    if space not in self.edges:
                        self.edges[space] = []
                    self.edges[space].append(tensor.name)

        for space, space_tensors in self.edges.items():
            if len(space_tensors) > 2:
                raise ValueError(
                    'More then two tensors defined on one space during '\
                    f'contraction. Space {space}, tensors: '\
                    f'{space_tensors}.'
                )

        self.free_dimension = prod(self.sdict[s] for s in self.free_spaces)


    def _contr(self, *others):
        raise ValueError('This is never accessed.')


    @property
    def spaces(self) -> list[Hashable]:
        return list(self._spaces)


    @spaces.setter
    def spaces(self, spaces: list[Hashable]):
        pass


    def neighbors(self, name: str) -> set[str]:
        """
        Get set of neighbors.

        Parameters
        ----------
        name : str
            Node name.

        Returns
        -------
        neighbors : set[str]
            Set of neighbors' names.
        """
        ns = set()
        tensor0 = self.tensors[name]
        for space in tensor0.spaces:
            ns.update(self.edges[space])
        ns.remove(tensor0.name)
        return ns


    def remove(self, names: list[str]):
        """
        Remove tensors from the network.

        Parameters
        ----------
        names : list[str]
            Names of the tensors to remove.
        """
        for name in names:
            tensor = self.tensors[name]
            contr_spaces = []
            free_spaces = []
            for space in tensor.spaces:
                if space in self.contr_spaces:
                    contr_spaces.append(space)
                else: free_spaces.append(space)

            self._spaces.difference_update(free_spaces)
            self.free_spaces.difference_update(free_spaces)

            for space in contr_spaces:
                self.edges[space] = [
                    _name for _name in self.edges[space] if _name != name
                ]


    def compress(self, name: str | None = None,
        ignore: list[str] | None = None) -> TensorNetwork:
        """
        Make those contraction of constant nodes that do not increase
        used memory space.

        Parameters
        ----------
        name : str | None, optional
            Name of the new tensor network, by default None.
        ignore : list[str] | None, optional
            Names of tensors that cannot be contracted, by default None.

        Returns
        -------
        TensorNetwork
            Compressed tensor network.
        """
        if ignore is None:
            ignore = []

        removed = {
            _name for _name, choi in self.tensors.items()
            if isinstance(choi, VarTensor) or _name in ignore
        }
        components = self.connected_components(removed=removed)

        tensors = []
        for component in components:
            tensor0, *rest = [self.tensors[_name] for _name in component]
            if isinstance(tensor0, (ConstTensor, ParamTensor)):
                tensors.append(tensor0.contr(*rest))
            else:
                tensors.append(tensor0)

        return TensorNetwork(
            tensors=tensors, name=name if name is not None
            else self.name, sdict=self.sdict
        )


    def connected_components(self, subgraph: set[str] | None = None,
        removed: set[str] | None = None
        ) -> list[list[str]]:
        """
        Get list of connected components.

        Parameters
        ----------
        subgraph : set[str] | None, optional
            Nodes defining subgraph for which the computation is performed,
            by default None.
        removed : set[str] | None, optional
            Nodes assumed to be removed from the network, by default None.

        Returns
        -------
        components : list[list[str]]
            List of lists of names.
        """
        if subgraph is not None:
            _subgraph = subgraph
        else:
            _subgraph = set(self.tensors.keys())
        _removed: set[str] = removed if removed is not None else set()
        names = {name for name in self.tensors if name in _subgraph}

        checked = {name: False for name in names}
        comps = []
        while names:
            name = names.pop()
            checked[name] = True
            comp = [name]

            if name not in _removed:
                to_check = set(self.neighbors(name))
                while to_check:
                    _name = to_check.pop()
                    if (
                        _name not in _subgraph
                        or checked[_name]
                        or _name in _removed
                    ):
                        continue

                    checked[_name] = True
                    comp.append(_name)
                    to_check.update(self.neighbors(_name))

            comps.append(comp)
            names.difference_update(comp)
        return comps


    def respace(self: TensorNetwork, spaces: list[Hashable] | None = None,
        space_map: Callable[[Hashable], Hashable] | None = None,
        qsystem: SpaceDict | None = None, name: str | None = None
        ) -> TensorNetwork:
        templ = super().respace(spaces, space_map, qsystem, name)
        raise NotImplementedError()


    def plot(self, **kwargs: Any):
        """
        Plot the network using matplotlib.pyplot.
        """
        graph = nx.Graph()

        for name in self.tensors:
            graph.add_node(name)

        free = set()
        edge_labels = {}
        for space, space_tensors in self.edges.items():
            if len(space_tensors) == 2:
                t0, t1 = space_tensors
            else: #len(space_tensors) == 1
                t0 = space_tensors[0]
                t1 = f'FREE {space}'
                free.add(t1)
                graph.add_node(t1)

            graph.add_edge(t0, t1)
            edge_labels[(t0, t1)] = space

        layers = {}
        q = deque()
        for node in graph.nodes:
            if node in free:
                continue
            t = self.tensors[node]
            if not t.output_spaces:
                layers[node] = 0
                q.extend(zip(repeat(node), self.neighbors(node)))

        while q:
            origin, node = q.popleft()
            if node in layers:
                layers[node] = min(layers[node], layers[origin] + 1)
            else:
                layers[node] = layers[origin] + 1
                for neighbor in graph.neighbors(node):
                    if neighbor not in layers:
                        q.append((node, neighbor))

        for node in graph.nodes:
            graph.nodes[node]['layer'] = -layers[node]

        pos = nx.multipartite_layout(graph, subset_key='layer')

        nx.draw(
            graph, pos, labels={node: node for node in graph.nodes},
            **kwargs
        )
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_labels, rotate=False
        )

    
    def copy(self) -> TensorNetwork:
        """
        Make a copy.

        Returns
        -------
        copy : TensorNetwork
            Tensor network copy.
        """
        ts = [t.copy() for t in self.tensors.values()]
        new = TensorNetwork(ts, self.sdict, self.name)
        return new




class ParamTensor(ConstTensor):
    """
    Class of parametrised tensor. It is a const tensor with additional
    attribute ``dtensor`` which represents its derivative in the parameter
    at the point.

    Parameters
    ----------
    spaces : list[Hashable]
        Tensor spaces.
    choi_matrix : np.ndarray | None, optional
        Choi matrix which from which the tensor will be constructed. If
        None then it will be initialized from the array argument, by
        default None.
    sdict : SpaceDict, optional
        Space dictionary, by default ``DEFAULT_SDICT``.
    array : np.ndarray | None, optional
        Tensor as numpy array, by default None.
    name : str | None, optional
        Tensor name, by default None.
    dchoi_matrix : np.ndarray | None, optional
        Choi matrix which from which the dtensor will be constructed. If
        None then it will be initialized from the darray argument, by
        default None.
    darray : np.ndarray | None, optional
        Derivative as numpy array, by default None.
    output_spaces : list[Hashable] | None, optional
        Tensor output spaces, by default None.
    comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        Comb (causal) structure in a form of
        ``[(input_0, output_0), (input_1, output_1), ...]``
        where ``input_i`` (``output_i``) is a list of input (output) spaces
        of the i-th tooth.
    
    Attributes
    ----------
    dtensor : ConstTensor
        Tensor representing the derivative.
    """
    name_prefix = 'PARAM TENSOR '


    def __init__(self, spaces: list[Hashable],
        choi: np.ndarray | None = None, sdict: SpaceDict = DEFAULT_SDICT,
        array: np.ndarray | None = None, name: str | None = None,
        dchoi: np.ndarray | None = None, darray: np.ndarray | None = None,
        output_spaces: list[Hashable] | None = None,
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        = None):
        """
        Class of parametrised tensor. It is a const tensor with additional
        attribute 'dtensor' which represents its derivative over the
        paramter in 0.

        Parameters
        ----------
        spaces[0...*] : Hashable
            Tensor spaces.
        choi : np.ndarray | None, optional
            Choi matrix which from which the tensor will be constructed. If
            None then it will be initialized from the array argument, by
            default None.
        sdict : SpaceDict, optional
            Space dictionary, by default DEFAULT_SDICT.
        array : np.ndarray | None, optional
            Tensor as numpy array, by default None.
        name : str | None, optional
            Tensor name, by default None.
        dchoi : np.ndarray | None, optional
            Choi matrix which from which the dtensor will be constructed.
            If None then it will be initialized from the darray argument,
            by default None.
        darray : np.ndarray | None, optional
            Derivative as numpy array, by default None.
        output_spaces : list[Hashable] | None, optional
            Tensor output spaces, by default None.
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
            Comb (causal) structure in a form of [(input_0, output_0),
            (input_1, output_1), ...] where input_i (output_i) is a list
            of input (output) spaces of the i-th tooth.
        """
        super().__init__(
            spaces, choi=choi, sdict=sdict, array=array,
            name=name, output_spaces=output_spaces,
            comb_structure=comb_structure
        )
        self.dtensor: ConstTensor = ConstTensor(
            spaces, choi=dchoi, sdict=sdict, array=darray,
            name=f'{self.name} derivative', output_spaces=output_spaces,
            comb_structure=comb_structure
        )


    @staticmethod
    def from_const(tensor: ConstTensor, name: str | None = None
        ) -> ParamTensor:
        """
        Get a parametrized tensor from a constant tensor. The derivative is
        assumed to be 0.

        Parameters
        ----------
        tensor : ConstTensor
            Conastant tensor.
        name : str | None, optional
            Tensor name. If None then tensor.name, by default None.

        Returns
        -------
        tensor : ParamTensor
            The same tensor but with defined derivative.
        """
        return ParamTensor(
            tensor.spaces, sdict=tensor.sdict, array=tensor.array,
            name=name if name is not None else tensor.name,
            output_spaces=tensor.output_spaces,
            comb_structure=tensor.comb_structure
        )


    @staticmethod
    def to_const(tensor: ParamTensor | ConstTensor,
        name: str | None = None) -> ConstTensor:
        """
        Discards the derivative and returns the constant tensor.

        Parameters
        ----------
        tensor : ParamTensor | ConstTensor
            Parametrized or constant tensor.
        name : str | None, optional
            Tensor name. If None then tensor.name, by default None.

        Returns
        -------
        tensor : ConstTensor
            Tensor without the derivative.
        """
        return ConstTensor(
            tensor.spaces, sdict=tensor.sdict, array=tensor.array,
            name=name if name is not None else tensor.name,
            output_spaces=tensor.output_spaces,
            comb_structure=tensor.comb_structure
        )


    def _contr(self, *others: ConstTensor | ParamTensor) -> ParamTensor:
        new = self.copy()
        # TODO: Probably, there exists a better ordering + docstr
        for tensor in others:
            new_val = super(ParamTensor, new)._contr(tensor)

            dnew = new.dtensor._contr(self.to_const(tensor))
            if (
                isinstance(tensor, ParamTensor)
                and np.any(tensor.dtensor.array)
            ):
                dnew += tensor.dtensor._contr(self.to_const(new))

            new = self.from_const(new_val)
            new.dtensor = dnew

        return new


    def __add__(self, other: ConstTensor | ParamTensor):
        new = self.from_const(super().__add__(other))
        dother = self.sdict.zero(other.spaces)
        if isinstance(other, ParamTensor):
            dother = other.dtensor
        new.dtensor = self.dtensor + dother
        return new
    

    __radd__ = __add__


    def __iadd__(self, other: ConstTensor | ParamTensor):
        new = self.from_const(super().__iadd__(other), name=self.name)
        if isinstance(other, ParamTensor):
            new.dtensor += other.dtensor
            return new
        return new


    def respace(self, spaces: list[Hashable] | None = None,
        space_map: Callable[[Hashable], Hashable] | None = None,
        sdict: SpaceDict | None = None,
        name: str | None = None, make_copy: bool = False
        ) -> ParamTensor:
        """
        Make a copy of self but with renamed spaces.

        Parameters
        ----------
        spaces : list[Hashable] | None, optional
            The change of spaces will take the form
            self.spaces[i] -> spaces[i]. If None then change of spaces
            will be carried out using space_map, by default None.
        space_map : Callable[[Hashable], Hashable] | None, optional
            The change of spaces will take the form
            space -> space_map(space), by default None
        sdict : SpaceDict | None, optional
            Space dictionary of the copy. If None then it will be
            self.sdict, by default None.
        name : str | None, optional
            Name of the copy. If None then it will be self.name, by
            default None.
        make_copy : bool, optional
            Whether to make a copy of `self.array` and
            `self.dtensor.array`, by default False.

        Returns
        -------
        copy : ParamTensor
            New tensor with renamed spaces.

        """
        args = (spaces, space_map, sdict, name, make_copy)
        new = self.from_const(super().respace(*args))
        new.dtensor = self.dtensor.respace(*args)
        return new


    def choi_transpose(self, *spaces: Hashable, full: bool = False
        ) -> ParamTensor:
        """
        Computes partial transposition of Choi-like tensor and for other
        tensors it does transpose-like reshuffling of entries on
        physcial spaces.

        Parameters
        ----------
        full : bool, optional
            If True tranposes all spaces, by default False.

        Returns
        -------
        transpostion: ParamTensor
            Tensor of the transposed matrix.
        """
        raise NotImplementedError


    def __str__(self):
        result = super().__str__()
        result += f'\ndtensor:\n {self.dtensor.array}'
        return result


    def __eq__(self, other: ConstTensor | ParamTensor) -> bool:
        _eq = super().__eq__(other)
        if isinstance(other, ParamTensor):
            return _eq and self.dtensor == other.dtensor
        return bool(_eq and np.all(self.dtensor.array == 0))


    def copy(self) -> ParamTensor:
        """
        Make a copy.

        Returns
        -------
        copy : ParamTensor
            Tensor copy.
        """
        new = copy.copy(self)
        new.sdict = self.sdict
        new.array = copy.deepcopy(self.array)
        new.dtensor = self.dtensor.copy()
        return new


    def square_without(self, spaces: list[Hashable]) -> ParamTensor:
        raise NotImplementedError


    @property
    def output_spaces(self) -> list[Hashable]:
        return super().output_spaces


    @output_spaces.setter
    def output_spaces(self, output_spaces: list[Hashable]):
        ConstTensor.output_spaces.__set__(self, output_spaces)
        self.dtensor.output_spaces = output_spaces


    @property
    def input_spaces(self) -> list[Hashable]:
        return super().input_spaces


    @input_spaces.setter
    def input_spaces(self, input_spaces: list[Hashable]):
        ConstTensor.input_spaces.__set__(self, input_spaces)
        self.dtensor.input_spaces = input_spaces


    @property
    def comb_structure(self
        ) -> list[tuple[list[Hashable], list[Hashable]]] | None:
        return super().comb_structure


    @comb_structure.setter
    def comb_structure(self,
        comb_structure: list[tuple[list[Hashable], list[Hashable]]] | None
        ):
        super().comb_structure = comb_structure
        self.dtensor.comb_structure = comb_structure


    def dkrauses(self, input_spaces: list[Hashable] | None = None,
        output_spaces: list[Hashable] | None = None) -> tuple[
            list[np.ndarray], list[np.ndarray]
        ]:
        """
        Computes Kraus operatiors of the Choi matrix of the Choi-like
        tensor.

        Parameters
        ----------
        input_spaces : list[Hashable] | None, optional
            Input spaces. If None then self.input_spaces, by default None.
        output_spaces : list[Hashable] | None, optional
            Output spaces. If None then self.output_spaces, by default
            None.

        Returns
        -------
        krauses : list[np.ndarray]
            List of Kraus operators.
        dkrause L list[np.ndarray]
            List of derivatives of Kraus operators.
        """
        if not self.is_choi_like:
            raise ValueError(
                'Kraus operators can be obtained only for Choi-like'\
                'tensors, that is tensors without bond spaces ('\
                f'{self.bond_spaces}).'
            )
        _input = input_spaces if input_spaces else self.input_spaces
        output = output_spaces if output_spaces else self.output_spaces
        rest = set(self.spaces).difference(_input).difference(output)
        chm: ParamTensor = self.choi_trace(*rest)

        inp_dim = prod(self.sdict[space] for space in _input)
        out_dim = prod(self.sdict[space] for space in output)

        matrix = chm.choi(output + _input)
        dmatrix = chm.dtensor.choi(output + _input)

        return dkrauses_from_choi(matrix, dmatrix, (inp_dim, out_dim))


    def dchoi(self, spaces: list[Hashable]) -> np.ndarray:
        """
        Compute derivative of Choi matrix for the Choi-like tensor.

        Parameters
        ----------
        spaces : list[Hashable]
            List of spaces defining the order of spaces of the result.

        Returns
        -------
        matrix : np.ndarray
            Derivative of Choi matrix.

        """
        if not self.is_choi_like:
            raise ValueError(
                'Choi matrix is defined only for tensors without '\
                f'bond spaces ({self.bond_spaces}).'
            )
        if (set(spaces) != set(self.spaces)
            or len(spaces) != len(self.spaces)):
            raise ValueError(
                f'Tried to get matrix on {spaces} for tensor on '\
                f' {self.spaces}.'
            )
        if spaces:
            self.reorder(spaces)
        return self._tensor_arr_to_choi(
            self.dtensor.array, list(self.dimensions)
        )


    def reorder(self, new_spaces: list[Hashable]) -> ParamTensor:
        """
        Change the order of spaces.

        Parameters
        ----------
        new_spaces : list[Hashable]
            Spaces names in the new order.

        Returns
        -------
        self : ParamTensor
            self
        """
        super().reorder(new_spaces)
        self.dtensor.reorder(new_spaces)
        return self




Tensor  = Union[GeneralTensor, ParamTensor, VarTensor, TensorNetwork]


my_complex = np.complex128
