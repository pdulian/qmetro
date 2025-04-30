from collections.abc import Hashable
from itertools import chain
from math import prod
from typing import Any, Callable

import numpy as np

from ...qtools import choi_from_krauses, dchoi_from_krauses

from ..classes import SpaceDict, ConstTensor, ParamTensor, DEFAULT_SDICT




def choi_identity(spaces: list[Hashable] | None = None,
    sdict: SpaceDict = DEFAULT_SDICT, **kwargs: Any) -> ConstTensor:
    """
    Creates tensor form of identity Choi matrix.

    Parameters
    ----------
    spaces : list[Hashable]
        Tensor spaces, by default empty list.
    sdict : SpaceDict, optional
        Space dictionary, by default DEFAULT_SDICT
    **kwargs :
        Key-word arguments passed to ConstTensor constructor.

    Returns
    -------
    tensor : ConstTensor
        Choi-like constant tensor of identity matrix.
    """
    return sdict.choi_identity(spaces, **kwargs)


def zero(spaces: list[Hashable] | None = None,
    sdict: SpaceDict = DEFAULT_SDICT, **kwargs: Any) -> ConstTensor:
    """
    Creates constant tensor filled with zeros with sdict=self.

    Parameters
    ----------
    spaces : list[Hashable]
        Tensor spaces, by default empty list.
    sdict : SpaceDict, optional
        Space dictionary, by default DEFAULT_SDICT
    **kwargs :
        Key-word arguments passed to ConstTensor constructor.

    Returns
    -------
    tensor : ConstTensor
        Zero const tensor.
    """
    return sdict.zero(spaces, **kwargs)


def const_tensor_from_fun(fun: Callable[[np.ndarray], np.ndarray],
    input_spaces: list[Hashable], output_spaces: list[Hashable],
    sdict: SpaceDict = DEFAULT_SDICT, **kwargs: Any) -> ConstTensor:
    """
    Returns a tensor form of Choi matrix created from a quantum channel
    using Choi-Jemiolkowski isomorphism.


    Parameters
    ----------
    fun : Callable[[np.ndarray], np.ndarray]
        Function defining the quantum channel.
    input_spaces : list[Hashable]
        Input spaces.
    output_spaces : list[Hashable]
        Output spaces.
    sdict : SpaceDict, optional
        Space dictionary, by default DEFAULT_SDICT.
    **kwargs :
        Key-word arguments passed to ConstTensor constructor.

    Returns
    -------
    tensor : ConstTensor
        Tensor form of the Choi matrix.
    """
    for space in chain(input_spaces, output_spaces):
        if space in sdict.bond_spaces:
            raise ValueError(
                'Choi matrix can act on physical spaces only but '\
                f'{input_spaces + output_spaces} was provided.'
            )
    input_dim = prod(sdict[space] for space in input_spaces)
    output_dim = prod(sdict[space] for space in output_spaces)
    choi_dim = input_dim * output_dim

    result_matrix = np.zeros((choi_dim, choi_dim), dtype=np.complex128)
    for i in range(input_dim):
        for j in range(input_dim):
            eij = np.zeros((input_dim, input_dim), dtype=np.complex128)
            eij[i, j] = 1
            result_matrix += np.kron(fun(eij), eij)

    return ConstTensor(
        output_spaces + input_spaces, choi=result_matrix,
        output_spaces=output_spaces, sdict=sdict, **kwargs
    )


def tensor_from_krauses(krauses: list[np.ndarray],
    input_spaces: list[Hashable], output_spaces: list[Hashable],
    sdict: SpaceDict = DEFAULT_SDICT,
    dkrauses: list[np.ndarray] | None = None, **kwargs: Any
    ) -> ConstTensor | ParamTensor:
    """
    Computes tensor form of a Choi matrix from Kraus operators
    and their derivatives.

    Parameters
    ----------
    krauses : list[np.ndarray]
        Kraus operators.
    input_spaces : list[Hashable]
        Input spaces.
    output_spaces : list[Hashable]
        Output spaces.
    sdict : SpaceDict, optional
        Space dictionary of the result, by default DEFAULT_SDICT.
    dkrauses : list[np.ndarray] | None, optional
        Derivatives of Kraus operators. If provided the result will be
        a ParamTensor with derivative coming from these operators,
        by default None.
    **kwargs :
        Key-word arguments passed to ConstTensor/ParamTensor constructor.

    Returns
    -------
    matrix : ConstTensor | ParamTensor
        Tensor form of the Choi matrix.
    """
    C = choi_from_krauses(krauses)

    if dkrauses is None:
        return ConstTensor(
            output_spaces + input_spaces, choi=C, sdict=sdict,
            output_spaces=output_spaces, **kwargs
        )

    dC = dchoi_from_krauses(krauses, dkrauses)
    return ParamTensor(
        output_spaces + input_spaces, choi=C, dchoi=dC,
        sdict=sdict, output_spaces=output_spaces, **kwargs
    )
