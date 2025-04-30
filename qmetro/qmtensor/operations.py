from __future__ import annotations

from collections.abc import Hashable
from typing import get_args

import numpy as np

from .classes import (
    ConstTensor, ParamTensor, VarTensor, Tensor, Scalar
)




def Tr(tensor: Tensor, *spaces: Hashable, full: bool=False
    ) -> Tensor | Scalar:
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
    return tensor.choi_trace(*spaces, full=full)


def kron(*args: Tensor | Scalar) -> Tensor:
    """
    For Choi-like tensors it computes their Kronecker product and for
    other tensors it is just contraction where there are no doubled
    indices (no gets contracted).

    Parameters
    ----------
    args[0...*] : Hashable
        Tensors to be Kronecker multiplied.

    Returns
    -------
    tensor : Tensor
        Kronecker product.
    """
    x = 1
    for i, arg in enumerate(args):
        if isinstance(arg, (int, float, complex)):
            x *= arg
        elif isinstance(arg, get_args(Tensor)):
            return arg.kron(*args[i + 1:], x)
        else:
            raise ValueError(f'Unsupported type: {type(arg)}')
    raise ValueError('Empty list given.')


def contr(*args: Tensor | Scalar) -> Tensor:
    """
    Contract tensors.

    Parameters
    ----------
    others[0...*] : Tensor | Scalar
        Tensors to be contracted.

    Returns
    -------
    tensor : Tensor
        Contraction result.
    """
    x = 1
    for i, arg in enumerate(args):
        if isinstance(arg, (int, float, complex)):
            x *= arg
        elif isinstance(arg, get_args(Tensor)):
            return arg.contr(*args[i + 1:], x)
        else:
            raise ValueError(f'Unsupported type: {arg}')
    raise ValueError('Empty list given.')


def is_scalar(x: Tensor | Scalar) -> bool:
    if isinstance(x, get_args(Scalar)):
        return True

    if isinstance(x, ParamTensor):
        return bool(
            len(x.spaces) == 0 and np.all(x.dtensor.choi([]) == 0)
        )

    if isinstance(x, ConstTensor):
        return len(x.spaces) == 0

    return False


def is_var(x : Tensor | Scalar) -> bool:
    return isinstance(x, VarTensor)


def is_cptp_var(x : Tensor | Scalar) -> bool:
    return is_var(x) and not x.bond_spaces and not x.is_measurement


def is_mps_var(x : Tensor | Scalar) -> bool:
    return bool(is_var(x) and x.bond_spaces and not x.is_measurement)


def is_comb_var(x : Tensor | Scalar) -> bool:
    return bool(is_var(x) and x.is_comb)


def is_mps(x : Tensor | Scalar) -> bool:
    return bool(
        is_mps_var(x)
        or (
            isinstance(x, ConstTensor)
            and x.bond_spaces
            and not x.input_spaces
        )
    )


def is_param(x : Tensor | Scalar) -> bool:
    return isinstance(x, ParamTensor)


def is_measurement(x: Tensor | Scalar) -> bool:
    return is_var(x) and x.is_measurement


def is_sld_mpo(x: Tensor | Scalar) -> bool:
    return bool(is_measurement(x) and x.bond_spaces)


def is_full_sld(x: Tensor | Scalar) -> bool:
    return bool(is_measurement(x) and not x.bond_spaces)


def is_const(x: Tensor) -> bool:
    return isinstance(x, ConstTensor)
