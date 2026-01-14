from __future__ import annotations

import numpy as np

from ..qmtensor import ConstTensor




class MaxIterExceededError(Exception):
    """
    Exception raised when the maximal number of iterations is exceeded.

    Parameters
    ----------
    max_iter : int
        Maximal iterations allowed.
    qfi : float
        Quantum Fisher Information value at the moment of raising
        the error.
    qfis : list[float]
        List of Quantum Fisher Information values from all previous
        iterations.
    """    
    def __init__(self, max_iter: int, qfi: float, qfis: list[float],
        *args: object) -> None:
        self.max_iter = max_iter
        self.qfi = qfi
        self.qfis = qfis
        self.cause = f'Maximal ({max_iter}) number of iterations exceeded.'
        self.iteration = max_iter
        super().__init__(
            f'Maximal ({max_iter}) number of iterations exceeded.', *args
        )


class SolverError(Exception):
    """
    Exception raised when the solver for optimizing a tensor fails.

    Parameters
    ----------
    name : str
        Name of the tensor being optimized.
    cause : Exception | str
        Exception or message indicating the cause of the error.
    m0 : ConstTensor
        Term of the pre-QFI linear in pre-SLD.
    m1 : ConstTensor
        Term of the pre-QFI quadratic in pre-SLD.
    """
    def __init__(self, name: str, cause: Exception | str, m0: ConstTensor,
        m1: ConstTensor, *_: object) -> None:
        self.name = name
        self.cause = cause
        self.m0 = m0
        self.m1 = m1
        super().__init__(
            f'Solver error for optimization of {name}:\nCause: {cause}\n'\
            f'm0 =\n{m0}\nm1 =\n{m1}'
        )


class NonHermitianError(Exception):
    """
    Exception raised when the optimized tensor is non-hermitian.

    Parameters
    ----------
    name : str
        Name of the tensor being optimized.
    nonhermiticity : float
        Measure of non-hermiticity of the optimized tensor.
        See
        :func:`enhance_hermiticity <qmetro.utils.enhance_hermiticity>`.
    """
    def __init__(self, name: str, nonhermiticity: float, *_) -> None:
        self.name = name
        self.nonhermiticity = nonhermiticity
        super().__init__(
            f'Non-hermitian parameter for optimization of {name}.'\
            f' Nonhermiticity: {nonhermiticity}'
        )


class SingleIterError(Exception):
    """
    Exception raised when an iteration of the ISS algorithm fails.

    Parameters
    ----------
    cause : Exception
        Cause of the error.
    iteration : int
        Iteration number at which the error occurred.
    qfi : float
        Quantum Fisher Information value at the moment of raising
        the error.
    qfis : list[float]
        List of Quantum Fisher Information values from all previous
        iterations.
    """    
    def __init__(self, cause: Exception, iteration: int, qfi: float,
        qfis: list[float], *_) -> None:
        self.cause = cause
        self.iteration = iteration
        self.qfi = qfi
        self.qfis = qfis
        super().__init__(
            f'Iteration number: {iteration} failed at QFI = {qfi}. {cause}'
        )


class NormMatZeroEigenval(Exception):
    """
    Exception raised when a near-zero eigenvalue of the norm matrix
    is encountered during the MPS optimization :cite:`Chabuda2020`.

    Parameters
    ----------
    eigval : float
        The near-zero eigenvalue.
    """    
    def __init__(self, eigval: float, *_) -> None:
        self.eigval = eigval
        super().__init__(
            f'Encountered near-zero ({eigval}) eigenvalue of norm matrix '
            'in MPS optimize.'
        )
