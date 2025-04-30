from __future__ import annotations

import numpy as np

from ..qmtensor import ConstTensor




class MaxIterExceededError(Exception):
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
    def __init__(self, name: str, nonhermiticity: float, *_) -> None:
        self.name = name
        self.nonhermiticity = nonhermiticity
        super().__init__(
            f'Non-hermitian parameter for optimization of {name}.'\
            f' Nonhermiticity: {nonhermiticity}'
        )


class SingleIterError(Exception):
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
    def __init__(self, eigval: float, *_) -> None:
        self.eigval = eigval
        super().__init__(
            f'Encountered near-zero ({eigval}) eigenvalue of norm matrix '
            'in MPS optimize.'
        )
