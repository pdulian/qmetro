from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from typing import TypeVar, Callable, Union

import numpy as np
import cvxpy as cp
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.stats import unitary_group

from ncon import ncon

from .consts import LINE_WIDTH




S = TypeVar("S")
T = TypeVar("T")




def matrix_exp_derivative(x: NDArray[np.complex128],
                          y: NDArray[np.complex128],
                          t0: float = 0.0) -> NDArray[np.complex128]:
    """Compute derivative of the function:

        t -> exp(x + t * y),

    where x and y are matrices. Derivative is computed at point t0.

    Args:
        x (NDArray[np.complex128]): matrix
        y (NDArray[np.complex128]): matrix
        t0 (float): Point at which derivative will be computed.
        Defaults to 0.0.

    Returns:
        np.ndarray: Derivative.
    """
    x = x + t0 * y
    dim = x.shape[0]
    zeros = np.zeros((dim, dim))  
    M = np.block([[x, y], [zeros, x]])
    exp_blocks = expm(M)

    return exp_blocks[:dim, -dim:]


def get_random_hermitian_matrix(d: int) -> np.ndarray:
    M = np.random.random((d, d)) + 1j * np.random.random((d, d))
    # M = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    return (M + np.conj(M.T)) / np.sqrt(4 * d)


def get_random_positive_matrix(d: int, trace: float = 1.0) -> np.ndarray:
    """Returns random positive matrix with a given trace.

    Args:
        d (int): Matrix dimension.
        trace (float): Trace.

    Returns:
        np.ndarray: Random density matrix.
    """
    rho = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    rho = rho @ rho.conjugate().T
    return rho * (trace/np.trace(rho))


def get_random_den_mat(d: int) -> np.ndarray:
    """Returns random positive matrix with trace one.

    Args:
        d (int): Matrix dimension.

    Returns:
        np.ndarray: Random density matrix.
    """
    return get_random_positive_matrix(d)


def get_random_pure_state(d: int) -> np.ndarray:
    if d == 2:
        theta = np.arccos(np.random.uniform(-1, 1))
        phi = np.random.uniform(0, 2 * np.pi)
        return np.array([
            np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)
        ])

    v = np.random.randn(d) + 1j * np.random.randn(d)
    return v / np.linalg.norm(v)


def hc(x: np.ndarray | cp.Expression) -> np.ndarray | cp.Expression:
    return x.conjugate().T


def ket_bra(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.kron(x.reshape(len(x), 1), y.conjugate())


def schmidt(state: np.ndarray, dims: list[int], eps: float = 1e-5
    ) -> tuple[list, list[list[np.ndarray]]]:
    """
    Gives Schmidt decomposition of the state.

    It returns tuple of lists (a, s) such that:

        | state > = sum_i a_i |s[i][0]> (x) ... (x) |s[i][r - 1]>,
    
    where r is the number of spaces in the decompostion and
    <s[i][k]|s[j][k]> = int(i == k).

    Parameters
    ----------                                                 
        state : np.ndarray
            One-dimensional (len(state.shape) == 1) array representing
            the state.
        dims : list[int]
            Dimensions of the spaces in the decomposition.
        eps : float
            Cut-off for small Schmidt coefficients (singular values).

    Returns
    -------
        a : list[float]
            Amplitudes.
        s : list[list[np.ndarray]]
            States.
    """
    if len(state.shape) != 1:
        raise ValueError(
            'In Schmidt decomposition state argument has to be '
            f'one-dimensional but got state.shape = {state.shape}.'
        )

    if len(state) != np.prod(dims):
        raise ValueError(
            'In Schmidt decomposition state has to have length equal '
            f'to the product of dims but got len(state) = {len(state)}'
            f' and np.prod(dims) = {np.prod(dims)}.'
        )

    if len(dims) == 2:
        x = state.reshape(dims)
        u, svs, v = np.linalg.svd(x)

        a, s = [], []
        for i, sv in enumerate(svs):
            if sv < eps:
                continue

            a.append(sv)
            s.append([u[:, i], v[i]])

        return a, s

    a0, s0 = schmidt(state, [dims[0], np.prod(dims[1:])], eps)
    a, s = [], []
    for i, a0i in enumerate(a0):
        a1, s1 = schmidt(s0[i][1], dims[1:], eps)
        for j, a1j in enumerate(a1):
            a.append(a0i * a1j)
            s.append([s0[i][0], *s1[j]])

    return a, s


def kron(*vs):
    result = 1 + 0j
    for v in vs:
        result = np.kron(result, v)
    return result


def fst(x: tuple[T, ...]) -> T:
    """
    Returns
    -------
        First element.
    """
    return x[0]


def snd(x: tuple[T, ...]) -> T:
    """
    Returns
    -------
        Second element.
    """
    return x[1]


def in_sorted(a, x):
    i = bisect_left(a, x)
    return i < len(a) and a[i] == x


def enhance_hermiticity(m: np.ndarray) -> tuple[np.ndarray, float]:
    herm_m = (m + hc(m)) / 2
    delta = np.max(np.abs(m - herm_m))
    return herm_m, delta


def _flatten(seq: Iterable[Iterable[T]]) -> list[T]:
    return [x for subseq in seq for x in subseq]


S = Union[T, Iterable['S']]
def flatten(seq: Iterable[Iterable[S]], depth: int=1) -> list[S]:
    result = seq
    for _ in range(depth):
        result = _flatten(result)
    return result


def limited_print(*args):
    mess = ''
    for arg in args:
        mess += str(arg)
        if len(mess) > LINE_WIDTH:
            mess = mess[:LINE_WIDTH - 3] + '...'
    print(mess)


def is_perfect_square(n: int) -> bool:
    return n == int(np.sqrt(n))**2
