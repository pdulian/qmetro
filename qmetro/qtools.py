from __future__ import annotations
from typing import Callable, Any
import inspect

from itertools import product
from math import sqrt

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
import numpy as np
from scipy.linalg import expm

from .consts import PAULI_X, PAULI_Z, PAULIS, QubitStates as QS2
from .utils import matrix_exp_derivative




def ket_bra(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.kron(x.reshape(len(x), 1), y.conjugate())


def cmarkov_krauses(krauses: list[np.ndarray], transition_mat: np.ndarray
    ) -> list[np.ndarray]:
    """
    Computes Kraus operators of channels with classical Markov
    correlations.

    Parameters
    ----------
    krauses : list[np.ndarray]
        Kraus operators or their derivatives.
    transition_mat : np.ndarray
        Matrix `M` of conditional probabilities. The coefficient `M[i, j]`
        is equal to P(j|i) - the conditional probability of ith Kraus
        operator given the occurence of jth operator in the previous step.

    Returns
    -------
    list[np.ndarray]
        List of new operators with correlations.
    """
    raise NotImplementedError
    # This works only for unitary krauses
    transition_mat = np.array(transition_mat)
    d_env= len(krauses)
    id_env = np.identity(d_env)
    corr_ks = []
    for i in range(d_env):
        ei = id_env[i]
        for j in range(d_env):
            ej = id_env[j]
            corr = np.sqrt(transition_mat[i, j]) * ket_bra(ej, ei)
            corr_k = np.kron(corr, krauses[i])
            corr_ks.append(corr_k)

    return corr_ks


def choi_from_krauses(krauses: list[np.ndarray]) -> np.ndarray:
    """
    Get Choi matrix of a channel defined using its Kraus operators.

    For channel Phi: H_in -> H_out returned matrix acts on tensor product
    H_out (x) H_in.

    Parameters
    ----------
    krauses : list[np.ndarray]
        List of Kraus operators.

    Returns
    -------
    choi: np.ndarray
        Choi matrix.
    """
    flat_ks = [K.flatten() for K in krauses]
    C = sum(ket_bra(fk, fk) for fk in flat_ks)
    return C


def dchoi_from_krauses(krauses: list[np.ndarray],
    dkrauses: list[np.ndarray]) -> np.ndarray:
    """
    Get derivative of a Choi matrix of a channel defined using its Kraus
    operators and their derivatives.

    For channel Phi: H_in -> H_out returnde matrix acts on tensor product
    H_out (x) H_in.

    Parameters
    ----------
    krauses : list[np.ndarray]
        List of Kraus operators.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators.

    Returns
    -------
    dchoi: np.ndarray
        Derivative of a Choi matrix.
    """
    flat_ks = [K.flatten() for K in krauses]
    flat_dks = [dK.flatten() for dK in dkrauses]
    dC = sum(
        ket_bra(fdk, fk) + ket_bra(fk, fdk)
        for fk, fdk in zip(flat_ks, flat_dks)
    )
    return dC


def krauses_from_choi(choi: np.ndarray, dims: tuple[int, int],
    eps: float = 1e-7, return_eigensystem: bool = False
    ) -> list[np.ndarray] | tuple[list[np.ndarray], np.ndarray,
    np.ndarray]:
    """
    Calculates Kraus operators of a channel with a given Choi.
    
    Parameters
    ----------
    choi: np.ndarray
        Choi-Jamiolkowski matrix of a channel : Lin(`dout` x `din`)
    dims: tuple[int, int]
        Tuple [`din`, `dout`],`din`, `dout` are input/output dimensions
        of a channel
    eps: float, optional
        Eigenvectors of choi with eigenvalues smaller than eps are
        ignored.
    return_eigensystem: bool, optional
        If true, returns additionally eigenvalues and eigenvectors.
        
    Returns
    -------
    krauses: list[np.ndarray] | tuple[list[np.ndarray], np.ndarray,
    np.ndarray]
        Kraus operators of a channel (arrays of dimensions `dout` x `din`)
        or if return_eigensystem=True a tuple (Kraus operators,
        eigenvalues, eigenvectors) where i-th eigenvector is
        eigenvectors[:, i].
        
    Notes
    -------
        Normalization convention: Trace of the Choi matrix is equal to the
        dimension of the input space.
    """
    D = len(choi)
    din, dout = dims
    if din * dout != D:
        raise ValueError(
            'Invalid dims.\n'
            'The product of two dimensions should be equal to size of '\
            'choi matrix'
        )

    vals, vecs = np.linalg.eigh(choi)
    vals = np.maximum(vals, 0) # for correcting numerical -epsilon
    if any(vals < -eps):
        raise ValueError('Choi matrix must be positive.')

    krauses=[]
    for i in range(D):
        if vals[i] > eps:
            krauses.append(
                np.reshape(np.sqrt(vals[i]) * vecs[:,i], (dout, din))
            )

    if return_eigensystem:
        return krauses, vals, vecs

    return krauses


def hc(x: np.ndarray | cp.Expression) -> np.ndarray | cp.Expression:
    """
    Hermitian conjugation of a numpy matrix or a cvxpy expression.

    Parameters
    ----------
    x : np.ndarray | cp.Expression
        Matrix.

    Returns
    -------
    np.ndarray | cp.Expression
        `x.conjugate().T`
    """    
    return x.conjugate().T


def dkrauses_from_choi(choi: np.ndarray, dchoi: np.ndarray,
    dims: tuple[int, int], eps: float = 1e-7
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Calculates Kraus operators and derivaties of a channel with a given
    Choi and derivative.
    
    Parameters
    ----------
    choi: np.ndarray
        Choi-Jamiolkowski matrix of a channel : Lin(`dout` x `din`). It
        has to be a positive semi-definite matrix.
    dchoi: np.ndarray
        Derivative of aChoi-Jamiolkowski matrix of a channel.
    dims: tuple[int, int]
        Tuple [`din`, `dout`],`din`, `dout` are input/output dimensions of
        a channel
    eps: float, optional
        Eigenvectors of choi with eigenvalues smaller than eps are
        ignored.
        
    Returns
    -------
    krauses: list[np.ndarray]
        Kraus operators of a channel (arrays of dimensions `dout` x
        `din`).
    dkrauses: list[np.ndarray]
        Derivatives of Kraus operators of a channel (arrays of dimensions
        `dout` x `din`).
    
    Notes
    -------
        Normalization convention: choi trace is dimension of input space
    """
    D = len(choi)
    din, dout = dims
    krauses, vals, vecs = krauses_from_choi(choi, dims, eps, True)

    X = hc(vecs) @ dchoi @ vecs # X matrix, Choi_to_Krauses.pdf
    salpha = np.add.outer(np.sqrt(vals), np.sqrt(vals))

    dkrauses = []
    for i in range(D):
        if vals[i] > eps:
            dK = np.sum(
                [X[j][i] / salpha[i][j] * vecs[:, j] for j in range(D)],
                axis=0
            )
            dkrauses.append(np.reshape(dK, (dout, din)))

    return krauses, dkrauses


def choi_dchoi_from_lindblad(lindblad: Callable[[np.ndarray], np.ndarray],
    dlindblad:Callable[[np.ndarray], np.ndarray], dim: int, t: float,
    **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates choi and its derivative of a channel simulating evolution
    with a given lindbladian for time t.

    This function computes choi its derivative over estimated parameter
    for continuous time evolution with a fixed lindbladian for some given
    time. Computations are done fully algebraically, without numerical
    integration

    Parameters
    ----------
    lindblad: Callable[[np.ndarray], np.ndarray]
        Lindbladian: function, that for input density matrix returns its 
        derivative over time; lindblad(rho) = drho/dt. It is assumed that
        Lindbladian is constant over time. For example
        dlindblad(rho, omega) =  0.5j * omega* (rho@sigma_z - sigma_z@rho)
        for rotation around z with angular velocity omega, sigma_z is a 
        Z Pauli matrix.
    dlindblad: Callable[[np.ndarray], np.ndarray]
        Derivative of Lindbladian over estimated parameter. More precisely,
        dlindblad(rho) returns double derivative d^2 rho / (dt * dx), x
        is an estimated parameter. For example 
        dlindblad(rho) =  0.5j * (rho@sigma_z - sigma_z@rho) when parameter
        is angular velocity around z.
    dim: int
        Dimension of Hilbert space on which Lindbladian acts
    t: float
        Evolution time
    **kwargs: Any
        Additional keyword arguments passed to lindblad and dlindblad

    Returns
    -------
    choi: 
        Choi matrix
    dchoi:
        Derivative of Choi matrix over some parameter

    """
    #processing kwargs: accepted by lindblad or dlindblad?
    sig_lindblad = inspect.signature(lindblad)
    sig_dlindblad = inspect.signature(dlindblad)

    lindblad_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in sig_lindblad.parameters
    }
    dlindblad_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in sig_dlindblad.parameters
    }

    #creating basis matrices #E_ij (element ij = 1, other 0)
    basis_matrices = [np.reshape(v, (dim, dim)) for v in np.eye(dim**2)]

    L_columns = [
        lindblad(b, **lindblad_kwargs).flatten() for b in basis_matrices
    ]
    L_mat =  np.array(L_columns).T

    Lt_exp = expm(t * L_mat)
    choi_elements = [
        np.kron((Lt_exp @ b.flatten()).reshape((dim, dim)), b)
        for b in basis_matrices
    ]
    choi = np.sum(choi_elements, axis = 0)

    dL_columns = [
        dlindblad(b, **dlindblad_kwargs).flatten() for b in basis_matrices
    ]
    dL_mat =  np.array(dL_columns).T

    dLt_exp = matrix_exp_derivative(t*L_mat, t*dL_mat)
    dchoi_elements = [
        np.kron((dLt_exp @ b.flatten()).reshape((dim, dim)), b)
        for b in basis_matrices
    ]
    dchoi = np.sum(dchoi_elements, axis = 0)
    
    return choi, dchoi


def depolarization_krauses(p: float | None = None,
    noise_first: bool = True, eta: float | None = None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Computes Kraus operators and their derivatives for a qubit channel
    where the signal is rotating Bloch sphere around the z-axis and noise
    shrinks the whole sphere towards the point (0, 0, 0), that is
    the maximally entangled state. More precisely, it is a channel:

        rho -> p * rho + (1-p)/2 * Id,

    which Kraus operators are:

        sqrt[(3p+2)/5] * Id, sqrt[(1-p)/5] * sigma_i for i=1, 2, 3,

    multiplied from left or right by the signal:
     
        U(phi) = exp(-1j/2 * phi * sigma_z), 
        
    where phi = 0 is a measured parameter defining the derivative.

    Parameters
    ----------
    p : float | None, optional
        Probability that the input state will remain unchanged.
    noise_first : bool, optional
        Whether noise is before signal, by default True.
    eta : float | None, optional
        Alternative method of determining the noise strength that when
        provided is used instead of p (either p or eta argument has to be
        provided). In this parametrisation eta is the scale by which Bloch
        sphere gets shrunken. The Kruas operators of the noise become:

            sqrt(1+3eta)/2 * Id, sqrt(1-eta)/2 * sigma_i for i=1, 2, 3.

        The relation between eta and p is given by eta = (4p+1)/5.

    Returns
    -------
    krauses : list[np.ndarray]
        List of Kraus operators.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators.

    Raises
    ------
    ValueError
        When poth p and eta are provided.
    """
    if p is None and eta is None:
        raise ValueError(
            'Either p or eta must be provided.'
        )
    if p is None:
        p = (5*eta - 1) / 4
    else:
        eta = (4*p + 1) / 5

    krauses = [sqrt(1 + 3 * p) / 2 * np.identity(2)]
    krauses += [sqrt(1 - p) / 2 * sigma for sigma in PAULIS]
    U = -1j/2 * PAULI_Z
    if noise_first:
        dkrauses = [U @ K for K in krauses]
    else:
        dkrauses = [K @ U for K in krauses]

    return krauses, dkrauses


def par_dephasing_krauses(p: float | None = None, noise_first: bool = True,
    eps: float | None = None, rot_like: bool = False
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Computes Kraus operators and their derivatives for a qubit channel
    where the signal is rotating Bloch sphere around the z-axis and noise
    shrinks the xy-plane preserving the z-axis. More precisely, it is
    a channel with Kraus operators:

        sqrt(p) * Id, sqrt(1-p) * sigma_z,

    multiplied from left or right by signal:
     
        U(phi) = exp(-1j/2 * phi * sigma_z), 
        
    where phi = 0 is a measured parameter defining the derivative.

    Parameters
    ----------
    p : float | None, optional
        Probability that the input state will remain unchanged.
    noise_first : bool, optional
        Whether noise is before signal, by default True.
    eps : float | None, optional
        Alternative way of determining the noise strength:

            p = cos(eps/2)**2,
         
        that when provided is used instead of p (p argument is ignored).
    rot_like : bool, optional
        If True then Kraus operators of noise are:

            exp(-1j/2 * eps * sigma_z) / sqrt(2),
            exp(+1j/2 * eps * sigma_z) / sqrt(2),

        where p = cos(eps/2)**2.

    Returns
    -------
    krauses : list[np.ndarray]
        List of Kraus operators.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators.
    
    Raises
    ------
    ValueError
        When both p and eps are provided.
    """
    if p is None and eps is None:
        raise ValueError(
            'Either p or eps must be provided.'
        )
    if p is None:
        p = np.cos(eps / 2)**2
    else:
        eps = 2 * np.arccos(np.sqrt(p))

    if rot_like:
        krauses = [expm(-1j/2 * eps * PAULI_Z), expm(1j/2 * eps * PAULI_Z)]
        krauses = [k / np.sqrt(2) for k in krauses]
    else:
        krauses = [sqrt(p) * np.identity(2), sqrt(1 - p) * PAULI_Z]

    U = -1j/2 * PAULI_Z
    if noise_first:
        dkrauses = [U @ K for K in krauses]
    else:
        dkrauses = [K @ U for K in krauses]
    
    return krauses, dkrauses


def per_dephasing_krauses(p: float, noise_first: bool = True) -> tuple[
    list[np.ndarray], list[np.ndarray]]:
    """
    Computes Kraus operators and their derivatives for a qubit channel
    where the signal is rotating Bloch sphere around the z-axis and noise
    shrinks the yz-plane preserving the x-axis. More precisely, it is
    a channel with Kraus operators:

        sqrt(p) * Id, sqrt(1-p) * sigma_x,

    multiplied from left or right by signal:
     
        U(phi) = exp(-1j/2 * phi * sigma_z), 
        
    where phi = 0 is a measured parameter defining the derivative.

    Parameters
    ----------
    p : float
        Probability that the input state will remain unchanged.
    noise_first : bool, optional
        Whether noise is before signal, by default True.

    Returns
    -------
    krauses : list[np.ndarray]
        List of Kraus operators.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators.
    """
    krauses = [np.sqrt(p) * np.identity(2), np.sqrt(1-p) * PAULI_X]
    U = -1j/2 * PAULI_Z
    if noise_first:
        dkrauses = [U @ K for K in krauses]
    else:
        dkrauses = [K @ U for K in krauses]

    return krauses, dkrauses


def par_amp_damping_krauses(p: float, noise_first: bool = True) -> tuple[
    list[np.ndarray], list[np.ndarray]]:
    """
    Computes Kraus operators and their derivatives for a qubit channel
    where the signal is rotating Bloch sphere around the z-axis and noise
    shrinks the whole sphere towards the point (0, 0, 1) that is | 0 >
    state. More precisely, it is a channel with Kraus operators:

        | 0 >< 0 | + sqrt(p)| 1 >< 1 |, sqrt(1-p)| 0 >< 1|,

    multiplied from left or right by the signal:
     
        U(phi) = exp(-1j/2 * phi * sigma_z), 
        
    where phi = 0 is a measured parameter defining the derivative.

    Parameters
    ----------
    p : float
        Noise parametrization. For p = 1 there is no noise for p = 0
        the noise is maximal.
    noise_first : bool, optional
        Whether noise is before signal, by default True.

    Returns
    -------
    krauses : list[np.ndarray]
        List of Kraus operators.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators.
    """
    krauses = [
        ket_bra(QS2.ZERO, QS2.ZERO)
        + np.sqrt(p) * ket_bra(QS2.ONE, QS2.ONE),
        np.sqrt(1-p) * ket_bra(QS2.ZERO, QS2.ONE)
    ]
    U = -1j/2 * PAULI_Z
    if noise_first:
        dkrauses = [U @ K for K in krauses]
    else:
        dkrauses = [K @ U for K in krauses]
    return krauses, dkrauses


def per_amp_damping_krauses(p: float, noise_first: bool = True) -> tuple[
    list[np.ndarray], list[np.ndarray]]:
    """
    Computes Kraus operators and their derivatives for a qubit channel
    where the signal is rotating Bloch sphere around the z-axis and noise
    shrinks the whole sphere towards the point (1, 0, 0) that is | + >
    state. More precisely, it is a channel with Kraus operators:

        | + >< + | + sqrt(p)| - >< - |, sqrt(1-p)| + >< -|,

    multiplied from left or right by the signal:
     
        U(phi) = exp(-1j/2 * phi * sigma_z), 
        
    where phi = 0 is a measured parameter defining the derivative.

    Parameters
    ----------
    p : float
        Noise parametrization. For p = 1 there is no noise for p = 0
        the noise is maximal.
    noise_first : bool, optional
        Whether noise is before signal, by default True.

    Returns
    -------
    krauses : list[np.ndarray]
        List of Kraus operators.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators.
    """
    krauses = [
        ket_bra(QS2.MINUS, QS2.MINUS)
        + np.sqrt(p) * ket_bra(QS2.PLUS, QS2.PLUS),
        np.sqrt(1-p) * ket_bra(QS2.MINUS, QS2.PLUS)
    ]
    U = -1j/2 * PAULI_Z
    if noise_first:
        dkrauses = [U @ K for K in krauses]
    else:
        dkrauses = [K @ U for K in krauses]
    return krauses, dkrauses


def parallel_krauses(krauses: list[np.ndarray], dkrauses: list[np.ndarray],
    n: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns Kraus operators and their derivatives for n parallel channels.

    Parameters
    ----------
    krauses : list[np.ndarray]
        Kraus operators of single channel.
    dkrauses : list[np.ndarray]
        Derivatives of Kraus operators of single channel.
    n : int
        Number of parallel channels.

    Returns
    -------
    krauses : list[np.ndarray]
        Kraus operators of n channels.
    dkrauses : list[np.ndarray]
        Derivatives of Kraus operators of n channels.        
    """
    rank = len(krauses)
    par_krauses = []
    par_dkrauses = []
    for indices in product(range(rank), repeat=n):
        new_kraus = 1 + 0j
        for i in indices:
            new_kraus = np.kron(new_kraus, krauses[i])
        par_krauses.append(new_kraus)

        new_dkraus = 0
        for i in range(n): # chain rule
            tmp = 1 + 0j
            for j, k in enumerate(indices):
                if j == i:
                    tmp = np.kron(tmp, dkrauses[k])
                else:
                    tmp = np.kron(tmp, krauses[k])
            new_dkraus += tmp
        par_dkrauses.append(new_dkraus)

    return par_krauses, par_dkrauses


def swap_operator(dims: tuple[int, ...], i1: int, i2: int) -> np.ndarray:
    """
    Construct a swap operator matrix for a tensor product space,
    swapping two specified subsystems.

    This function returns a matrix that, when applied to a vector or
    matrix in the tensor product space `H_1 x H_2 x ... x H_N`, swaps
    the specified subsystems with indices `i1` and `i2`.
    The resulting space ordering will be:
    
        - Original structure: `H_1 x ... x H_i1 x ... x H_i2 x ... H_N`
        - New structure: `H_1 x ... x H_i2 x ... x H_i1 x ... H_N` 

    If `v` is a vector in the original structure, `swap_operator @ v`
    gives the vector in the swapped structure.
    If `M` is a matrix in the original structure, `swap_operator @ M @
    swap_operator.T` gives the matrix in the swapped structure.

    Parameters
    ----------
    dims : tuple of int
        Dimensions of each subsystem `H_1`, ..., `H_N` in the tensor
        product space.
    i1 : int
        Index of the first subsystem to swap.
    i2 : int
        Index of the second subsystem to swap.

    Returns
    -------
    np.ndarray
        The swap operator matrix that swaps subsystems `i1` and `i2` in
        the tensor product space.

    Notes
    -----
    The swap operator constructs the necessary index mappings by
    decomposing the tensor product space indices and swapping the basis
    states of the specified subsystems.

    Examples
    --------
    >>> dims = (2, 2, 2)
    >>> i1, i2 = 0, 1
    >>> swap_operator(dims, i1, i2)
    array([...])  # Swap operator matrix for a 3-qubit system
    """
    # Number of subsystems
    n = len(dims)
    dims = np.array(dims)

    # Create swapped dimensions array
    swapped_dims = np.copy(dims)
    swapped_dims[i1], swapped_dims[i2] = dims[i2], dims[i1]

    # Calculate total dimension and dimension products
    total_dim = np.prod(dims)
    dim_products = np.array(
        [np.prod(dims[i : n+1]) for i in range(1, n + 1)]
    )
    swapped_dim_products = np.array(
        [np.prod(swapped_dims[i : n+1]) for i in range(1, n + 1)]
    )

    # Initialize the swap matrix
    swap_matrix = np.zeros((total_dim, total_dim))

    # Populate the swap matrix
    for j in range(total_dim):
        j_index = j
        T = []
        for k in range(n):
            index_k = j_index // swapped_dim_products[k]
            T.append(index_k)
            j_index -= index_k * swapped_dim_products[k]

        # Swap the specified subsystem indices
        T[i1], T[i2] = T[i2], T[i1]

        i_index = np.sum(np.array(T) * dim_products)
        swap_matrix[j, i_index] = 1

    return swap_matrix


def comb_variables(dims: tuple[int, ...], hermitian: bool = True,
    trace_constraint: None | float = 1
    ) -> tuple[
        list[cp.Variable], list[Constraint], cp.Variable | float
    ]:
    """
    Construct a sequence of CVXPY variables that represent quantum combs
    satisfying causality constraints.

    Each element in the returned `combs` list represents a quantum comb
    matrix acting on progressively larger subsystems of a composite
    Hilbert space, where subsystems alternate as input (odd-indexed,
    starting from  1) and output (even-indexed) spaces.

    Parameters
    ----------
    dims : tuple of int
        Dimensions of each Hilbert space `H_1, H_2, ..., H_2N` in
        the composite system. The number of spaces (length of `dims`) must
        be even, with N pairs of input-output spaces.
    hermitian : bool, optional
        If True, each matrix in `combs` is constrained to be Hermitian, by
        default True.
    trace_constraint : float or None, optional
        Specifies the trace constraint on the first comb operator.
        If `None`, a trace variable is used instead of a fixed trace, by
        default 1.

    Returns
    -------
    tuple
        A tuple containing:
            - combs : list of cp.Variable
                List of CVXPY variables representing the quantum comb
                operators.
            - constraints : list of cp.Constraint
                List of CVXPY constraints on the comb operators ensuring
                causality.
            - trace_var or trace_constraint : cp.Variable or float
                If `trace_constraint` is None, returns a trace variable
                for the constraint. Otherwise, returns the specified
                `trace_constraint` value.

    Raises
    ------
    ValueError
        If the length of `dims` is not even, as quantum comb spaces must
        have matching input-output pairs.

    Notes
    -----
    - Element `combs[i]` is a Choi-Jamiolkowski operator acting on
        H_2i+2 (x) ... (x)H_2 (x) H_2i+1 (x) ... H_1, so it belongs to
        set Comb[(H_1, H_2), ..., (H_2i+1, H_2i+2)]
    - The full comb is the last element `combs[N-1]` Comb[(H_1, H_2), ...,
        (H_2N-1, H_2N)]
    - If `trace_constraint` is not `None`, it represents the trace of
        the initial operator scaled by input space dimensions. This is not
        the overall comb trace, which includes all input dimensions.
    - The function does not assume positivity for the combs. A positivity
        constraint can be added externally if needed.

    Examples
    --------
    >>> dims = (2, 2, 2, 2)
    >>> combs, constraints, trace_constraint = comb_variables(dims)
    >>> len(combs)
    2
    >>> len(constraints)
    2
    """
    # Check if the number of dimensions is even
    if len(dims) % 2 != 0:
        raise ValueError(
            'Number of comb spaces must be even.\n'\
            'When the first tooth has no input, set 1 as the first'\
            'dimension.'
        )

    # Determine the number of comb teeth (N) and total dimension
    N = len(dims) // 2

    # Construct SDP variables for each comb matrix
    combs = []
    for i in range(N):
        shape = (np.prod(dims[:2 * i + 2]), np.prod(dims[:2 * i + 2]))
        if hermitian:
            comb = cp.Variable(shape, hermitian=hermitian)
        else:
            comb = cp.Variable(shape, complex=True)
        combs.append(comb)

    # Define constraints on the comb matrices
    constraints = []
    if trace_constraint is None:
        trace_var = cp.Variable(complex=not hermitian)
        constraints.append(
            cp.partial_trace(combs[0], (dims[1], dims[0]), axis=0)
            == trace_var * np.eye(dims[0])
        )
    else:
        constraints.append(
            cp.partial_trace(combs[0], (dims[1], dims[0]), axis=0)
            == trace_constraint * np.eye(dims[0])
        )

    # Initialize subsystem dimensions for current comb
    even_dim, odd_dim = dims[1], dims[0]
    for i in range(1, N):
        # Dimensions for constructing swap operator and partial trace
        # constraints
        swap_dims = (dims[2 * i], even_dim, odd_dim)
        even_dim *= dims[2 * i + 1]
        odd_dim *= dims[2 * i]
        partial_trace_dims = (
            dims[2*i + 1], even_dim*odd_dim // dims[2*i + 1]
        )

        # Apply swap operator to make causality constraints correct
        swap_op = swap_operator(swap_dims, 0, 1)
        if dims[2*i + 1] > 1:
            # Apply partial trace when the subsystem dimension is greater
            # than 1
            x = cp.partial_trace(combs[i], partial_trace_dims, axis=0)
            y = (
                swap_op
                @ cp.kron(np.eye(dims[2 * i]), combs[i - 1]) 
                @ swap_op.T
            )
            constraints.append(x == y)
        else:
            #Skip partial trace when the action is trivial (dimension is 1)
            #constraint is not added, variable is replace with expression
            #instead. This is faster  then adding equality constraint
            #sometimes more then 10 times faster!
            #13s vs 170s for N=4 comb QFI
            combs[i] = (
                swap_op
                @ cp.kron(np.eye(dims[2 * i]), combs[i - 1])
                @ swap_op.T
            )

    # Return combs, constraints, and either trace variable or fixed trace
    # constant
    if trace_constraint is None:
        return combs, constraints, trace_var
    return combs, constraints, trace_constraint


def krauses_kron(
    krauses1: list[np.ndarray], dkrauses1: list[np.ndarray],
    krauses2: list[np.ndarray], dkrauses2: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Computes the Kraus representation of the Kronecker product of two
    quantum channels and their derivatives.

    This function creates a representation of the Kronecker product of two
    quantum channels, including the combined Kraus operators and their
    derivatives.

    Parameters
    ----------
    krauses1 : list of np.ndarray
        List of Kraus operators for the first channel.
    dkrauses1 : list of np.ndarray
        List of derivatives of Kraus operators for the first channel.
    krauses2 : list of np.ndarray
        List of Kraus operators for the second channel.
    dkrauses2 : list of np.ndarray
        List of derivatives of Kraus operators for the second channel.

    Returns
    -------
    krauses12 : list of np.ndarray
        The Kraus operators of the Kronecker product channel.
    dkrauses12 : list of np.ndarray
        The derivatives of the Kraus operators for the Kronecker product
        channel.
    """
    # Compute the Kraus operators for the tensor product channel
    krauses12 = [np.kron(K1, K2) for K1 in krauses1 for K2 in krauses2]

    # Number of Kraus operators for each channel
    r1 = len(krauses1)
    r2 = len(krauses2)

    # Compute the derivatives of the Kraus operators using the product
    # rule
    dkrauses12 = []
    for i, j in product(range(r1), range(r2)):
        dK1i_x_K2j = np.kron(dkrauses1[i], krauses2[j])
        K1i_x_dK2j = np.kron(krauses1[i], dkrauses2[j])
        dkrauses12.append(dK1i_x_K2j + K1i_x_dK2j)

    return krauses12, dkrauses12


def krauses_sequential(
    krauses1: list[np.ndarray], dkrauses1: list[np.ndarray],
    krauses2: list[np.ndarray], dkrauses2: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Computes the Kraus representation of the sequential composition of two
    quantum channels and their derivatives.

    This function creates a representation of the sequential composition
    of two quantum channels (krauses and their derivatives of a resulting
    channel). krauses1 go before krauses2

    Parameters
    ----------
    krauses1 : list of np.ndarray
        List of Kraus operators for the first channel.
    dkrauses1 : list of np.ndarray
        List of derivatives of Kraus operators for the first channel.
    krauses2 : list of np.ndarray
        List of Kraus operators for the second channel.
    dkrauses2 : list of np.ndarray
        List of derivatives of Kraus operators for the second channel.

    Returns
    -------
    krauses12 : list of np.ndarray
        The Kraus operators of the Kronecker product channel.
    dkrauses12 : list of np.ndarray
        The derivatives of the Kraus operators for the Kronecker product
        channel.
    """
    # Compute the Kraus operators for the composition channel
    krauses12 = [K1 @ K2 for K1 in krauses1 for K2 in krauses2]

    # Number of Kraus operators for each channel
    r1 = len(krauses1)
    r2 = len(krauses2)

    # Compute the derivatives of the Kraus operators using the product
    # rule
    dkrauses12 = []
    for i, j in product(range(r1), range(r2)):
        dK1i_K2j = dkrauses1[i] @ krauses2[j]
        K1i_dK2j = krauses1[i] @ dkrauses2[j]
        dkrauses12.append(dK1i_K2j + K1i_dK2j)

    return krauses12, dkrauses12


def minimize_alpha(krauses: list[np.ndarray], dkrauses: list[np.ndarray]
    ) -> float:
    """
    Minimize the norm of alpha over all Kraus representations for a given
    channel.

    Given a list of Kraus operators and their derivatives, this function
    calculates the minimum norm of alpha over all possible Kraus
    representations for the input channel.

    Parameters
    ----------
    krauses : list[np.ndarray]
        List of Kraus operators, each represented as a 2D NumPy array.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators, each represented as a 2D
        NumPy array.

    Returns
    -------
    float
        The minimum value of norm of alpha over all Kraus representations.

    Bibliography
    --------
    .. [1] Demkowicz-Dobrzański, R., Kołodyński, J. & Guţă, M.
       The elusive Heisenberg limit in quantum-enhanced metrology.
       Nat Commun 3, 1063 (2012). https://doi.org/10.1038/ncomms2067
    """
    # Number of Kraus operators
    num_kraus = len(krauses)

    # Dimensions of the input and output spaces
    dout = krauses[0].shape[0]  # Output dimension
    din = krauses[0].shape[1]   # Input dimension

    # Concatenate Kraus operators and their derivatives along the first
    # axis
    K = np.concatenate(krauses)
    dK = np.concatenate(dkrauses)

    # Define the hermitian matrix h and scalar t for minimization
    h = cp.Variable((num_kraus, num_kraus), hermitian=True)
    t = cp.Variable()

    # Construct the block matrix A for the semidefinite constraint
    A00 = t * np.eye(din)  # Top-left block
    A11 = np.eye(dout * num_kraus)  # Bottom-right block
    A10 = dK - 1j * cp.kron(h, np.eye(dout)) @ K  # Top-right block
    A01 = hc(A10)  # Bottom-left block
    A = cp.bmat([
        [A00, A01],
        [A10, A11]
    ])

    # Define constraints and objective for the optimization problem
    constraints = [A >> 0]  # Positive semidefinite constraint
    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem and return the minimum value of alpha
    return problem.solve()


def get_sld(rho: np.ndarray, drho: np.ndarray, return_qfi: bool = False
    ) -> np.ndarray | tuple[float, np.ndarray]:
    """
    Computes symmetric logarithmic derivative (SLD) of a parametrized
    state.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix.
    drho : np.ndarray
        Derivative of density matrix.
    return_qfi : bool, optional
        Whether to return also a quantum Fisher information (QFI) of
        the state, by default False.

    Returns
    -------
    np.ndarray | tuple[float, np.ndarray]
        SLD matrix or a pair (SLD, QFI) if return_qfi is True.
    """
    d = len(rho)
    L = cp.Variable((d, d), hermitian=True)
    L2 = cp.Variable((d, d), hermitian=True)
    A = cp.bmat([[L2, L], [L, np.eye(d)]])
    constraint = [A >> 0,] # L2 >> L^2
    obj = cp.Maximize(cp.real(cp.trace(2 * drho @ L - rho @ L2)))
    prob = cp.Problem(obj, constraint)
    qfi = prob.solve()

    if return_qfi:
        return qfi, L.value
    return L.value


def povm_from_sld(sld: np.ndarray) -> list[np.ndarray]:
    """
    Calculates optimal measurement POVM from SLD matrix.

    Parameters
    ----------
    sld: np.ndarray
        Symmetric logarithmic derivative (SLD) matrix

    Returns
    -------
    list[np.ndarray]
        List of optimal measurement operators: projections on eigenvectors
        of SLD matrix.
    """
    _, SLD_vecs = np.linalg.eigh(sld)
    return [ket_bra(vec, vec) for vec in SLD_vecs.T]
