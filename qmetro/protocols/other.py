from __future__ import annotations

import numpy as np

from ..qtools import get_sld




def multiple_measurements_qfi(single_qfis: list[float],
    ns: list[int] | None = None) -> tuple[list[float], list[list[int]]]:
    """
    For a list of QFIs for measurements with n = 1, 2, ..., max_n probes
    computes optimal QFI for each number of probes ditributed among
    multiple measurments.

    If QFI at some point starts to scale with number of probes sublinearly
    for example:
    
        QFI1 = 1, QFI2 = 2.1, QFI3 = 2.7, ...
    
    then it is advantegous to use probes in multiple measurements. In this
    case:
    
        QFI1_opt = QFI1, QFI2_opt = QFI2, QFI3_opt = 3 * QFI1, ...
    
    In general, it requires solving a special case of 
    `a knapsack problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_ .

    Parameters
    ----------
    single_qfis : list[float]
        List of QFIs obtained in a single measurement such that
        single_qfi[i] is QFI obtained with i + 1 probes.
    ns : list[int] | None, optional
        If provided number of probes for single_qfi[i] is ns[i], by
        default None.

    Returns
    -------
    qfis : list[float]
        List of optimal QFIs such that single_qfi[i] is QFI obtained with
        i probes (note the shift -1 shift from single_qfis ordering).
    strategies : list[list[int]]
        List of optimal strategies. The element strategies[i] is a list
        of single measurements constituting the optimal strategy for i
        probes.
    
    """
    if ns is None:
        max_n = len(single_qfis)
        _ns = range(1, max_n + 1)
    else:
        max_n = max(ns)
        _ns = ns
    sqfis = dict(zip(_ns, single_qfis))

    qfis = []
    strategies = []
    for n in range(0, max_n + 1):
        if n in sqfis:
            qfi = sqfis[n]
            strategy = [n] if n else []
        else:
            qfi = 0.0
            strategy = []

        for i in range(n):
            tmp = qfis[i] + qfis[n - i]
            if tmp > qfi:
                qfi = tmp
                strategy = strategies[i] + strategies[n - i]

        qfis.append(qfi)
        strategies.append(sorted(strategy))

    return qfis, strategies


def state_qfi(state: tuple[np.ndarray, np.ndarray], **kwargs
    ) -> tuple[float, np.ndarray]:
    """
    Calculates quantum Fisher information of a quantum state represented
    by density matrix and its derivative over estimated parameter.

    Parameters
    ----------
    state: tuple[np.ndarray, np.ndarray]
        A tuple containing:
            1) rho: np.ndarray
                Density matrix
            2) drho: np.ndarray
                Derivative of density matrix over estimated parameter
    **kwargs
        Additional keyword arguments passed to the CVXPY ``solve`` method
        (see `docs <https://www.cvxpy.org/tutorial/solvers/index.html>`_).

    Returns
    -------
    float
        State qfi
    np.ndarray
        Symmetric logarithmic derivative (SLD) matrix
    """
    rho, drho = state
    return get_sld(rho, drho, return_qfi=True, **kwargs)


def state_cfi(state: tuple[np.ndarray, np.ndarray], povm: list[np.ndarray],
    eps: float = 1e-7) -> float:
    """
    Calculates classical Fisher information of a quantum state represented
    by density matrix and its derivative over estimated parameter for
    a given measurement represented by POVM.

    Parameters
    ----------
    state: tuple[np.ndarray, np.ndarray]
        A tuple containing:
            1) rho: np.ndarray
                Density matrix
            2) drho: np.ndarray
                Derivative of density matrix over estimated parameter
    povm: list[np.ndarray]
        List of operators defining a generalized measurement (POVM)
    eps: float, optional
        probabilities smaller then eps will not contribute to cfi

    Returns
    -------
    float
        Classical Fisher information for input state and measurement
    """
    rho, drho = state

    p_list = [np.real(np.trace(rho@M)) for M in povm]
    dp_list = [np.real(np.trace(drho@M)) for M in povm]

    cfi = 0

    for p, dp in zip(p_list, dp_list):
        if p > eps:
            cfi += dp**2 / p

    return cfi
