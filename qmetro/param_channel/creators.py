from typing import Any

import numpy as np

from ..qtools import (
    depolarization_krauses, par_dephasing_krauses, per_dephasing_krauses,
    par_amp_damping_krauses, per_amp_damping_krauses,
    ket_bra
)

from . import ParamChannel




def par_dephasing(p: float, noise_first: bool = True,
    eps: float | None = None, rot_like: bool = False,
    c: float | None = None, **kwargs: Any) -> ParamChannel:
    """
    Returnes paramtrised channel of qubit channel where the signal is
    rotating Bloch sphere around the z-axis and noise shrinks the xy-plane
    preserving the z-axis. More precisely, it is a channel with Kraus
    operators:

        sqrt(p) * Id, sqrt(1-p) * sigma_z,

    multiplied from left or right by signal:

        U(phi) = exp(-1j/2 * phi * sigma_z),

    where phi = 0 is a measured parameter defining the derivative.

    Parameters
    ----------
    p : float
        Probability that the input state will remain unchanged.
    noise_first : bool, optional
        Whether noise is before signal, by default True.
    eps : float | None, optional
        Alternative way of determining the noise strength:

            p = cos(eps/2)**2,

        that when provided is used instead of p (p argument is ignored).
    rot_like : bool, optional
        If True then Kraus operators of noise are:

            U+ = exp(-1j/2 * eps * sigma_z) / sqrt(2),
            U- = exp(+1j/2 * eps * sigma_z) / sqrt(2),

        where p = cos(eps/2)**2, by default False.
    c : float | None, optional
        Correlation parameter from the interval [-1, 1]. When set it will
        put rot_like=True and create a channel with environment space such
        that U+ and U- will be correlated. They are fully correlated for
        c=1, no correlated for c=0 and anticorrelated for c=-1. If None
        creates ParamChannel without enviornment space. By default None.
    **kwargs : dict, optional
        Arguments that will be passed to ParamChannel constructor.

    Returns
    -------
    channel : ParamChannel
        Parametrised channel.
    """
    rot_like = rot_like or c is not None

    if eps is None:
        krauses, dkrauses = par_dephasing_krauses(
            p, noise_first, rot_like=rot_like
        )
    else:
        krauses, dkrauses = par_dephasing_krauses(
            noise_first=noise_first, eps=eps, rot_like=rot_like
        )

    if c is None:
        return ParamChannel(krauses=krauses, dkrauses=dkrauses, **kwargs)

    trans = np.array([
        [(1+c)/2, (1-c)/2],
        [(1-c)/2, (1+c)/2]
    ])
    d_env= len(krauses)
    id_env = np.identity(d_env)
    corr_ks = []
    corr_dks = []
    for i in range(d_env):
        ei = id_env[i]
        for j in range(d_env):
            ej = id_env[j]
            corr = np.sqrt(trans[i, j]) * ket_bra(ej, ei)

            corr_k = np.kron(corr, krauses[i]) * np.sqrt(2)
            corr_ks.append(corr_k)
            
            corr_dk = np.kron(corr, dkrauses[i]) * np.sqrt(2)
            corr_dks.append(corr_dk)
    
    return ParamChannel(
        krauses=corr_ks, dkrauses=corr_dks, env_dim=2, **kwargs
    )


def per_dephasing(p: float, noise_first: bool = True, **kwargs
    ) -> ParamChannel:
    """
    Returnes paramtrised channel of qubit channel where the signal is
    rotating Bloch sphere around the z-axis and noise shrinks the yz-plane
    preserving the x-axis. More precisely, it is a channel with Kraus
    operators:

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
    **kwargs : dict, optional
        Arguments that will be passed to ParamChannel constructor.

    Returns
    -------
    channel : ParamChannel
        Parametrised channel.
    """
    krauses, dkrauses = per_dephasing_krauses(p, noise_first)
    return ParamChannel(krauses=krauses, dkrauses=dkrauses, **kwargs)


def per_amp_damping(p: float, noise_first: bool = True, **kwargs: Any
    ) -> ParamChannel:
    """
    Returnes paramtrised channel of qubit channel where the signal is
    rotating Bloch sphere around the z-axis and noise shrinks the whole
    sphere towards the point (1, 0, 0) that is | + > state. More
    precisely, it is a channel with Kraus operators:

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
    **kwargs : dict, optional
        Arguments that will be passed to ParamChannel constructor.

    Returns
    -------
    channel : ParamChannel
        Parametrised channel.
    """
    krauses, dkrauses = per_amp_damping_krauses(p, noise_first)
    return ParamChannel(krauses=krauses, dkrauses=dkrauses, **kwargs)


def par_amp_damping(p: float, noise_first: bool = True, **kwargs
    ) -> ParamChannel:
    """
    Returnes paramtrised channel of qubit channel where the signal is
    rotating Bloch sphere around the z-axis and noise shrinks the whole
    sphere towards the point (0, 0, 1) that is | 0 > state. More
    precisely, it is a channel with Kraus operators:

        | 0 >< 0 | + sqrt(p)| 1 >< 1 |, sqrt(1-p)| 0 >< 1|,

    multiplied from left or right by the signal:

        U(phi) = exp(-1j/2 * phi * sigma_z),

    where phi = 0 is a measured parameter defining the derivative.

    Parameters
    ----------
    p : float
        Noise strength. For p = 1 there is no noise for p = 0 the noise is
        maximal.
    noise_first : bool, optional
        Whether noise is before signal, by default True.
    **kwargs : dict, optional
        Arguments that will be passed to ParamChannel constructor.

    Returns
    -------
    channel : ParamChannel
        Parametrised channel.
    """
    krauses, dkrauses = par_amp_damping_krauses(p, noise_first)
    return ParamChannel(krauses=krauses, dkrauses=dkrauses, **kwargs)


def depolarization(p: float, noise_first: bool = True,
    eta: float | None = None, **kwargs: Any) -> ParamChannel:
    """
    Returnes paramtrised channel of qubit channel where signal is
    rotating Bloch sphere around the z-axis and noise shrinks the whole
     sphere towards the point (0, 0, 0), that is the maximally entangled
    state. More precisely, it is a channel:

        rho -> p * rho + (1-p)/2 * Id,

    which Kraus operators are:

        sqrt[(3p+2)/5] * Id, sqrt[(1-p)/5] * sigma_i for i=1, 2, 3,

    multiplied from left or right by the signal:

        U(phi) = exp(-1j/2 * phi * sigma_z),

    where phi = 0 is a measured parameter defining the derivative.

    Parameters
    ----------
    p : float
        Probability that the input state will remain unchanged.
    noise_first : bool, optional
        Whether noise is before signal, by default True.
    eta : float | None, optional
        Alternative method of determining the noise strength that when
        provided is used instead of p (p argument is ignored). In this
        parametrisation eta is the scale by which Bloch sphere gets
        shrunken. The Kruas operators of the noise become:

            sqrt(1+3eta)/2 * Id, sqrt(1-eta)/2 * sigma_i for i=1, 2, 3.

        The relation between eta and p is given by eta = (4p+1)/5.
    **kwargs : dict, optional
        Arguments that will be passed to ParamChannel constructor.


    Returns
    -------
    channel : ParamChannel
        Parametrised channel.
    """
    if eta is None:
        krauses, dkrauses = depolarization_krauses(p, noise_first)
    else:
        krauses, dkrauses = depolarization_krauses(
            noise_first=noise_first, eta=eta
        )
    return ParamChannel(krauses=krauses, dkrauses=dkrauses, **kwargs)


def corr_dephasing(p: float, c: float, angle: float = 0,
    noise_first: bool = True, c_in: float | None = None) -> tuple[
    list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Computes:
    
    - Kraus operators,
    - derivative of Kraus operators,
    - Choi matrix,
    - derivative of Choi matrix
    
    of two qubit channel representing rotation of Bloch sphere around
    z-axis accompanied by dephasing noise. The 1st qubit on which
    the channel acts is the register, which allows to model binary
    correlations between subsequent dephasing angles. 

    Parameters
    ----------
    p : float
        Probability that the input state will remain unchanged.
        No dephasing for p=1, maximal dephasing for p=0.5.
    c : float
        Correlation parameter. When first qubits of subsequent channels
        (environments) are connected, then dephasing angles are fully 
        correlated for c=1, no correlated for c=0 and anticorrelated for
        c=-1.
    angle : float, optional
        Angle between signal and dephasing axis, by default 0.
    noise_first : bool, optional
        Whether noise is before signal, by default True.
    c_in : float | None, optional
        Correlation of noise map acting on input environment.
        If None, then c_in = sqrt(c), then noise is equally distributed
        between input and output environments.

    Returns
    -------
    krauses : list[np.ndarray]
        List of Kraus operators.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators.
    choi: np.ndarray
        Choi matrix.
    dchoi: np.ndarray
        Derivative of Choi matrix.
    
    """
    # TODO
    raise NotImplementedError(
        'This function is not implemented yet.'
    )
