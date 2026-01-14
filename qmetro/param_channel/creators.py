from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from ..qtools import ket_bra

from ..qtools import (
    depolarization_krauses, par_dephasing_krauses, per_dephasing_krauses,
    par_amp_damping_krauses, per_amp_damping_krauses
)

from . import ParamChannel




def par_dephasing(p: float, noise_first: bool = True,
    eps: float | None = None, rot_like: bool = False,
    c: float | None = None, **kwargs: Any) -> ParamChannel:
    """
    Returnes paramtrised channel of a qubit channel
    where:
    - the signal is rotating Bloch sphere around the z-axis,
    - noise shrinks the xy-plane preserving the z-axis.
    
    See more details in :ref:`the documentation <par-deph>`.

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

            K+ = exp(-1j/2 * eps * sigma_z) / sqrt(2),
            K- = exp(+1j/2 * eps * sigma_z) / sqrt(2),

        where p = cos(eps/2)**2, by default False.
    c : float | None, optional
        Correlation parameter from the interval [-1, 1]. When set it will
        put rot_like=True and create a channel with environment space such
        that K+ and K- will be correlated (see also :func:`cmarkov_channel`
        ). They are fully correlated for c=1, no correlated for c=0 and
        anticorrelated for c=-1. If None creates ParamChannel without
        environment space. By default None.
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

    Kp, Km = krauses
    dKp, dKm = dkrauses
    
    sqrt2 = np.sqrt(2)
    rot_p = ParamChannel(krauses=[sqrt2*Kp], dkrauses=[sqrt2*dKp])
    rot_m = ParamChannel(krauses=[sqrt2*Km], dkrauses=[sqrt2*dKm])

    return cmarkov_channel([rot_p, rot_m], trans)


def per_dephasing(p: float, noise_first: bool = True, **kwargs
    ) -> ParamChannel:
    """
    Returnes paramtrised channel of a qubit channel
    where:
    - the signal is rotating Bloch sphere around the z-axis,
    - noise shrinks the yz-plane preserving the x-axis.

    See more details in :ref:`the documentation <per-deph>`.

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
    Returnes paramtrised channel of a qubit channel
    where:
    - the signal is rotating Bloch sphere around the z-axis,
    - noise models decay from +1 to -1 eigenstate of Pauli x-matrix.

    See more details in :ref:`the documentation <per-amp>`.

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
    Returnes parametrised channel of a qubit channel
    where:
    - the signal is rotating Bloch sphere around the z-axis,
    - noise models decay from -1 to +1 eigenstate of Pauli z-matrix.

    See more details in :ref:`the documentation <par-amp>`.

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
    Returnes paramtrised channel of a qubit channel
    where:
    - the signal is rotating Bloch sphere around the z-axis,
    - noise shrinks uniformly the whole Bloch ball.

    See more details in :ref:`the documentation <depolarization>`.

    Parameters
    ----------
    p : float
        Probability that the input state will remain unchanged.
    noise_first : bool, optional
        Whether noise is before signal, by default True.
    eta : float | None, optional
        Alternative method of determining the noise strength that when
        provided is used instead of p (either p or eta argument has to be
        provided). In this parametrisation eta is the factor by which Bloch
        sphere gets shrunken.
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


def cmarkov_channel(channels: list[ParamChannel], trans_mat: np.ndarray
    ) -> ParamChannel:
    """
    Creates a channel encoding classical Markovian correlations between
    a list of channels with transition probabilities given by the
    transition matrix.

    It is a channel G such that `G.link_env(G, ..., G)` is a comb
    representing a Markov chain of channels from the input list.

    Formally, for a list of channels F0, F1, ..., Fn-1 of the form
    L(H) -> L(K) and a transition matrix T, the function constructs
    a channel G: L(H x E) -> L(K x E), where E is a newly created
    environment space of dimension n. When the environment is in
    the i-th state G applies Fi and changes the state of the environment
    to j with probability T[i, j]. In this way, the state of
    the environment encodes which channel from the list will be applied
    next.

    Parameters
    ----------
    channels : list[ParamChannel]
        List of channels F0, ..., Fn-1to be connected in a chain. All must
        have the same input and output space dimensions, be single (not
        comb) and have trivial environments.
    trans_mat : np.ndarray
        Transition matrix defining the transition probabilities, that is
        T[i, j] is the probability of transiting from channel i to channel
        j.
    
    Returns
    -------
    channel : ParamChannel
        Channel G encoding the Markovian correlations.
    """
    n = len(channels)
    d_in = channels[0].input_dim
    d_out = channels[0].output_dim

    for ch in channels:
        if ch.input_dim != d_in or ch.output_dim != d_out:
            raise ValueError(
                'All channels must have the same input and output '
                'dimensions.'
            )
        if not ch.trivial_env:
            raise ValueError(
                'All channels must have trivial environment spaces.'
            )
        if ch.is_comb:
            raise ValueError(
                'All channels must be single channels, not combs.'
            )
    
    # Krauses of channel G.
    As = []
    dAs = []
    id = np.identity(n)
    for i, j in product(range(n), repeat=2):
        prob = trans_mat[i, j]
        if prob == 0:
            continue
        
        ei = id[i]
        ej = id[j]
        
        for k, dk in zip(*channels[i].dkrauses()):
            A = np.sqrt(prob) * np.kron(ket_bra(ej, ei), k)
            dA = np.sqrt(prob) * np.kron(ket_bra(ej, ei), dk)
            As.append(A)
            dAs.append(dA)

    return ParamChannel(krauses=As, dkrauses=dAs, env_dim=n)


def corr_dephasing(p: float, c: float, angle: float = 0,
    noise_first: bool = True, c_in: float | None = None) -> ParamChannel:
    """
    Returns rotation-like parametrised dephasing channel with correlation
    c between subsequent U+ and U- Kraus operators.

    Creates a channel G encoding classical Markov correlation with
    transition matrix:
        
        T = [ (1+c)/2 , (1-c)/2 ]
            [ (1-c)/2 , (1+c)/2 ]
    
    The information about correlations is stored in an environment space
    such that G.link_env(G, ..., G) is a comb representing a Markov
    chain of dephasing channels with correlation c.

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
    channel : ParamChannel
        Correlated dephasing channel.
    
    """
    raise NotImplementedError()
    # TODO: Make it into general dephasing function.
    if angle or c_in is not None:
        raise NotImplementedError(
            'Currently only angle=0 and c_in=None are supported.'
        )

    T = np.array([
        [1+c, 1-c],
        [1-c, 1+c]
    ]) / 2

    deph = par_dephasing(p, rot_like=True, noise_first=noise_first)
    (Up, Um), (dUp, dUm) = deph.dkrauses()

    sqrt2 = np.sqrt(2)
    rot_p = ParamChannel(krauses=[sqrt2*Up], dkrauses=[sqrt2*dUp])
    rot_m = ParamChannel(krauses=[sqrt2*Um], dkrauses=[sqrt2*dUm])

    return cmarkov_channel([rot_p, rot_m], T)
