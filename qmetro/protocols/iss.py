from __future__ import annotations

from collections.abc import Hashable
from itertools import cycle
from warnings import warn, filterwarnings, catch_warnings

import numpy as np

from ..iss_opt import iss_opt
from ..qmtensor import (
    SpaceDict, TensorNetwork, VarTensor, ConstTensor,
    mps_var_tnet, choi_identity, mpo_measure_var_tnet, measure_var,
    input_state_var, comb_var
)
from ..param_channel import ParamChannel

from .errors import EnvDimsError, UnitalDimsError
from .warnings import ENV_FOR_SINGLE, COMB_FOR_SINGLE




def iss_channel_qfi(channel: ParamChannel, ancilla_dim: int = 1,
    env_inp_state: np.ndarray | None = None,
    artificial_noise_after: bool | None = True, **kwargs
    ) -> tuple[float, list[float], np.ndarray, np.ndarray, bool]:
    """
    Computes quantum Fisher information for a single parametrized channel
    using iterative see-saw (ISS) method [1]_.

    Parameters
    ----------
    channel : ParamChannel
        Channel to compute quantum Fisher information for. In case this
        argument is a comb created from single channels the whole comb
        will be treated as a single channel.
    ancilla_dim : int, optional
        Dimension of the ancilla, by default 1.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state.
    artificial_noise_after : bool | None, optional
        Whether auxiliary noise is added after the channel:
        - True : auxiliary noise is added after the channel,
        - False : auxiliary noise is added before the channel,
        - None : auxiliary noise is not used.
        By default True.
    kwargs :
        Additional arguments passed on to `iss` optimization function.

    Returns
    -------
    qfi : float
        Quantum Fisher information.
    qfis : list[float]
        QFI per algorithm iteration number.
    input_state : np.ndarray
        Density matrix of the optimal input state. The input space of
        the channel goes first and the ancilla second.
    sld : np.ndarray
        Symmetric loagarithmic derivative. The output space of
        the channel goes first and the ancilla second.
    status : bool
        True if the algorithm converged, False otherwise.

    Bibliography
    ------
    .. [1] S. Kurdziałek et al. Quantum metrology using quantum combs and
           tensor network formalism, New Journal of Physics (2024).
           https://doi.org/10.1088/1367-2630/ada8d1
    """
    if not channel.trivial_env:
        warn(ENV_FOR_SINGLE)
        channel = channel.trace_env(env_inp_state)

    if channel.is_comb:
        warn(COMB_FOR_SINGLE)

    output_dim = channel.output_dim
    input_dim = channel.input_dim

    sd = SpaceDict()
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'
    ANCILLA = 'ANCILLA'
    sd[INPUT] = input_dim
    sd[OUTPUT] = output_dim
    sd[ANCILLA] = ancilla_dim

    RHO0 = 'RHO0'
    rho0 = input_state_var([INPUT, ANCILLA], RHO0, sd)

    chann_ten = channel.tensor([INPUT], [OUTPUT], sdict=sd, name='CHANNEL')

    MEASUREMENT = 'MEASUREMENT'
    measure = measure_var([OUTPUT, ANCILLA], MEASUREMENT, sd)

    tn = rho0 * chann_ten * measure

    if artificial_noise_after is None:
        art_noise_spaces = []
    elif artificial_noise_after:
        art_noise_spaces = [[OUTPUT]]
    else:
        art_noise_spaces = [[INPUT]]

    qfi, qfis, new_tn, status = iss_opt(
        tn, art_noise_spaces=art_noise_spaces, **kwargs
    )

    rho0_arr = new_tn.tensors[RHO0].choi([INPUT, ANCILLA])
    sld_arr = new_tn.tensors[MEASUREMENT].choi([OUTPUT, ANCILLA])

    return qfi, qfis[0], rho0_arr, sld_arr, status


def iss_parallel_qfi(channel: ParamChannel, number_of_channels: int,
    ancilla_dim: int, artificial_noise_after: bool | None = None,
    env_inp_state: np.ndarray | None = None, **kwargs
    ) -> tuple[float, list[float], np.ndarray, np.ndarray, bool]:
    """
    Computes quantum Fisher information for channels in parallel using
    iterative see-saw (ISS) algorithm [1]_.

    In parallel strategy all channels are simultaneously probed by an
    entangled input state and their output is collectively measured.

    Parameters
    ----------
    channel : ParamChannel
        Channel to compute quantum Fisher information for.
    number_of_channels : int
        Number of channel uses. In case `channel` is a comb created from
        `m` single channels the total number of channels will be equal to
        `number_of_channels * m`.
    ancilla_dim : int
        Dimension of the ancilla space.
    artificial_noise_after : bool | None, optional
        Whether auxiliary noise is added after the channel:
        - True : auxiliary noise is added after the channel,
        - False : auxiliary noise is added before the channel,
        - None : auxiliary noise is not used.
        By default None.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state.
    kwargs :
        Additional arguments passed on to `iss` optimization function.

    Returns
    -------
    qfi : float
        Qunatum Fisher information.
    qfis : list[float]
        QFI per algorithm iteration number.
    input_state : np.ndarray
        Density matrix of the optimal input state. The input spaces of
        the channels go first and the ancilla goes last.
    sld : np.ndarray
        Symmetric loagarithmic derivative. The output spaces of
        the channels go first and the ancilla goes last.
    status : bool
        True if the algorithm converged, False otherwise.

    Bibliography
    -------
    .. [1] K. Chabuda et al. Tensor-Network Approach for Quantum Metrology
           in Many-Body Quantum Systems, Nature Commun 11, 250 (2020).
           https://doi.org/10.1038/s41467-019-13735-9
    """
    if (channel.env_inp_dim != channel.env_out_dim
        and number_of_channels > 1):
        raise EnvDimsError(channel.env_inp_dim, channel.env_out_dim)

    c = channel.markov_series(number_of_channels).trace_env(env_inp_state)
    c = c.merge_spaces()
    with catch_warnings():
        filterwarnings('ignore', message=COMB_FOR_SINGLE)
        return iss_channel_qfi(
            c, ancilla_dim, None, artificial_noise_after, **kwargs
        )


def iss_tnet_parallel_qfi(channel: ParamChannel, number_of_channels: int,
    ancilla_dim: int, mps_bond_dim: int, measure_bond_dim: int,
    artificial_noise_after: bool | None = None,
    env_inp_state: np.ndarray | None = None, **kwargs
    ) -> tuple[float, list[float], list[np.ndarray], list[np.ndarray],
    bool]:
    """
    Computes quantum Fisher information for channels in parallel using
    iterative see-saw algorithm and tensor networks, that is matrix
    product state (MPS) and matrix product operators (MPOs).

    In parallel strategy all channels are simultaneously probed by an
    entangled input state and their output is collectively measured.

    In the approach with tensor networks the input state is expressed as
    a tensor network in a shape of a line where i-th tensor represents
    part of the input state that goes to the i-th probe channel and
    has is connected with two other tensors (one tensor in case of the
    first and the last one) with bond spaces/indices [1]_.

    The measurement and then the symmetric logarithmic derivative (SLD)
    matrix is expressed as an analogous tensor network (representing
    operators instead of states) called matrix product operator (MPO).

    Parameters
    ----------
    channel : ParamChannel
        Channel to compute quantum Fisher information for.
    number_of_channels : int
        Number of channel uses. In case `channel` is a comb created from
        `m` single channels the total number of channels will be equal to
        `number_of_channels * m`.
    ancilla_dim : int
        Dimension of the ancilla space.
    mps_bond_dim : int
        Dimension of the bond space of the input state which is a matrix
        product state (MPS).
    measure_bond_dim : int
        Dimension of the bond space of the measurement (SLD) which is
        a matrix product operator.
    artificial_noise_after : bool | None, optional
        Whether auxiliary noise is added after the channel:
        - True : auxiliary noise is added after the channel,
        - False : auxiliary noise is added before the channel,
        - None : auxiliary noise is not used.
        By default None.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state.
    kwargs :
        Additional arguments passed on to `iss` optimization function.

    Returns
    -------
    qfi : float
        Qunatum Fisher information.
    qfis : list[float]
        QFI per algorithm iteration number.
    input_state : list[np.array]
        The optimal matrix product state (MPS). The i-th tensor on the list
        - R_i is the input of the i-th channel except the last which is the
        part going to ancilla. Its indices are in the order: bond space
        connecting to the previous tensor (if it exists), input state of
        the channel/ancilla, bond space connecting to the next tensor (if
        it exists).
    sld : list[np.array]
        Matrix product operator (MPO) of the optimal symmetric logarithmic
        derivative (SLD). The i-th tensor on the list - L_i is sld MPO
        elemnt on the output of the i-th channel except the last one which
        is on ancilla. Its indices/spaces are in the order: bond space
        connecting to the previous tensor (if it exists), channel output
        or ancilla, bond space connecting to the next tensor (if it
        exists).
    status : bool
        True if the algorithm converged, False otherwise.

    Bibliography
    -------
    .. [1] K. Chabuda et al. Tensor-Network Approach for Quantum Metrology
           in Many-Body Quantum Systems, Nature Commun 11, 250 (2020).
           https://doi.org/10.1038/s41467-019-13735-9
    """
    if channel.env_inp_dim != channel.env_out_dim:
        if number_of_channels == 1:
            channel = channel.trace_env(env_inp_state)
            # To avoid error for env_inp_tensor declaration:
            env_inp_state = None
        else:
            raise EnvDimsError(channel.env_inp_dim, channel.env_out_dim)

    env_dim = channel.env_inp_dim

    n_combs = number_of_channels
    ch_per_comb = len(channel.input_spaces)
    n_ch = n_combs * ch_per_comb
    n_a = 1
    n = n_ch + n_a

    sd = SpaceDict()
    inp = [('IN', i) for i in range(n_ch)]
    out = [('OUT', i) for i in range(n_ch)]
    for space, dim in zip(inp, cycle(channel.input_dims)):
        sd[space] = dim
    for space, dim in zip(out, cycle(channel.output_dims)):
        sd[space] = dim
    anc = sd.arrange_spaces(n_a, ancilla_dim, 'ANC')
    env = sd.arrange_spaces(n_combs + 1, env_dim, 'ENV')

    RHO0 = 'RHO0'
    mps, mps_names, mps_bonds = mps_var_tnet(
        inp + anc, RHO0, sd, mps_bond_dim
    )

    ENV_INP_STATE = 'ENVIRONMENT INPUT STATE'
    if env_inp_state is None:
        env_inp_state = np.identity(env_dim) / env_dim
    env_inp_tensor = ConstTensor(
        [env[0]], sdict=sd, output_spaces=[env[0]], name=ENV_INP_STATE,
        choi=env_inp_state
    )

    ENV_TRACE = 'ENVIRONMENT TRACE'
    env_trace = choi_identity([env[-1]], sdict=sd, name=ENV_TRACE)

    CHANNEL = 'CHANNEL'
    comb_tensors = []
    comb_ten_names = []
    for i in range(n_combs):
        x = i * ch_per_comb
        y = (i + 1) * ch_per_comb

        name = f'{CHANNEL}, {i}'
        comb_tensor = channel.tensor(
            inp[x:y], out[x:y], env[i], env[i+1], sd, name=name
        )

        comb_ten_names.append(name)
        comb_tensors.append(comb_tensor)
    channels_tnet = TensorNetwork(comb_tensors, sd, CHANNEL)

    MEASUREMENT = 'MEASUREMENT'
    meas_mpo, meas_names, meas_bonds = mpo_measure_var_tnet(
        out + anc, MEASUREMENT, sd, measure_bond_dim
    )

    tn = TensorNetwork(
        [mps, env_inp_tensor, channels_tnet, env_trace, meas_mpo],
        sdict=sd
    )

    contraction_order = []
    for i in range(n):
        contraction_order.append(mps_names[i])
        if i == 0:
            contraction_order.append(ENV_INP_STATE)
        if i < n_ch and i % ch_per_comb == 0:
            j = int(i / ch_per_comb)
            contraction_order.append(comb_ten_names[j])
            if j == n_combs - 1:
                contraction_order.append(ENV_TRACE)
        contraction_order.append(meas_names[i])

    if artificial_noise_after is None:
        noise_spaces = []
    elif artificial_noise_after:
        noise_spaces = [[out[i]] for i in range(n_ch)]
    else:
        noise_spaces = [[inp[i]] for i in range(n_ch)]

    qfi, qfis, new_tn, status = iss_opt(
        tn, art_noise_spaces=noise_spaces,
        contraction_order=contraction_order, **kwargs
    )

    mps_result = []
    sld_result = []
    for i, (mps_name, sld_name) in enumerate(zip(mps_names, meas_names)):
        mps_el = new_tn.tensors[mps_name]
        sld_el = new_tn.tensors[sld_name]

        mps_spaces = []
        sld_spaces = []

        if i > 0:
            mps_spaces.append(mps_bonds[i - 1])
            sld_spaces.append(meas_bonds[i - 1])

        if i < len(mps_names) - 1:
            mps_spaces += [inp[i], mps_bonds[i]]
            sld_spaces += [out[i], meas_bonds[i]]
        else:
            mps_spaces.append(anc[0])
            sld_spaces.append(anc[0])

        mps_ten = mps_el.to_mps(mps_spaces)[-1]
        sld_ten = sld_el.reorder(sld_spaces).array

        mps_result.append(mps_ten)
        sld_result.append(sld_ten)

    return qfi, qfis[0], mps_result, sld_result, status


def iss_adaptive_qfi(channel : ParamChannel, number_of_channels: int,
    ancilla_dim: int, artificial_noise_after: bool | None = None,
    env_inp_state: np.ndarray | None = None, **kwargs
    ) -> tuple[float, list[float], np.ndarray, np.ndarray, bool]:
    """
    Computes quantum Fisher information channels in an adaptive control
    system called quantum comb using iterative see-saw (ISS) algorithm
    [1]_.

    In this approach we consider n copies of a quantum channel Phi:
    
        Phi_i: L(env_i (x) in_i) -> L(env_i+1 (x) out_i)
        for i = 0, ..., n - 1,

    put in a quantum comb C from a set:

        Comb[(Null, in_0), (out_0, in_1), ..., (out_n-2, in_n-1 (x) anc)],

    and a measurement at the end:

        P: L(out_n-1 (x) anc) -> R,

    where (x) denotes a tensor product of Hilbert spaces.

    The input of the first environment space (env_0) is initialized with
    `env_inp_state` and the last environment spcace (env_n) is traced out.

    Parameters
    ----------
    channel : ParamChannel
        Channel to compute quantum Fisher information for. In case this
        argument is a comb created from single channels the control
        operation (comb's tooth) will be put between every **single**
        channel.
    number_of_channels : int
        Number of channel uses. In case `channel` is a comb created from
        `m` single channels the total number of channels will be equal to
        `number_of_channels * m`.
    ancilla_dim : int
        Dimension of the ancilla space connecting controls (teeth),
        dim(anc).
    artificial_noise_after : bool, optional
        Whether auxiliary noise is added after the channel:
        - True : auxiliary noise is added after the channel,
        - False : auxiliary noise is added before the channel,
        - None : auxiliary noise is not used.
        By default True.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state.

    Returns
    -------
    qfi : float
        Quantum Fisher information.
    qfis : list[floats]
        A list of quantum Fisher informations acvhieved in each iteration.
        The last value in the list is the final solution.
    comb : np.ndarray
        Choi matrix of the optimal comb with spaces in order: inp_0, ...,
        inp_n-1, anc, out_0, ..., out_n-2.
    L : np.ndarray
        Optimal syymetric logaritmic derivative (SLD) matrix with spaces
        in order: out_n-1, anc.
    status : bool
        True if the algorithm converged, False otherwise.

    Bibliography
    -------
    .. [1] S. Kurdziałek et al. Quantum metrology using quantum combs and
           tensor network formalism, New Journal of Physics (2024).
           https://doi.org/10.1088/1367-2630/ada8d1
    """
    if channel.env_inp_dim != channel.env_out_dim:
        if number_of_channels == 1:
            channel = channel.trace_env(env_inp_state)
            # To avoid error for env_inp_tensor declaration:
            env_inp_state = None
        else:
            raise EnvDimsError(channel.env_inp_dim, channel.env_out_dim)
    env_dim = channel.env_inp_dim

    # ch_comb for number "channel combs" i.e. combs of channels to
    # differentiate from comb of controls.
    n_ch_comb = number_of_channels
    ch_per_ch_comb = len(channel.input_spaces)
    n = n_ch_comb * ch_per_ch_comb

    sd = SpaceDict()
    inp = [('IN', i) for i in range(n)]
    out = [('OUT', i) for i in range(n)]
    for space, dim in zip(inp, cycle(channel.input_dims)):
        sd[space] = dim
    for space, dim in zip(out, cycle(channel.output_dims)):
        sd[space] = dim
    anc = 'ANC'
    sd[anc] = ancilla_dim
    env = sd.arrange_spaces(n_ch_comb + 1, env_dim, 'ENV')

    COMB = 'COMB'
    comb_structure = []
    for i in range(n):
        tooth_inp = []
        tooth_out = [inp[i]]

        if i > 0:
            tooth_inp.append(out[i - 1])

        if i == n - 1:
            tooth_out.append(anc)

        comb_structure.append((tooth_inp, tooth_out))
    comb = comb_var(comb_structure, sdict=sd, name=COMB)

    ENV_INP_STATE = 'ENVIRONMENT INPUT STATE'
    if env_inp_state is None:
        env_inp_state = np.identity(env_dim) / env_dim
    env_inp_tensor = ConstTensor(
        [env[0]], sdict=sd, output_spaces=[env[0]], name=ENV_INP_STATE,
        choi=env_inp_state
    )

    ENV_TRACE = 'ENVIRONMENT TRACE'
    env_trace = choi_identity([env[-1]], sdict=sd, name=ENV_TRACE)

    CHANNEL = 'CHANNEL'
    ch_comb_ten_names = []
    ch_comb_tensors = []
    for i in range(n_ch_comb):
        x = i * ch_per_ch_comb
        y = (i + 1) * ch_per_ch_comb

        name = f'{CHANNEL}, {i}'
        ch_comb_tensor = channel.tensor(
            inp[x:y], out[x:y], env[i], env[i+1], sd, name=name
        )
        ch_comb_ten_names.append(name)
        ch_comb_tensors.append(ch_comb_tensor)
    channels_tnet = TensorNetwork(ch_comb_tensors, sd, CHANNEL)

    MEASUREMENT = 'MEASUREMENT'
    m_tensor = measure_var([out[n - 1], anc], MEASUREMENT, sd)

    tn = comb * env_inp_tensor * channels_tnet * env_trace * m_tensor

    if artificial_noise_after is None:
        noise_spaces = []
    elif artificial_noise_after:
        noise_spaces = [[out[i]] for i in range(n)]
    else:
        noise_spaces = [[inp[i]] for i in range(n)]

    contraction_order: list[str] = [ENV_INP_STATE]
    contraction_order.append(ch_comb_ten_names[0])
    contraction_order.append(COMB)
    contraction_order += ch_comb_ten_names[1:]
    contraction_order.append(ENV_TRACE)
    contraction_order.append(MEASUREMENT)

    qfi, qfiss, new_tn, status = iss_opt(
        tn, art_noise_spaces=noise_spaces,
        contraction_order=contraction_order, **kwargs
    )

    comb_arr = new_tn.tensors[COMB].choi(inp + [anc] + out[:-1])

    sld_arr = new_tn.tensors[MEASUREMENT].choi([out[-1], anc])

    return qfi, qfiss[0], comb_arr, sld_arr, status


def iss_tnet_adaptive_qfi(channel : ParamChannel, number_of_channels: int,
    ancilla_dim: int, unital_teeth: bool = False,
    initial_teeth: list[np.ndarray] | None = None,
    initial_sld: np.ndarray | None = None,
    artificial_noise_after: bool | None = True,
    fixed_teeth: list[tuple[int, np.ndarray]] | None = None,
    env_inp_state: np.ndarray | None = None, **kwargs
    ) -> tuple[
        float, list[float], list[np.ndarray], np.ndarray, bool
    ]:
    """
    Computes quantum Fisher information in an adaptive control
    system called quantum comb using iterative see-saw (ISS) algorithm
    [1]_.

    In this approach we consider n copies of a quantum channel Phi:
    
        Phi_i: L(env_i (x) in_i) -> L(env_i+1 (x) out_i) 
        for i = 0, ..., n - 1,

    intertwined with n maps called teeth:
    - input state (or 0-th tooth):
    
        T_0: C -> L(in_0 (x) anc_0),
    
    - (proper) teeth:
    
        T_i: L(out_i-1 (x) anc_i-1) -> L(in_i (x) anc_i)
        for i = 1, ..., n - 1,

    which constitute an adaptive control system called quantum comb and a
    measurement at the end:

        P: L(out_n-1 (x) anc_n-1) -> R.

    The input of the first environment space (env_0) is initialized with
    `env_inp_state` and the last environment spcace (env_n) is traced out.

    Parameters
    ----------
    channel : ParamChannel
        Channel to compute quantum Fisher information for. In case this
        argument is a comb created from single channels the control
        operation (tooth) will be put between every **single** channel.
    number_of_channels : int
        Number of channel uses. In case `channel` is a comb created from
        `m` single channels the total number of channels will be equal to
        `number_of_channels * m`.
    ancilla_dim : int
        Dimension of the ancilla space connecting controls (teeth),
        dim(anc_i).
    unital_teeth : bool, optional
        If True than all proper teeth (T_i with i>0) are constrained to be
        unital, that is they preserve identity matrix T_i(Id) = Id, by
        default False.
    initial_teeth : list[np.ndarray] | None, optional
        Initial values of teeth: [T_0, T_1, ...]. First element that is
        the value for the input state should be its density matrix with
        spaces in the order: (inp_0, anc_0). For proper teeth it should be
        Choi matrices with spaces in the order: (inp_i, anc_i, out_i-1,
        anc_i-1), by default None.
    initial_sld : np.ndarray | None, optional
        Initial value of SLD matrix with spaces in order (out_n-1,
        anc_n-1), by default None.
    artificial_noise_after : bool, optional
        Whether auxiliary noise is added after the channel:
        - True : auxiliary noise is added after the channel,
        - False : auxiliary noise is added before the channel,
        - None : auxiliary noise is not used.
        By default True.
    fixed_teeth : list[tuple[int, np.ndarray]] | None, optional
        Teeths to be fixed during optimization. Element (i, Ti) in the
        list means that tooth number i will be fixed to Ti. Note that this
        value will replace the value given in initial_teeth argument, by
        default None.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state.

    Returns
    -------
    qfi : float
        Quantum Fisher information.
    qfis : list[floats]
        A list of quantum Fisher informations acvhieved in each iteration.
        The last value in the list is the final solution.
    Ts : list[np.ndarray]
        List of optimal teeth in the form of density matrix and Choi
        matrices. For the i-th element - T_i spaces are in order:
        - (inp_0, anc_0) for i=0,
        - (inp_i, anc_i, out_i-1, anc_i-1) for i>0.
    L : np.ndarray
        Optimal syymetric logaritmic derivative (SLD) matrix with spaces
        in order: out_n-1, anc.
    status : bool
        True if the algorithm converged, False otherwise.

    Bibliography
    -------
    .. [1] S. Kurdziałek et al. Quantum metrology using quantum combs and
           tensor network formalism, New Journal of Physics (2024).
           https://doi.org/10.1088/1367-2630/ada8d1
    """
    if channel.env_inp_dim != channel.env_out_dim:
        if number_of_channels == 1:
            channel = channel.trace_env(env_inp_state)
            # To avoid error for env_inp_tensor declaration:
            env_inp_state = None
        else:
            raise EnvDimsError(channel.env_inp_dim, channel.env_out_dim)

    if unital_teeth and channel.input_dims != channel.output_dims:
        raise UnitalDimsError(channel.input_dims, channel.output_dims)

    env_dim = channel.env_inp_dim

    # ch_comb for number "channel combs" i.e. combs of channels to
    # differentiate from comb of controls.
    n_ch_comb = number_of_channels
    ch_per_ch_comb = len(channel.input_spaces)
    n = n_ch_comb * ch_per_ch_comb

    sd = SpaceDict()
    inp = [('IN', i) for i in range(n)]
    out = [('OUT', i) for i in range(n)]
    for space, dim in zip(inp, cycle(channel.input_dims)):
        sd[space] = dim
    for space, dim in zip(out, cycle(channel.output_dims)):
        sd[space] = dim
    anc = sd.arrange_spaces(n, ancilla_dim, 'ANC')
    env = sd.arrange_spaces(n_ch_comb + 1, env_dim, 'ENV')

    fixed_teeth_dict = {}
    if fixed_teeth:
        for i, arr in fixed_teeth:
            fixed_teeth_dict[i] = arr

    COMB = 'COMB'
    teeth = []
    teeth_names = []
    for i in range(n):
        tooth_name = f'{COMB}, {i}'
        teeth_names.append(tooth_name)

        if i == 0:
            tooth_inp = []
        else:
            tooth_inp = [out[i - 1], anc[i - 1]]

        tooth_out = [inp[i], anc[i]]

        if i not in fixed_teeth_dict:
            tooth_tensor = VarTensor(
                tooth_out + tooth_inp, sd, tooth_name, tooth_out,
                unital_teeth and i > 0
            )
        else:
            tooth_tensor = ConstTensor(
                tooth_out + tooth_inp, sdict=sd, name=tooth_name,
                output_spaces=tooth_out, choi=fixed_teeth_dict[i]
            )

        teeth.append(tooth_tensor)

    comb = TensorNetwork(teeth, sd, COMB)

    ENV_INP_STATE = 'ENVIRONMENT INPUT STATE'
    if env_inp_state is None:
        env_inp_state = np.identity(env_dim) / env_dim
    env_inp_tensor = ConstTensor(
        [env[0]], sdict=sd, output_spaces=[env[0]], name=ENV_INP_STATE,
        choi=env_inp_state
    )

    ENV_TRACE = 'ENVIRONMENT TRACE'
    env_trace = choi_identity([env[-1]], sdict=sd, name=ENV_TRACE)

    CHANNEL = 'CHANNEL'
    ch_comb_ten_names = []
    ch_comb_tensors = []
    for i in range(n_ch_comb):
        x = i * ch_per_ch_comb
        y = (i + 1) * ch_per_ch_comb

        name = f'{CHANNEL}, {i}'
        ch_comb_tensor = channel.tensor(
            inp[x:y], out[x:y], env[i], env[i+1], sd, name=name
        )
        ch_comb_ten_names.append(name)
        ch_comb_tensors.append(ch_comb_tensor)
    channels_tnet = TensorNetwork(ch_comb_tensors, sd, CHANNEL)

    MEASUREMENT = 'MEASUREMENT'
    m_tensor = measure_var([out[-1], anc[-1]], MEASUREMENT, sd)

    tn = comb * env_inp_tensor * channels_tnet * env_trace * m_tensor

    init_tensors = []

    if initial_teeth is not None:
        for i, init_tooth in enumerate(initial_teeth):
            spaces = [inp[i], anc[i]]
            if i > 0:
                spaces += [out[i - 1], anc[i - 1]]

            _ct = ConstTensor(
                spaces, choi=init_tooth, sdict=sd,
                output_spaces=[inp[i], anc[i]], name=teeth_names[i],
            )
            init_tensors.append(_ct)
    if initial_sld is not None:
        t = ConstTensor(
            [out[-1], anc[-1]], initial_sld, sd, name=MEASUREMENT
        )
        init_tensors.append(t)
    init_tn = TensorNetwork(init_tensors, sd) if init_tensors else None

    if artificial_noise_after is None:
        noise_spaces = []
    elif artificial_noise_after:
        noise_spaces = [[out[i]] for i in range(n)]
    else:
        noise_spaces = [[inp[i]] for i in range(n)]

    contraction_order: list[str] = []
    for i in range(n):
        contraction_order.append(teeth_names[i])
        if i == 0:
            contraction_order.append(ENV_INP_STATE)
        if i % ch_per_ch_comb == 0:
            j = int(i / ch_per_ch_comb)
            contraction_order.append(ch_comb_ten_names[j])
    contraction_order.append(ENV_TRACE)
    contraction_order.append(MEASUREMENT)

    qfi, qfiss, new_tn, status = iss_opt(
        tn, init_tn=init_tn, art_noise_spaces=noise_spaces,
        contraction_order=contraction_order, **kwargs
    )

    teeth_arrs = []
    for i in range(n):
        t = new_tn.tensors[teeth_names[i]]

        spaces: list[Hashable] = [inp[i], anc[i]]
        if i > 0:
            spaces += [out[i - 1], anc[i - 1]]

        t_arr = t.choi(spaces)
        teeth_arrs.append(t_arr)

    sld_arr = new_tn.tensors[MEASUREMENT].choi([out[n - 1], anc[n - 1]])

    return qfi, qfiss[0], teeth_arrs, sld_arr, status


def iss_tnet_collisional_qfi(channel: ParamChannel,
    number_of_channels: int, ancilla_dim: int, mps_bond_dim: int,
    measure_bond_dim: int, unital_teeth: bool = False,
    initial_teeth: list[np.ndarray] | None = None,
    artificial_noise_after: bool | None = True,
    fixed_teeth: list[tuple[int, np.ndarray]] | None = None,
    env_inp_state: np.ndarray | None = None, **kwargs
    ) -> tuple[
        float, list[float], list[np.ndarray], list[np.ndarray],
        list[np.ndarray], bool
    ]:
    """
    Returns the quantum Fisher information in a scenario with channels in
    a comb which teeth have separate ancillas using iterative see-saw
    (ISS) algorithm. This is a very similar structure to the one in the
    adaptive strategy the only difference being that comb's teeth cannot
    communicate with each other using their common ancilla space.

    More precisely, in this approach we consider n copies of a quantum
    channel Phi:
    
        Phi_i: L(env_i (x) in_i) -> L(env_i+1 (x) out_i) 
        for i = 0, ..., n - 1,

    an input state:

        rho0 in L(in_0 (x) anc_0 (x) ... (x) anc_n-2),
    
    controls called also teeth:
        
        T_i: L(out_i-1 (x) anc_i-1) -> L(in_i (x) anc'_i-1)
        for i = 1, ..., n - 1,

    and measurement at the end:

        P: L(anc'_0 (x) ... (x) anc'_n-2 (x) out_n-1) -> R.

    The input of the first environment space (env_0) is initialized with
    `env_inp_state` and the last environment spcace (env_n) is traced out.

    Parameters
    ----------
    channel : ParamChannel
        Channel to compute quantum Fisher information for. In case this
        argument is a comb created from single channels the control
        operation (comb's tooth) will be put between every **single**
        channel.
    number_of_channels : int
        Number of channel uses. In case `channel` is a comb created from
        `m` single channels the total number of channels will be equal to
        `number_of_channels * m`.
    ancilla_dim : int
        Dimension of the ancilla space connecting controls (teeth).
    mps_bond_dim : int
        Dimension of the bond space of the input state which is a matrix
        product state.
    measure_bond_dim : int
        Dimension of the bond space of the measurement (SLD matrix) which
        is a matrix product operator.
    unital_teeth : bool, optional
        If True than all proper teeth (T_i with i>0) are constrained to be
        unital, that is they preserve identity matrix T_i(Id) = Id, by
        default False.
    initial_teeth : list[np.ndarray] | None, optional
        Initial values of teeth: [T_1, ...]. For T_i it should be its
        Choi matrix with spaces in the order: (inp_i, anc'_i-1, out_i-1,
        anc_i-1), by default None.
    artificial_noise_after : bool, optional
        Whether auxiliary noise is added after the channel:
        - True : auxiliary noise is added after the channel,
        - False : auxiliary noise is added before the channel,
        - None : auxiliary noise is not used.
        By default True.
    fixed_teeth : list[tuple[int, np.ndarray]] | None, optional
        Teeths to be fixed during optimization. Element (i, Ti) in the
        list means that tooth number i will be fixed to Ti. Note that this
        value will replace the value given in initial_teeth argument, by
        default None.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state.
    
    Returns
    -------
    qfi : float
        Quantum Fisher information.
    qfis : list[floats]
        A list of quantum Fisher informations computed in each iteration.
        The last value in the list is the final solution.
    input_state : list[np.array]
        The optimal matrix product state (MPS). The i-th tensor on the list
        - R_i is the input of the i-th channel except the last which is the
        part going to ancilla. Its indices are in the order: bond space
        connecting to the previous tensor (if it exists), input state of
        the channel/ancilla, bond space connecting to the next tensor (if
        it exists).
    Ts : list[np.ndarray]
        List of optimal teeth [T_1, T_2, ...] in the form of their Choi
        matrices. For the i-th element - T_i spaces are in order (inp_i,
        anc'_i-1, out_i-1, anc_i-1).
    sld : list[np.array]
        Matrix product operator (MPO) of the optimal symmetric logarithmic
        derivative (SLD). For i < n-1 i-th tensor on the list is a sld
        element on anc'_i then the last tensor on the list is a sld
        element on space out_n-1. Indices/spaces of the i-th tnesor are in
        the order: bond space connecting to the previous tensor (if it
        exists), out_n-1 or anc'_i-1, bond space connecting to the next
        tensor (if it exists).
    status : bool
        True if the algorithm converged, False otherwise.
    
    Bibliography
    -------
    .. [1] K. Chabuda et al. Tensor-Network Approach for Quantum Metrology
           in Many-Body Quantum Systems, Nature Commun 11, 250 (2020).
           https://doi.org/10.1038/s41467-019-13735-9
    .. [2] S. Kurdziałek et al. Quantum metrology using quantum combs and
           tensor network formalism, New Journal of Physics (2024).
           https://doi.org/10.1088/1367-2630/ada8d1
    """
    if channel.env_inp_dim != channel.env_out_dim:
        if number_of_channels == 1:
            channel = channel.trace_env(env_inp_state)
            # To avoid error for later env_inp_tensor declaration:
            env_inp_state = None
        else:
            raise EnvDimsError(channel.env_inp_dim, channel.env_out_dim)
    env_dim = channel.env_inp_dim

    if unital_teeth and channel.input_dims != channel.output_dims:
        raise UnitalDimsError(channel.input_dims, channel.output_dims)

    # ch_comb for number "channel combs" i.e. combs of channels to
    # differentiate from comb of controls.
    n_ch_comb = number_of_channels
    ch_per_ch_comb = len(channel.input_spaces)
    n = n_ch_comb * ch_per_ch_comb

    sd = SpaceDict()
    inp = [('IN', i) for i in range(n)]
    out = [('OUT', i) for i in range(n)]
    for space, dim in zip(inp, cycle(channel.input_dims)):
        sd[space] = dim
    for space, dim in zip(out, cycle(channel.output_dims)):
        sd[space] = dim
    anc_pre = sd.arrange_spaces(n - 1, ancilla_dim, ('ANC', 'PRE'))
    anc_post = sd.arrange_spaces(n - 1, ancilla_dim, ('ANC', 'POST'))
    env = sd.arrange_spaces(n_ch_comb + 1, env_dim, 'ENV')

    RHO0 = 'RHO0'
    mps, mps_names, mps_bonds = mps_var_tnet(
        [inp[0]] + anc_pre, RHO0, sd, mps_bond_dim
    )

    fixed_teeth_dict = {}
    if fixed_teeth:
        for i, arr in fixed_teeth:
            fixed_teeth_dict[i] = arr

    TOOTH = 'TOOTH'
    teeth = []
    teeth_names = []
    for i in range(n - 1):
        tooth_name = f'{TOOTH}, {i}'
        teeth_names.append(tooth_name)
        tooth_inp = [out[i], anc_pre[i]]
        tooth_out = [inp[i + 1], anc_post[i]]

        if i in fixed_teeth_dict:
            tooth = ConstTensor(
                tooth_out + tooth_inp, sdict=sd, name=tooth_name,
                output_spaces=tooth_out, choi=fixed_teeth_dict[i]
            )
        else:
            tooth = VarTensor(
                tooth_out + tooth_inp, sdict=sd, name=tooth_name,
                output_spaces=tooth_out, is_unital=unital_teeth
            )
        teeth.append(tooth)

    ENV_INP_STATE = 'ENVIRONMENT INPUT STATE'
    if env_inp_state is None:
        env_inp_state = np.identity(env_dim) / env_dim
    env_inp_tensor = ConstTensor(
        [env[0]], sdict=sd, output_spaces=[env[0]], name=ENV_INP_STATE,
        choi=env_inp_state
    )

    ENV_TRACE = 'ENVIRONMENT TRACE'
    env_trace = choi_identity([env[-1]], sdict=sd, name=ENV_TRACE)

    CHANNEL = 'CHANNEL'
    ch_names = []
    chans = []
    for i in range(n_ch_comb):
        x = i * ch_per_ch_comb
        y = (i + 1) * ch_per_ch_comb

        channel_name = f'{CHANNEL}, {i}'
        chan = channel.tensor(
            inp[x:y], out[x:y], env[i], env[i+1], sd, name=channel_name
        )

        ch_names.append(channel_name)
        chans.append(chan)
    channels = TensorNetwork(chans, sd, CHANNEL)

    MEASUREMENT = 'MEASUREMENT'
    meas_mpo, meas_names, meas_bonds = mpo_measure_var_tnet(
        anc_post + [out[-1]], MEASUREMENT, sd, measure_bond_dim
    )

    tn = TensorNetwork(
        [mps, *teeth, env_inp_tensor, channels, env_trace, meas_mpo], sd
    )

    init_teeth = []
    if initial_teeth:
        for i, t_arr in enumerate(initial_teeth):
            _c = ConstTensor(
                [inp[i + 1], anc_post[i], out[i], anc_pre[i]], sdict=sd,
                name=teeth_names[i], choi=t_arr,
                output_spaces=[inp[i + 1], anc_post[i]]
            )
            init_teeth.append(_c)
    # TODO: Add initial input state and initial sld.
    init_tn = TensorNetwork(init_teeth, sd) if init_teeth else None

    if artificial_noise_after is None:
        noise_spaces = []
    elif artificial_noise_after:
        noise_spaces = [[out[i]] for i in range(n)]
    else:
        noise_spaces = [[inp[i]] for i in range(n)]

    contraction_order = []
    for i in range(n):
        contraction_order.append(mps_names[i])
        if i == 0:
            contraction_order.append(ENV_INP_STATE)
        if i % ch_per_ch_comb == 0:
            j = int(i / ch_per_ch_comb)
            contraction_order.append(ch_names[j])
        if i < n - 1:
            contraction_order.append(teeth_names[i])
        contraction_order.append(meas_names[i])
    contraction_order.append(ENV_TRACE)

    qfi, qfiss, new_tn, status = iss_opt(
        tn, init_tn=init_tn, art_noise_spaces=noise_spaces,
        contraction_order=contraction_order, **kwargs
    )

    mps_arrs = []
    sld_arrs = []
    for i in range(n):
        mps_el = new_tn.tensors[mps_names[i]]
        sld_el = new_tn.tensors[meas_names[i]]
        mps_spaces = []
        sld_spaces = []

        if i == 0:
            mps_spaces.append(inp[0])
        else:
            mps_spaces += [mps_bonds[i - 1], anc_pre[i - 1]]
            sld_spaces.append(meas_bonds[i - 1])

        if i < n - 1:
            mps_spaces.append(mps_bonds[i])
            sld_spaces += [anc_post[i], meas_bonds[i]]
        else:
            sld_spaces.append(out[-1])

        mps_arrs.append(mps_el.to_mps(mps_spaces)[-1])

        sld_el.reorder(sld_spaces)
        sld_arrs.append(sld_el.array)

    teeth_arrs = []
    for i in range(n - 1):
        t = new_tn.tensors[teeth_names[i]]
        t_arr = t.choi([inp[i + 1], anc_post[i], out[i], anc_pre[i]])
        teeth_arrs.append(t_arr)

    return qfi, qfiss[0], mps_arrs, teeth_arrs, sld_arrs, status
