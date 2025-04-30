from __future__ import annotations

from collections.abc import Iterable
from warnings import warn, catch_warnings, filterwarnings

import cvxpy as cp
import numpy as np

from ..utils import hc
from ..param_channel import ParamChannel
from ..qtools import minimize_alpha, comb_variables, krauses_kron

from .errors import EnvDimsError
from .warnings import ENV_FOR_SINGLE, COMB_FOR_SINGLE




def mop_channel_qfi(channel: ParamChannel,
    env_inp_state: np.ndarray | None = None
    ) -> float:
    """
    Computes quantum Fisher information for single parametrized channel
    using minimization over purifications (MOP) method [1]_.

    Parameters
    ----------
    channel : ParamChannel
        Channel to compute quantum Fisher information for. In case this
        argument is a comb created from single channels the whole comb
        will be treated as a single channel.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state.

    Returns
    -------
    qfi : float 
        maximal QFI of a single channel.

    Bibliography
    --------
    .. [1] Demkowicz-Dobrzański, R., Kołodyński, J. & Guţă, M. 
           The elusive Heisenberg limit in quantum-enhanced metrology.
           Nat Commun 3, 1063 (2012). https://doi.org/10.1038/ncomms2067
    """
    if not channel.trivial_env:
        warn(ENV_FOR_SINGLE)
        channel = channel.trace_env(env_inp_state)
    if channel.is_comb:
        warn(COMB_FOR_SINGLE)
    return 4 * minimize_alpha(*channel.dkrauses())


def mop_parallel_qfi(channel: ParamChannel, number_of_channels: int,
    env_inp_state: np.ndarray | None = None) -> float:
    """
    Computes quantum Fisher information in a scenario with parallel
    channels.

    Parameters
    ----------
    channel : ParamChannel
        Channel to compute quantum Fisher information for.
    number_of_channels : int, optional
        Number of channel uses. In case `channel` is a comb created from
        `m` single channels the total number of channels will be equal to
        `number_of_channels * m`.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state.

    Returns
    -------
    qfi : float
        Qunatum Fisher information.

    Bibliography
    --------
    .. [1] Demkowicz-Dobrzański, R., Kołodyński, J. & Guţă, M. 
           The elusive Heisenberg limit in quantum-enhanced metrology.
           Nat Commun 3, 1063 (2012). https://doi.org/10.1038/ncomms2067
    """
    if (channel.env_inp_dim != channel.env_out_dim
        and number_of_channels > 1):
        raise EnvDimsError(channel.env_inp_dim, channel.env_out_dim)

    with catch_warnings():
        filterwarnings('ignore', message=COMB_FOR_SINGLE)
        comb = channel.markov_series(number_of_channels)
        comb = comb.trace_env(env_inp_state)
        return mop_channel_qfi(comb)


def mop_adaptive_qfi(channel: ParamChannel, number_of_channels: int,
    env_control: bool | tuple[bool, bool] = False,
    env_inp_state: np.ndarray | None = None, input_pure_qfi: float = 0,
    print_solver_messages = False) -> float:
    """
    Computes the comb QFI using minimization over purifications method
    (MOP).

    This function computes the QFI of comb consisting of
    `number_of_channels` channels connected with their environments.
    Notice that `channel` may be a single channel or a comb created by the
    user. The resulting QFI is achievable when arbitrary adaptive control
    can be applied in each step. Control may act between channels, but
    also between different teeth of  `channel`, when `channel` represents
    a comb.

    Parameters
    ----------
    channel: ParamChannel
        Parametrized channel characterizing signal encoding. Might be a
        single channel or a comb.
    number_of_channels : int
        Number of elementary channels/combs linked to create the full
        comb of interest.
    env_control: bool | tuple[bool, bool], optional
        If True, then adaptive control may also act on first input
        and last output environment. If tuple, then
        its elements refer to input and output environment respectively.
    env_inp_state : np.ndarray | None, optional
        Density matrix of the initial state of the environment. If None
        then it becomes a maximally mixed state. This argument is
        ignored when environment input is controlled.
    input_pure_qfi : float, optional
        The QFI of an additional pure parameter-dependent state.
        This state can be treated as a first tooth of estimated comb.
        By default, there is no such a state (input_pure_qfi = 0).
        Non-zero values are typically used to calculate bounds.
    print_solver_messages : bool, optional
        Whether to print solver messages to console during computation, by
        default False.

    Returns
    -------
    float
        Quantum Fisher Information optimized over all comb controls.
      
    Bibliography
    ------------
    .. [1] Altherr, A. & Yang, Y.(2021). 
       "Quantum Metrology for Non-Markovian Processes"
       Physical Review Letters, 127, 060501
       https://doi.org/10.1103/PhysRevLett.127.060501
    """
    env_inp_dim = channel.env_inp_dim
    env_out_dim = channel.env_out_dim

    # Processing env_control parameter
    if isinstance(env_control, Iterable):
        env_inp_control, env_out_control = env_control
    else:
        env_inp_control = env_out_control = env_control

    # Creating the full comb for which the QFI will be computed
    comb_channel = channel.markov_series(number_of_channels)

    # Contracting input state with input environment (if env not
    # controlled)
    if not env_inp_control and not comb_channel.trivial_env_inp:
        comb_channel = comb_channel.trace_env_inp(env_inp_state)

    #number of teeth of created comb
    teeth_number = len(comb_channel.input_spaces)

    #names of all input and output spaces according to causal order
    #these lists include environment if controlled
    if env_inp_control:
        #CAREFUL: order matters in the line below!
        input_spaces = [comb_channel.env_inp] + comb_channel.input_spaces
    else:
        input_spaces = comb_channel.input_spaces

    if env_out_control:
        #CAREFUL: order matters in the line below!
        output_spaces = comb_channel.output_spaces + [comb_channel.env_out]
    else:
        output_spaces = comb_channel.output_spaces

    # creating list of dimension of all spaces according to causal order
    dims = []
    for i in range(teeth_number):
        dims.append(comb_channel.input_dims[i])
        dims.append(comb_channel.output_dims[i])

    # environment is merged with first input/ last output space
    if env_inp_control:
        dims[0] *= env_inp_dim
    if env_out_control:
        dims[-1] *= env_out_dim

    # Kraus operators are created with reverse order of input and output
    # spaces
    krauses_comb, dkrauses_comb = comb_channel.tensor().dkrauses(
        input_spaces[::-1], output_spaces[::-1]
    )

    #add additional pure state to existing Krauses if input_pure_qfi > 0
    if input_pure_qfi != 0:
        # Input pure state and its derivative
        Ku = [np.array([[1], [0]])]
        dKu = [np.array([[0], [np.sqrt(input_pure_qfi) / 2]])]

        # Creating Kraus operators of extended comb (with input state)
        krauses_comb, dkrauses_comb = krauses_kron(
            krauses_comb, dkrauses_comb, Ku, dKu
        )

        #updating dimensions
        dims = np.concatenate([[1, 2], dims])

    # Number of Kraus operators and total dimension
    num_kraus = len(krauses_comb)
    total_dim = np.prod(dims)

    # Decomposition vectors of Choi operator
    C_vectors = np.array([K.flatten() for K in krauses_comb])
    dC_vectors = np.array([dK.flatten() for dK in dkrauses_comb])

    # Formulate the SDP problem
    dims_traced_output = list(dims).copy()
    dims_traced_output[-1] = 1

    #Comb variables and constraints
    #trace_var: variable, plays the role of lambda from [1]
    combs, constraints, trace_var = comb_variables(
        tuple(dims_traced_output), trace_constraint=None
    )

    # h matrix generating different Kraus representation
    h = cp.Variable((num_kraus, num_kraus), hermitian=True)

    # Create matrix containing `c` vectors (last dimension of `C` traced
    # out) horizontal `c` vectors are stacked vertically
    c_list_shape = num_kraus * dims[-1], total_dim // dims[-1]
    clist = np.reshape(C_vectors, c_list_shape)

    #the same for derivatives
    dclist = np.reshape(dC_vectors, c_list_shape)

    #derivatives transformed by matrix `h`
    tclist = dclist - 1j * cp.kron(h, np.eye(dims[-1])) @ clist

    #Create block matrix `A` (constrained to be positive)
    A00 = np.eye(num_kraus * dims[-1])
    A01 = tclist
    A10 = hc(A01)
    A11 = combs[-1]

    A = cp.bmat([[A00, A01], [A10, A11]])
    constraints.append(A >> 0)

    # Define the objective and solve
    obj = cp.Minimize(trace_var)
    prob = cp.Problem(obj, constraints)
    sol = prob.solve(verbose=print_solver_messages)

    return 4 * sol
