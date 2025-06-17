from __future__ import annotations

from warnings import warn

import cvxpy as cp
import numpy as np

from ..qtools import comb_variables, swap_operator, minimize_alpha, hc
from ..protocols import mop_adaptive_qfi
from ..param_channel import ParamChannel




def minimize_beta(krauses: list[np.ndarray], dkrauses: list[np.ndarray]
    ) -> float:
    """
    Minimize the norm of beta over all Kraus representations for a given
    channel.

    This function calculates the minimum norm of beta over all possible
    Kraus representations of a channel, given a list of Kraus operators
    and their derivatives.

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
        The minimum value of norm of beta over all Kraus representations.
        
    Bibliography
    --------
    .. [1] Demkowicz-Dobrzański, R., Kołodyński, J. & Guţă, M. 
       The elusive Heisenberg limit in quantum-enhanced metrology. 
       Nat Commun 3, 1063 (2012). https://doi.org/10.1038/ncomms2067
    """
    # Number of Kraus operators
    num_kraus = len(krauses)

    # Dimensions of the input and output spaces
    dout = krauses[0].shape[0] # Output dimension
    din = krauses[0].shape[1] # Input dimension

    # Concatenate Kraus operators and their derivatives along the first
    # axis
    K = np.concatenate(krauses)
    dK = np.concatenate(dkrauses)

    # Define the hermitian matrix h and scalar t for minimization
    h = cp.Variable((num_kraus, num_kraus), hermitian=True)
    t = cp.Variable()

    # Construct the block matrix B for the semidefinite constraint
    B00 = t * np.eye(din) # Top-left block
    B11 = t * np.eye(din) # Bottom-right block
    B01 = hc(dK - 1j * cp.kron(h, np.eye(dout)) @ K) @ K
    # Top-right block
    B10 = hc(B01) # Bottom-left block
    B = cp.bmat([
        [B00, B01],
        [B10, B11]
    ])

    # Define constraints and objective for the optimization problem
    constraints = [B >> 0] # Positive semidefinite constraint
    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem with the specified solver if provided
    return problem.solve()


def minimize_alpha_given_beta(krauses: list[np.ndarray],
    dkrauses: list[np.ndarray], bmax: float) -> float:
    """
    Minimize the norm of alpha given a constraint on norm of beta.

    This function calculates the minimum norm of alpha over all possible
    Kraus representations for a channel, given a constraint that norm of
    beta is less than or equal to a specified maximum (`bmax`).

    Parameters
    ----------
    krauses : list[np.ndarray]
        List of Kraus operators, each represented as a 2D NumPy array.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators, each represented as a 2D
        NumPy array.
    bmax : float
        Upper bound on beta for the optimization.

    Returns
    -------
    float
        The minimum value of alpha given the constraint on beta.
        
    Bibliography
    --------
    .. [1] Kurdziałek, S., Górecki, W., Albarelli, F., &
        Demkowicz-Dobrzański, R. (2023). Using adaptiveness and causal
        superpositions against noise in quantum metrology. Physical Review
        Letters, 131(9), 090801.
    """
    # Number of Kraus operators
    num_kraus = len(krauses)

    # Dimensions of the input and output spaces
    dout = krauses[0].shape[0]  # Output dimension
    din = krauses[0].shape[1]   # Input dimension

    # Concatenate Kraus operators and their derivatives along the 1st axis
    K = np.concatenate(krauses)
    dK = np.concatenate(dkrauses)

    # Define the hermitian matrix h and scalar t1 for minimization
    h = cp.Variable((num_kraus, num_kraus), hermitian=True)
    t = cp.Variable()
    constraints = []

    # Construct the block matrix A for the alpha constraint
    A00 = t * np.eye(din)  # Top-left block for alpha
    A11 = np.eye(dout * num_kraus)  # Bottom-right block
    A10 = dK - 1j * cp.kron(h, np.eye(dout)) @ K  # Top-right block
    A01 = hc(A10)  # Bottom-left block
    A = cp.bmat([[A00, A01], [A10, A11]])
    constraints.append(A >> 0)  # Constraint enforcing t >= ||alpha||

    # Construct the block matrix B for the beta constraint
    B00 = bmax * np.eye(din)  # Top-left block for beta constraint
    B11 = bmax * np.eye(din)  # Bottom-right block
    B01 = A01 @ K  # Top-right block
    B10 = hc(B01)  # Bottom-left block
    B = cp.bmat([[B00, B01], [B10, B11]])

    if bmax == 0:
        #simplify problem for bmax = 0
        constraints.append(B01 == np.zeros(B01.shape))
    else:
        # Constraint enforcing bmax**2 >= ||beta||**2
        constraints.append(B >> 0)

    # Define the objective for the optimization problem
    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem and return the minimum value of alpha
    return problem.solve()


def beta_alpha_chart(krauses: list[np.ndarray],
    dkrauses: list[np.ndarray], p: int = 20, eps: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a chart of minimal alpha values given constraints on beta.

    This function calculates a list of minimal alpha values (`a_list`)
    for a range of beta values (`b_list`) for a given channel, for beta
    values between its minimum and the square root of the minimal alpha
    value.

    Parameters
    ----------
    krauses : list[np.ndarray]
        List of Kraus operators, each represented as a 2D NumPy array.
    dkrauses : list[np.ndarray]
        List of derivatives of Kraus operators, each represented as a 2D
        NumPy array.
    p : int, optional
        Number of beta samples, by default 20.
    eps : float, optional
        Small epsilon added to beta bounds to avoid numerical issues, by
        default 0.0.

    Returns
    -------
    b_list : np.ndarray
        Array of beta values sampled between the minimum beta and maximum
        alpha.
    a_list : np.ndarray
        Array of minimal alpha values corresponding to each beta in
        `b_list`.

    Notes
    -----
    This function first calculates `bmin` as the minimum achievable beta
    for the channel, and `bmax` as the square root of the minimum alpha.
    The `b_list` array is then generated with `n` samples between
    `bmin + eps` and `bmax + 2*eps`. For each beta value in `b_list`, the
    minimum alpha under that beta constraint is calculated using
    `minimize_alpha_given_beta`.

    Bibliography
    --------
    .. [1] Kurdziałek, S., Górecki, W., Albarelli, F., &
        Demkowicz-Dobrzański, R. (2023). Using adaptiveness and causal
        superpositions against noise in quantum metrology. Physical Review
        Letters, 131(9), 090801.
    """
    # Calculate minimum beta and maximum beta (based on minimum alpha)
    bmin = minimize_beta(krauses, dkrauses)
    bmax = np.sqrt(minimize_alpha(krauses, dkrauses))

    # Generate a list of beta values from bmin+eps to bmax+2*eps, with n
    # samples
    b_list = np.linspace(bmin + eps, bmax + 2 * eps, p)

    # Calculate the minimal alpha for each beta constraint in b_list
    a_list = np.array(
        [minimize_alpha_given_beta(krauses, dkrauses, b) for b in b_list]
    )

    return b_list, a_list


def par_bound_single_n(channel: ParamChannel, n: int) -> float:
    """
    Calculate parallel (PAR) bound for the Quantum Fisher Information
    (QFI) for a specific number of channels `n`.

    Parameters
    ----------
    channel : ParamChannel
        Object representing a single channel and its derivative
    n : int
        Number of channels for which the bounds are computed.

    Returns
    -------
    float
        PAR bounds for QFI for `n` channels probed in parallel.
    
    Bibliography
    --------
    .. [1] Kołodyński, J., & Demkowicz-Dobrzański, R. (2013).
       Efficient tools for quantum metrology with uncorrelated noise.
       New Journal of Physics, 15(7), 073043.            
    """
    if not channel.trivial_env:
        raise ValueError(
            "This function doesn't work for correlated channels"
            "The input channel must have env_dim equal to 1."
        )

    if not channel.single_tooth:
        warn(
            "A non-trivial comb (more than one tooth) was provided."
            "In this function, causal structure is ignored and all"
            "inputs/outputs are merged into single input/output.",
            UserWarning
        )
    
    krauses, dkrauses = channel.dkrauses()

    # Number of Kraus operators
    num_kraus = len(krauses)

    # Dimensions of the input and output spaces
    dout = krauses[0].shape[0]  # Output dimension
    din = krauses[0].shape[1]   # Input dimension

    # Concatenate Kraus operators and their derivatives along the first axis
    K = np.concatenate(krauses)
    dK = np.concatenate(dkrauses)

    # Define the hermitian matrix h and scalars a, b for minimization
    h = cp.Variable((num_kraus, num_kraus), hermitian=True)
    a = cp.Variable()
    b = cp.Variable()
    constraints = []

    # Construct the block matrix A for the alpha constraint
    A00 = a * np.eye(din)  # Top-left block for alpha
    A11 = np.eye(dout * num_kraus)  # Bottom-right block
    A10 = dK - 1j * cp.kron(h, np.eye(dout)) @ K  # Top-right block
    A01 = hc(A10)  # Bottom-left block
    A = cp.bmat([[A00, A01], [A10, A11]])
    constraints.append(A >> 0)  # Constraint enforcing a >= ||alpha||

    # Construct the block matrix B for the beta constraint
    B00 = b * np.eye(din)  # Top-left block for beta constraint
    B11 =  np.eye(din)  # Bottom-right block
    B01 = A01 @ K  # Top-right block
    B10 = hc(B01)  # Bottom-left block
    B = cp.bmat([[B00, B01], [B10, B11]])
    constraints.append(B >> 0)  # Constraint enforcing b >= ||beta||^2

    # Define the objective for the optimization problem
    objective = cp.Minimize(a + (n-1) * b)
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem and return the bound for QFI
    return 4 * n * problem.solve()


def par_bounds(channel: ParamChannel, nmax: int,
     method: str = 'default', p: int = 40, eps: float = 0.0
    ) -> np.ndarray:
    """
    Calculate parallel (par) bounds for the Quantum Fisher Information
    (QFI).
    
    This function computes the set of PAR bounds for the QFIs of up to
    `nmax` channels probed paralelly using arbitrary, possibly entangled
    probe (with ancilla).
    
    Parameters
    ----------
    channel : ParamChannel
        Object representing a single channel and its derivative
    nmax : int
        Maximum number of channels for which the bounds are computed.
    method: str, optional
        Method used to calculate bounds, there are two possibilities:
        - 'default': calculates the bound by solving SDP for each number
            of channels separately. This method calls function
            `par_bound_single_n` for each n.
        - 'ab_chart': construct minimal alpha vs beta chart and then
            calculates the bound for each n. Faster for large `nmax`, may
            give slightly less tight bounds then 'default' (but
            asymptotically both methods are equivalent).
    p : int, optional
        Number of samples of beta for precision in bound calculation, by
        default 40, used for 'ab_chart' method only.
    eps : float, optional
        Small epsilon added to beta bounds in sampling to avoid numerical
        issues, by default 0.0, used for 'ab_chart' method only.
        
    Returns
    -------
    np.ndarray
        Array of PAR bounds for QFI with up to `nmax` channels (starting
        from n=1).
    
    Bibliography
    --------
    .. [1] Kołodyński, J., & Demkowicz-Dobrzański, R. (2013).
       Efficient tools for quantum metrology with uncorrelated noise.
       New Journal of Physics, 15(7), 073043.
       
    .. [1] Kurdziałek, S., Górecki, W., Albarelli, F., &
        Demkowicz-Dobrzański, R. (2023). Using adaptiveness and causal
        superpositions against noise in quantum metrology. Physical Review
        Letters, 131(9), 090801.
    """
    if not channel.trivial_env:
        raise ValueError(
            "This function doesn't work for correlated channels"
            "The input channel must have env_dim equal to 1."
        )

    # minimal n for which warning is displayed when 'default' method is
    # picked
    AB_CHART_WARNING_THRESHOLD = 500
    AB_CHART = 'ab_chart'
    DEFAULT = 'default'

    if method == DEFAULT:
        if nmax > AB_CHART_WARNING_THRESHOLD:
            warn(
                f"The 'default' method may be slow for nmax={nmax}."
                "The 'ab_chart' may be significantly faster.",
                UserWarning
            )
        return np.array([
            par_bound_single_n(channel, n)
            for n in range(1, nmax + 1)
        ])

    if method == AB_CHART:
        krauses, dkrauses = channel.dkrauses()

        if not channel.single_tooth:
            warn(
                "A non-trivial comb (more than one tooth) was provided."
                "In this function, causal structure is ignored and all"
                "inputs/outputs are merged into single input/output.",
                UserWarning
            )

        b, a = beta_alpha_chart(krauses, dkrauses, p, eps)
        return 4 * np.array([
            np.min(n*a + n*(n-1) * b**2) for n in range(1, nmax + 1)
        ])

    raise ValueError(
        f"Invalid `method` argument {method}.\n"
        "Possible methods: 'default', 'ab_chart'."
    )


def ad_bounds(channel: ParamChannel,
    nmax: int, method: str = 'default', p: int = 40, eps: float = 0.0
    ) -> np.ndarray:
    """
    Calculate adaptive (AD) bounds for the Quantum Fisher Information
    (QFI).
    
    This function computes the set of AD bounds for the QFIs of up to
    `nmax` channels probed sequentially with arbitrary controls between,
    utilizing arbitrarily large ancilla.
    
    Parameters
    ----------
    channel : ParamChannel
        Object representing a single channel and its derivative
    nmax : int
        Maximum number of channels for which the bounds are computed.
    method: str, optional
        Method used to calculate bounds, there are two possibilities:
        - 'default': calculates the bound using the iterative procedure
          involing calculation of extended comb QFI described in [2]_.
        - 'ab_chart': construct minimal alpha vs beta chart and then
            calculates the bound for each n [1]_. Faster for large `nmax`,
            may give slightly less tight bounds then 'default' (but
            asymptotically both methods are equivalent).
    p : int, optional
        Number of samples of beta for precision in bound calculation, by
        default 40, used for 'ab_chart' method only.
    eps : float, optional
        Small epsilon added to beta bounds in sampling to avoid numerical
        issues, by default 0.0, used for 'ab_chart' method only.
        
    Returns
    -------
    np.ndarray
        Array of AD bounds for QFI with up to `nmax` channels (starting
        from n=1).
    
    Bibliography
    --------
    .. [1] Kurdziałek, S., Górecki, W., Albarelli, F., &
           Demkowicz-Dobrzański, R. (2023). Using adaptiveness and causal
           superpositions against noise in quantum metrology. Physical
           Review Letters, 131(9), 090801.
    .. [1] Kurdzialek, S., Albarelli, F., & Demkowicz-Dobrzanski, R.
           (2024). Universal bounds for quantum metrology in the presence
           of correlated noise. arXiv:2410.01881.
    """
    #minimal n for which warning is displayed for 'default' method
    AB_CHART_WARNING_THRESHOLD = 500

    if not channel.single_tooth:
        warn(
                "A non-trivial comb (more than one tooth) was provided."
                "In this function, causal structure is ignored and all"
                "inputs/outputs are merged into single input/output.",
                UserWarning
            )
    
    if not channel.trivial_env:
        raise ValueError(
            "This function doesn't work for correlated channels"
            "The input channel must have env_dim equal to 1."
            "Use ad_bounds_correlated for correlated channels"
        )

    if method == 'default':
        if nmax > AB_CHART_WARNING_THRESHOLD:
            warn(
                f"The 'default' method may be slow for nmax={nmax}."
                "The 'ab_chart' may be significantly faster.",
                UserWarning
            )

        # qfi for n=0
        current_bound = 0
        
        bounds = []
        for _ in range(nmax):
            current_bound = mop_adaptive_qfi(
                channel, 1, input_pure_qfi = current_bound
            )
            bounds.append(current_bound)

        return np.array(bounds)

    if method == 'ab_chart':
        krauses, dkrauses = channel.dkrauses()
        b, a = beta_alpha_chart(krauses, dkrauses, p, eps)

        # Computing AD bounds
        bounds = []  # List to store AD bounds at each step
        current_bound = 0  # Initial AD bound (for n=0)
        for _ in range(nmax):
            current_bound = np.min(current_bound + a +
                                    2 * b * np.sqrt(current_bound))
            bounds.append(current_bound)

        # Return the scaled results as per QFI bounds
        return 4*np.array(bounds)

    raise ValueError(
        f"Invalid `method` argument {method}.\n"
        "Possible methods: 'standard', 'ab_chart'."
    )


def cs_bounds(channel: ParamChannel,
    nmax: int,  p: int = 40, eps: float = 0.0
    ) -> np.ndarray:
    """
    Calculate causal superpositions (CS) bounds for the Quantum Fisher
    Information (QFI).
    
    This function computes the set of CS bounds for the QFIs of up to
    `nmax` channels probed using arbitrary causal superposition scheme
    involving arbitrary controls between channels. The alpha vs beta chart
    is constructed, and then the bound is computed
    
    Parameters
    ----------
    channel : ParamChannel
        Object representing a single channel and its derivative
    nmax : int
        Maximum number of channels for which the bounds are computed.
    p : int, optional
        Number of samples of beta for precision in bound calculation, by
        default 40, used for 'ab_chart' method only.
    eps : float, optional
        Small epsilon added to beta bounds in sampling to avoid numerical
        issues, by default 0.0, used for 'ab_chart' method only.
        
    Returns
    -------
    np.ndarray
        Array of CS bounds for QFI with up to `nmax` channels (starting
        from n=1).
    
    Bibliography
    --------
    .. [1] Kurdziałek, S., Górecki, W., Albarelli, F., &
           Demkowicz-Dobrzański, R. (2023). Using adaptiveness and causal
           superpositions against noise in quantum metrology. Physical
           Review Letters, 131(9), 090801.
        
    """
    krauses, dkrauses = channel.dkrauses()
        
    # Generate lists of minimal alpha and beta values
    b, a = beta_alpha_chart(krauses, dkrauses, p, eps)

    # Initialize CS bounds for different channel representations
    cs = np.zeros(p)
    cs_bounds_list = []
    for _ in range(nmax):
        cs = cs + a + 2 * b * np.sqrt(cs)
        # Take minimum over single channel representations
        cs_bounds_list.append(np.min(cs))

    return 4 * np.array(cs_bounds_list)


def par_ad_cs_bounds(channel: ParamChannel,
    nmax: int, p: int = 40, eps: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate parallel (PAR), adaptive (AD) and  causal superpositions (CS)
    bounds for the Quantum Fisher Information (QFI) using alpha vs beta
    chart method.
    
    This function computes three sets of bounds, PAR, AD and CS by 
    constructing list of minimal alpha for given beta constraints (see
    `beta_alpha_chart`). It is equivalent to calling par_bounds, ad_bounds
    and cs_bounds functions with method = 'ab_chart'.
    Calling this function is faster, since ab_chart is created only once.
    
    Parameters
    ----------
    channel : ParamChannel
        Object representing a single channel and its derivative
    nmax : int
        Maximum number of channels for which the bounds are computed.
    p : int, optional
        Number of samples of beta for precision in bound calculation, by
        default 40.
    eps : float, optional
        Small epsilon added to beta bounds in sampling to avoid numerical
        issues, by default 0.0.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
            - PAR_bounds_list : np.ndarray
                Array of AD bounds for QFI with up to `nmax` channels
                (starting from n=1).
            - AD_bounds_list : np.ndarray
                Array of AD bounds for QFI with up to `nmax` channels
                (starting from n=1).
            - CS_bounds_list : np.ndarray
                Array of CS bounds for QFI with up to `nmax` channels
                (starting from n=1).
    Bibliography
    --------
    .. [1] Kurdziałek, S., Górecki, W., Albarelli, F., &
        Demkowicz-Dobrzański, R. (2023). Using adaptiveness and causal
        superpositions against noise in quantum metrology. Physical Review
        Letters, 131(9), 090801.
    
    """
    raise NotImplementedError
    krauses, dkrauses = channel.dkrauses()

    # Generate lists of minimal alpha and beta values
    b, a = beta_alpha_chart(krauses, dkrauses, p, eps)

    #computing PAR_bounds
    par_bounds_list = [
        np.min(n*a + n*(n-1) * b**2) for n in range(1, nmax+1)
    ]

    # Computing AD bounds
    ad_bounds_list = []  # List to store AD bounds at each step
    ad_bound = 0  # Initial AD bound (for n=0)
    for _ in range(nmax):
        ad_bound = np.min(ad_bound + a + 2 * b * np.sqrt(ad_bound))
        ad_bounds_list.append(ad_bound)

    # Computing CS bounds
    # Initialize CS bounds for different channel representations
    cs = np.zeros(p)
    cs_bounds_list = []
    for _ in range(nmax):
        cs = cs + a + 2 * b * np.sqrt(cs)
        # Take minimum over single channel representations
        cs_bounds_list.append(np.min(cs))


def asym_scaling_qfi(channel: ParamChannel, power: int | None = None
    ) -> tuple[float, int]:
    """
    Calculate the scaling of QFI (with constant) when asymptotically many
    copies of an input channel are probed paralelly (PAR) using optimal
    input (possibly with ancilla) [1]_ .
    
    The result is also QFI scaling for optimal adaptive (AD) or causal
    superpostions (CS) strategy, see [2]_.
    
    Parameters
    ----------
    channel : ParamChannel
        Object representing a single channel and its derivative
    power: int | None optional
        Power in the QFI scaling law. It can be 1 (standard scaling) or
        2 (Heisenberg scaling). When not provided, the scaling type is 
        determined by the algorithm.
        
    Returns
    -------
    coef: float
        Coefficient in the QFI scaling law
    power: int
        Power in the QFI scaling law. Can be 1 or 2.
                
    Notes
    -----
    The QFI for asymptotically large number of channels `n` scales as 
    `coef` * `n` ^ `power`
    
    Bibliography
    .. [1] Demkowicz-Dobrzański, R., Kołodyński, J. & Guţă, M. 
           The elusive Heisenberg limit in quantum-enhanced metrology. 
           Nat Commun 3, 1063 (2012). https://doi.org/10.1038/ncomms2067
    .. [2] Kurdziałek, S., Górecki, W., Albarelli, F., &
           Demkowicz-Dobrzański, R. (2023). Using adaptiveness and causal
           superpositions against noise in quantum metrology.
           Physical Review Letters, 131(9), 090801.
    """

    no_scaling = power is None
    wrong_scaling = not (power is None or power == 1 or power == 2)

    if wrong_scaling:
        warn(
            f"Invalid `power` argument {power}."
            "Possible values: 1 (standard scaling), 2 (heisenberg scaling)."
            "The scaling type will be determined by the algorithm.",
            UserWarning
        )
    #channel Krauses and their derivatives
    K, dK = channel.dkrauses()

    if wrong_scaling or no_scaling:
        #first try to solve for standard scaling
        try:
            coef_standard = 4 * minimize_alpha_given_beta(K, dK, 0)
        except cp.error.SolverError:
            warn(
                "The solver failed while trying to determine the scaling"
                " type. Probably it is Heisenberg scaling with very low"
                " coefficient. The resulting value may be inaccuarate.",
                UserWarning     
            )
            return 4 * minimize_beta(K, dK)**2, 2

        if coef_standard == np.inf:
            #if the standard scaling is inf, then it is Heisenberg scaling
            return 4 * minimize_beta(K, dK)**2, 2
        return coef_standard, 1

    if power == 1:
        return 4 * minimize_alpha_given_beta(K, dK, 0), 1
    if power == 2:
        return 4 * minimize_beta(K, dK)**2, 2

    raise ValueError(
        f'Power parameter must be either None, 2 or 4 but {power} was '\
        'given.'
    )


def ad_bounds_correlated(channel: ParamChannel, nmax: int,
    block_size: int, for_every_n: bool = False,
    close_last_env: bool = False, env_inp_state: np.ndarray | None = None,
    print_messages: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate QFI bounds for adaptive strategies for general correlated
    models.
    
    This function computes the upper bounds for QFI when n copies of
    the given channel are linked using their environments. It is assumed
    that information leaks from environment aftery every `block_size` copies.
    
    Parameters
    ----------
    channel : ParamChannel
        Object representing an elementary channel or comb and its derivative.
    nmax : int
        Maximal number of elementary channels for which the bound is computed.
    block_size: int
        The number of channels merged in one sub-chain during the algorithm.
        The larger it is, the tighter the bounds, but also computations are 
        slower and more memory-consuming. It is denoted by `m` in [1]_.
    for_every_n: bool, optional
        When False (default), then the bound is calculated for number of copies
        which are multiples of `block_size` only. When True, bounds are computed
        for all number of copies up to `nmax`. In the latter case, the algorithm
        is slower, but bounds may be slightly tighter.
    close_last_env: bool, optional
        When True, then it is assumed that in the last channel in the chain,
        no information leaks from environment. This makes the bounds slightly tighter,
        but slower the algorithm, since the QFI with environment leakage must be computed
        anyway for the sake of next iteration step.
    env_inp_state: np.ndarray|None, optional
        When specified, this state is contracted with first environment input
        of the chain. Otherwise, control can arbitrarily act on first input environment.
    print_messages: bool, optional
        When True, then calculated bounds are printed in real time.
    
        
    Returns  
    -------  
    n_list : np.ndarray  
        Array of `n` values (number of elementary channel copies)  
        for which the bounds were computed.  
    bounds : np.ndarray  
        Array of corresponding upper bounds for QFI.  

    Bibliography
    --------
    .. [1] Kurdzialek, S. , Albarelli, F. & Demkowicz-Dobrzanski , R.
        "Universal bounds for quantum metrology in the presence of
        correlated noise", arxiv: 2410.01881
    """
    if channel.env_inp_dim != channel.env_out_dim:
        raise ValueError(
            "Channel must have the same input and output environment"
             " dimensions. Otherwise environments cannot be contracted."
        )
    
    if not for_every_n:
        n_list = []
        bounds_open_env = []
        
        if close_last_env:
            bounds_close_env = []
            
        #initial algorithm parameters
        n = block_size
        first_step = True
        open_bound = 0

        while n <= nmax:
            n_list.append(n)
            
            #bound with close (uncontrolled) last environment
            if close_last_env:
                if first_step and env_inp_state is not None:
                    close_bound = mop_adaptive_qfi(
                        channel, block_size, env_control = (False, False), 
                        env_inp_state = env_inp_state
                    )                    
                else:
                    close_bound = mop_adaptive_qfi(
                        channel, block_size, env_control = (True, False), 
                        input_pure_qfi = open_bound
                    )
                    
                bounds_close_env.append(close_bound)
                
            #bound with open (controlled) last environment
            if n + block_size <= nmax or not close_last_env:
                if first_step and env_inp_state is not None:
                    open_bound = mop_adaptive_qfi(
                        channel, block_size, env_control = (False, True), 
                        env_inp_state = env_inp_state
                    )                    
                else:
                    open_bound = mop_adaptive_qfi(
                        channel, block_size, env_control = (True, True), 
                        input_pure_qfi = open_bound
                    )
                
                bounds_open_env.append(open_bound)
                
            #printing a message
            if print_messages:
                print(n, open_bound if not close_last_env else close_bound)
            
            #updating algorithm parameters
            n += block_size
            first_step = False

    if for_every_n:
        n_list = list(range(1, nmax+1))
        bounds_open_env = np.inf * np.ones(nmax)
        
        if close_last_env:
            bounds_close_env = np.inf * np.ones(nmax)
        
        #initializing first `block_size` elements
        for i in range(min(block_size, nmax)):
            if close_last_env:
                if env_inp_state is not None:
                    close_bound = mop_adaptive_qfi(
                        channel, i+1, env_control=(False, False),
                        env_inp_state=env_inp_state
                    )
                else:
                    close_bound = mop_adaptive_qfi(
                        channel, i+1, env_control=(True, False)
                    )
                    
                bounds_close_env[i] = close_bound
                
            if i < nmax - 1 or not close_last_env:
                if env_inp_state is not None:
                    open_bound = mop_adaptive_qfi(
                        channel, i+1, env_control = (False, True),
                        env_inp_state = env_inp_state
                    )
                else:
                    open_bound = mop_adaptive_qfi(
                        channel, i+1, env_control = (True, True)
                    )
                    
                bounds_open_env[i] = open_bound
                
            #printing a message
            if print_messages:
                if close_last_env:
                    print(i+1, bounds_close_env[i])
                else:
                    print(i+1, bounds_open_env[i])
                
        #calculating the remaining elements by minimizing all possibilities
        for i in range(block_size, nmax):
            for j in range(1, block_size+1):
                input_qfi = bounds_open_env[i-j]
                
                if close_last_env:
                    close_bound = mop_adaptive_qfi(
                        channel, j, env_control = (True, False),
                        input_pure_qfi = input_qfi
                    )
                    bounds_close_env[i] = min(bounds_close_env[i],
                                              close_bound)
                if i < nmax - 1 or not close_last_env:
                    open_bound = mop_adaptive_qfi(
                        channel, j, env_control = (True, True),
                        input_pure_qfi = input_qfi
                    )
                    bounds_open_env[i] = min(bounds_open_env[i],
                                              open_bound)
                    
            #printing a message
            if print_messages:
                if close_last_env:
                    print(i+1, bounds_close_env[i])
                else:
                    print(i+1, bounds_open_env[i])
        
    if close_last_env:
        return np.array(n_list), np.array(bounds_close_env)
    else:
        return np.array(n_list), np.array(bounds_open_env)


def minimize_beta_correlated(krauses_comb: list[np.ndarray], 
    dkrauses_comb: list[np.ndarray], dims: tuple[int, ...]) -> float:
    """
    Computes the minimal 'comb-norm' of a correlated beta matrix.
    
    The returned minimal b value has significant physical implications:
    - When `b=0`, Heisenberg scaling (HS) is not possible.
    - When `b>0`, asymptotic upper bound for Quantum Fisher 
      Information (QFI) as `4 b^2 (N/m)^2`, `m` is the number of channels
      in comb, `N` is the total number of channels.
    
    Parameters
    ----------
    krauses_comb : list[np.ndarray]
        List of Kraus operators for the comb. Each operator is a linear
        map from `H_2m-1 x H_2m-3 x ... x H1` to `H_2m x H_2m-2 x ... x
        H2`.
    dkrauses_comb : list[np.ndarray]
        List of derivatives of Kraus operators for the comb.
    dims : tuple[int, ...]
        Dimensions of the spaces `[H1, H2, ..., H_2m]`.
    
    Returns
    -------
    float
        The minimal value of b, representing the optimization result.
        
    Notes
    ------
    - First input (`H_1`) and last output (`H_2m`) should also contain
      register spaces `R` containing information about correlations.
    - This is an implementation of algorithm from appendix C2 from [1]_.
    
    Bibliography
    --------
    .. [1] Kurdzialek, S. , Albarelli, F. & Demkowicz-Dobrzanski , R.
        "Universal bounds for quantum metrology in the presence of
        correlated noise", arxiv: 2410.01881
    """
    # Extend dimensions with H0 and artificial H-1 space
    dims_ext = np.concatenate([[1, 2], dims])
    dims_ext[-1] = 1  # The last output is traced out

    num_kraus = len(krauses_comb)
    total_dim = np.prod(dims)

    # SDP formulation: construct comb Q (see [1]) variables and
    # constraints
    combs_Q, constraints, trace_var = comb_variables(
        dims_ext, hermitian=True, trace_constraint=None
    )

    # Decomposition vectors of Choi operator
    C_vectors = np.array( [K.flatten() for K in krauses_comb] )
    dC_vectors = np.array( [dK.flatten() for dK in dkrauses_comb])

    # Matrix h generating different Kraus representations
    h = cp.Variable((num_kraus, num_kraus), hermitian=True)

    # Construct beta matrix

    # Create matrix containing `c` vectors (last dimension of `C` traced
    # out) horizontal `c` vectors are stacked vertically
    clist = np.reshape(
        C_vectors, (num_kraus * dims[-1], total_dim // dims[-1])
    )

    #the same for derivatives
    dclist = np.reshape(
        dC_vectors, (num_kraus * dims[-1], total_dim // dims[-1])
    )

    #derivatives transformed by matrix `h`
    tclist = dclist - 1j * cp.kron(h, np.eye(dims[-1])) @ clist

    #constructing correlated beta matrix
    beta = hc(tclist) @ clist

    # Build matrix B: beta on antidiagonals, 0 on diagonals
    B00 = np.zeros((total_dim // dims[-1], total_dim // dims[-1]))
    B11 = B00
    B01 = 0.5 * beta
    B10 = 0.5 * hc(beta)

    B = cp.bmat([[B00, B01], [B10, B11]])

    # combs_Q[-1] acts on H_2M-2 x H_2M-4 x ... x H_2 x V_0 x H_2M-1 x ...
    # ... x H_1 x null
    # B acts on V_0 x H_2M-2 x ... x H_2 x H_2M-1 x ... H_1 x null
    # to compare them, we need to swap subspaces

    # Dimension of H_2 x H_4 x ... H_2M-2
    even_dim = int(np.prod(dims[1:-1:2]))
    # Dimension of H0
    V0_dim = 2
    # Dimension of H1 x H3 x ... x H_2M-1
    odd_dim = int(np.prod(dims[::2]))

    swap_dims = (even_dim, V0_dim, odd_dim)
    swap_op = swap_operator(swap_dims, 0, 1)
    comb_Q_swapped = swap_op @ combs_Q[-1] @ swap_op.T

    # Add trace inequality constraint
    constraints.append(comb_Q_swapped >> B)

    # Define objective function
    obj = cp.Minimize(trace_var)

    # Solve the problem
    prob = cp.Problem(obj, constraints)
    sol = prob.solve()

    return sol


def minimize_alpha_beta_0_correlated(krauses_comb: list[np.ndarray],
    dkrauses_comb: list[np.ndarray], dims: tuple[int, ...]) -> float:
    """
    Computes the minimal 'comb-norm' of a correlated alpha matrix assuming
    beta_1=0.
    
    When the returned minimal `a` value is finite, then
    - Heisenberg scaling (HS) is not possible.
    - Asymptotic upper bound for Quantum Fisher Information (QFI)
      is `4 a N/m`, `m` is the number of channels in comb,
      `N` is the total number of channels.
    
    Parameters
    ----------
    krauses_comb : list[np.ndarray]
        List of Kraus operators for the comb. Each operator is a linear
        map from `H_2m-1 x H_2m-3 x ... x H1` to `H_2m x H_2m-2 x ... x
        H2`.
    dkrauses_comb : list[np.ndarray]
        List of derivatives of Kraus operators for the comb.
    dims : tuple[int, ...]
        Dimensions of the spaces `[H1, H2, ..., H_2m]`.
    
    Returns
    -------
    float
        The minimal value of a, representing the optimization result.
        
    Notes
    ------
    - First input (`H_1`) and last output (`H_2m`) should also contain
      register spaces `R` containing information about correlations.
    - This is an implementation of algorithm from appendix C2 from [1]_.
    
    Bibliography
    --------
    .. [1] Kurdzialek, S. , Albarelli, F. & Demkowicz-Dobrzanski , R.
        "Universal bounds for quantum metrology in the presence of
        correlated noise", arxiv: 2410.01881
    """
    #Number of Kraus operators and total dimension
    num_kraus = len(krauses_comb)
    total_dim = np.prod(dims)

    # Decomposition vectors of Choi operator
    C_vectors = np.array( [K.flatten() for K in krauses_comb] )
    dC_vectors = np.array( [dK.flatten() for dK in dkrauses_comb])

    # Matrix h generating different Kraus representations
    h = cp.Variable((num_kraus, num_kraus), hermitian=True)

    # dimensions of combs Q and Y (including identities)
    dims_traced_output = list(dims).copy()
    dims_traced_output[-1] = 1

    # Comb Q variables and constraints (see [1], appendix C2)
    combs_Q, constraints_Q, trace_var_Q = comb_variables(
        tuple(dims_traced_output), hermitian=True, trace_constraint=None
    )

    # Comb Y variables and constraints (see [1], appendix C2)
    combs_Y, constraints_Y, _ = comb_variables(
        tuple(dims_traced_output), hermitian=False, trace_constraint=None
    )

    # Construct beta matrix

    # Create matrix containing `c` vectors (last dimension of `C` traced out)
    # horizontal `c` vectors are stacked vertically 
    clist = np.reshape(
        C_vectors, (num_kraus * dims[-1], total_dim // dims[-1])
    )

    #the same for derivatives
    dclist = np.reshape(
        dC_vectors, (num_kraus * dims[-1], total_dim // dims[-1])
    )

    #derivatives transformed by matrix `h`
    tclist = dclist - 1j * cp.kron(h, np.eye(dims[-1])) @ clist

    #constructing correlated beta matrix
    beta = hc(tclist) @ clist

    #construct A matrix
    A00 = combs_Q[-1]
    A10 = tclist
    A01 = hc(A10)
    A11 = np.eye(num_kraus * dims[-1])

    A = cp.bmat([[A00, A01], [A10, A11]])

    #collecting all the comb constraints together
    constraints = constraints_Q + constraints_Y

    #positivity constraints for A
    constraints.append(A>>0)

    #constaint for beta (equivalent to beta_1 == 0)
    constraints.append(beta == combs_Y[-1])

    #defining objective and solving the problem
    obj = cp.Minimize(trace_var_Q)
    prob = cp.Problem(obj, constraints)
    sol = prob.solve()

    return sol


def ad_asym_bound_correlated(channel: ParamChannel, block_size: int,
    power: int | None = None) -> tuple[float, int]:
    """
    Calculate the asymptotic bound for QFI for adaptive strategies
    for correlated models.
    
    This function returns an upper bound for scaling coefficient
    and the scaling power (1 for standard scaling, 2 for Heisenberg).
    Unlike for uncorrelated models, this is just upper bound, not the
    exact value. The bound becomes tighter for larger `block_size`. 
    
    Parameters
    ----------
    channel : ParamChannel
        Object representing an elementary channel or comb and its
        derivative.
    block_size: int
        The number of channels merged in one sub-chain during the
        algorithm. The larger it is, the tighter the bounds, but also
        computations are slower and more memory-consuming. It is denoted
        by `m` in [1]_.
    power: int | None optional
        Power in the QFI scaling law. It can be 1 (standard scaling) or
        2 (Heisenberg scaling). When not provided, the scaling type is 
        determined by the algorithm.
        
    Returns
    -------
    coef: float  
        Coefficient in the QFI asymptotic bound.  
    power: int  
        Power in the QFI asymptotic bound. Can be 1 or 2.
                
    Notes
    -----
    The QFI for asymptotically large number of channels `n` is upper
    bounded by `coef` * `n` ^ `power`
        
    Bibliography
    --------
    .. [1] Kurdzialek, S. , Albarelli, F. & Demkowicz-Dobrzanski , R.
        "Universal bounds for quantum metrology in the presence of
        correlated noise", arxiv: 2410.01881
    """
    if channel.env_inp_dim != channel.env_out_dim:
        raise ValueError(
            "Channel must have the same input and output environment"
             " dimensions. Otherwise environments cannot be contracted."
        )
    env_dim = channel.env_inp_dim

    #Creating the full comb for which the QFI will be computed
    comb_channel = channel.markov_series(block_size)

    #number of teeth of created comb 
    teeth_number = len(comb_channel.input_spaces) 

    #names of all input and output spaces according to causal order
    #these lists include environment 
    input_spaces = [comb_channel.env_inp] + comb_channel.input_spaces
    output_spaces = comb_channel.output_spaces + [comb_channel.env_out]

    # creating list of dimension of all spaces according to causal order
    dims = []
    for i in range(teeth_number):
        dims.append(comb_channel.input_dims[i])
        dims.append(comb_channel.output_dims[i])

    # environment is merged with first input/ last output space
    dims[0] *= env_dim
    dims[-1] *= env_dim
    dims = tuple(dims)

    #Kraus operators are created with reverse order of inp and out spaces
    comb_tensor = comb_channel.tensor()
    krauses_comb, dkrauses_comb = comb_tensor.dkrauses(
        input_spaces[::-1], output_spaces[::-1]
    )

    no_scaling = power is None
    wrong_scaling = power not in (None, 1, 2)

    if wrong_scaling:
        warn(
            f"Invalid `power` argument {power}."
            "Possible values: 1(standard scaling), 2 (heisenberg scaling)."
            " The scaling type will be determined by the algorithm.",
            UserWarning
        )

    if wrong_scaling or no_scaling:
        #first try to solve for standard scaling
        try:
            min_alpha = minimize_alpha_beta_0_correlated(
                krauses_comb, dkrauses_comb, dims
            )
        except cp.SolverError:
            warn(
                "The solver failed while trying to determine the scaling"
                " type. Probably it is Heisenberg scaling with very low"
                " coefficient. The resulting value may be inaccuarate.",
                UserWarning
            )
            min_beta = minimize_beta_correlated(
                krauses_comb, dkrauses_comb, dims
            )
            return 4 * min_beta**2 / block_size**2, 2

        if min_alpha == np.inf:
            #if the standard scaling is inf, then it is Heisenberg scaling
            min_beta = minimize_beta_correlated(
                krauses_comb, dkrauses_comb, dims
            )
            return 4 * min_beta**2 / block_size**2, 2
        else:
            return 4 * min_alpha / block_size, 1

    if power == 1:
        min_alpha = minimize_alpha_beta_0_correlated(
                krauses_comb, dkrauses_comb, dims
            )
        return 4 * min_alpha / block_size, 1

    # if power == 2:
    min_beta = minimize_beta_correlated(
            krauses_comb, dkrauses_comb, dims
    )
    return 4 * min_beta**2 / block_size**2, 2
