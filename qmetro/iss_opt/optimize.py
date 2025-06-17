from collections.abc import Hashable
from itertools import repeat, product
from math import prod, sqrt
from warnings import warn

import cvxpy as cp
import numpy as np
from scipy.linalg import eig, eigh

from ncon import ncon

from ..qmtensor import ConstTensor, TensorNetwork
from ..qmtensor.operations import is_comb_var
from ..qtools import comb_variables, hc, ket_bra
from ..utils import enhance_hermiticity, snd

from .errors import NormMatZeroEigenval, SolverError
from .iss_config import IssConfig
from .iss_warnings import non_herimitan_message, parameter_norm_message




def optimize_var(tn: TensorNetwork, _tn: TensorNetwork,
    name: str, m0: ConstTensor, m1: ConstTensor, m: ConstTensor,
    cptp_vs: dict[str, ConstTensor], slds: dict[str, ConstTensor],
    mps_vs: dict[str, ConstTensor], mps_left: ConstTensor,
    mps_rights: dict[str, ConstTensor], _: IssConfig
    ) -> tuple[float, ConstTensor]:

    templ_t = tn.tensors[name]
    old_t = _tn.tensors[name]

    output_spaces = old_t.output_spaces
    bond_spaces = old_t.bond_spaces
    comb_str = old_t.comb_structure
    is_unital = templ_t.is_unital
    sing_val_cutoff = 1e-3 / len(slds)**2
    eps = 1e-7

    try:
        if name in cptp_vs:
            if is_comb_var(templ_t):
                qfi, x = _comb_optimize(m, comb_str)
            else:
                qfi, x = _cptp_optimize(m, output_spaces, is_unital)
        elif name in slds:
            qfi, x = _sld_optimize(m0, m1, bond_spaces, sing_val_cutoff)
        elif name in mps_vs:
            norm_t = mps_left * mps_rights[name]
            qfi, x = _mps_optimize(m, norm_t, eps)
        else:
            raise ValueError(f"Name {name} not recognized.")
    except cp.SolverError as e:
        raise SolverError(name, e, m0, m1) from e
    
    if qfi < -1e-5:
        msg = f'Obtained QFI ({qfi}) is negative.'
        raise SolverError(name, msg, m0, m1)

    x.name = name
    x.output_spaces = output_spaces
    x.comb_structure = old_t.comb_structure

    return qfi, x


def _cptp_optimize(m: ConstTensor, output_spaces: list[Hashable],
    unital: bool = False) -> tuple[float, ConstTensor]:
    _sd = m.sdict
    out_set = set(output_spaces)
    input_spaces = [space for space in m.spaces if space not in out_set]
    out_dim = prod(_sd[space] for space in output_spaces)
    inp_dim = prod(_sd[space] for space in input_spaces)

    m_arr = m.choi([*output_spaces, *input_spaces])
    try:
        m_par = cp.Parameter(m_arr.shape, value=m_arr.T, hermitian=True)
    except ValueError as e:
        if 'Parameter value must be hermitian.' in str(e):
            m_arr, delta = enhance_hermiticity(m_arr)
            m_par = cp.Parameter(
                m_arr.shape, value=m_arr.T, hermitian=True
            )
            warn(non_herimitan_message(delta))
        else:
            raise e

    v = cp.Variable(m_arr.shape, hermitian=True)
    constrs = [
        v >> 0,
        cp.partial_trace(v, (out_dim, inp_dim), 0) == np.identity(inp_dim)
    ]
    if unital:
        constrs.append(
            cp.partial_trace(v, (out_dim, inp_dim), 1)
            == np.identity(out_dim)
        )

    prob = cp.Problem(cp.Maximize(cp.real(cp.trace(m_par @ v))), constrs)

    ns = range(0, NORM_TRIES_MAX + 1, NORM_TRIES_STEP)
    for n in ns:
        norm_factor = 2**n
        m_par.value = m_arr.T / norm_factor
        try:
            qfi = prob.solve() * norm_factor
            break
        except cp.SolverError as e:
            if n < NORM_TRIES_MAX:
                warn(parameter_norm_message(n))
            else:
                raise e

    return qfi, ConstTensor(
        output_spaces + input_spaces, v.value, _sd,
        output_spaces=output_spaces
    )


def _sld_optimize(m0: ConstTensor, m1: ConstTensor,
    bond_spaces: list[Hashable], sing_val_cutoff: float
    ) -> tuple[float, ConstTensor]:
    if not bond_spaces:
        return _sld_optimize_sdp(m0, m1)

    return _sld_optimize_pseudoinverse(
        m0, m1, bond_spaces, sing_val_cutoff
    )


def _sld_optimize_sdp(m0: ConstTensor, m1: ConstTensor
    ) -> tuple[float, ConstTensor]:
    d = m0.dimension
    L = cp.Variable((d, d), hermitian=True)
    L2 = cp.Variable((d, d), hermitian=True)
    A = cp.bmat([[L2, L], [L, np.eye(d)]])
    constrs = [A >> 0] # L2 >> L^2

    spaces = m0.spaces
    m0_arr = m0.choi(spaces)
    m1_arr = m1.choi(spaces)

    try:
        m0_par = cp.Parameter(m0_arr.shape, value=m0_arr, hermitian=True)
    except ValueError as e:
        if 'Parameter value must be hermitian.' in str(e):
            m0_arr, delta = enhance_hermiticity(m0_arr)
            m0_par = cp.Parameter(
                m0_arr.shape, value=m0_arr, hermitian=True
            )
            warn(non_herimitan_message(delta))
        else:
            raise e

    try:
        m1_par = cp.Parameter(m1_arr.shape, value=m1_arr, hermitian=True)
    except ValueError as e:
        if 'Parameter value must be hermitian.' in str(e):
            m1_arr, delta = enhance_hermiticity(m1_arr)
            m1_par = cp.Parameter(
                m1_arr.shape, value=m1_arr, hermitian=True
            )
            warn(non_herimitan_message(delta))
        else:
            raise e

    obj = cp.Maximize(cp.real(cp.trace(2 * m0_par @ L - m1_par @ L2)))
    prob = cp.Problem(obj, constrs)

    ns = range(0, NORM_TRIES_MAX + 1, NORM_TRIES_STEP)
    for n in ns:
        norm_factor = 2**n
        m0_par.value = m0_arr / norm_factor
        m1_par.value = m1_arr / norm_factor
        try:
            qfi = prob.solve() * norm_factor
            break
        except cp.SolverError as e:
            if n < NORM_TRIES_MAX:
                warn(parameter_norm_message(n))
            else:
                raise e

    return qfi, ConstTensor(spaces, L.value.T, m0.sdict)


def _sld_optimize_pseudoinverse(m0: ConstTensor, m1: ConstTensor,
    bond_spaces: list[Hashable], sing_val_cutoff: float
    ) -> tuple[float, ConstTensor]:
    """
    Algorithm from K. Chabuda et al. "Tensor-Network Approach for Quantum
    Metrology in Many-Body Quantum Systems" section IIIC.
    """
    bond_spaces_set = set(bond_spaces)
    physical_spaces = [s for s in m0.spaces if s not in bond_spaces_set]

    b_vec = _get_b_vector(m0, physical_spaces, bond_spaces)
    a_mat = _get_a_matrix(m1, physical_spaces, bond_spaces)
    a_tylde = (a_mat + a_mat.T) / 2

    solution: np.ndarray = np.linalg.pinv(a_tylde, sing_val_cutoff) @ b_vec

    _sd = m0.sdict
    solution_tensor = solution.reshape(
        [_sd[s]**2 for s in physical_spaces]
        + [_sd[s] for s in bond_spaces]
    )

    # Inner product below is a substitute for tensor contraction thus
    # there should be no complex conjugation for the left vector.
    qfi = 2 * b_vec @ solution - solution @ (a_mat @ solution)
    qfi = np.real(qfi)

    spaces = physical_spaces + bond_spaces
    LT = ConstTensor(spaces, array=solution_tensor, sdict=_sd)
    arr_T = LT.choi_T(*LT.physical_spaces).array
    LT.array = (LT.array + arr_T.conjugate()) / 2

    return qfi, LT


def _sld_optimize_linear(m0: ConstTensor, m1: ConstTensor,
    sing_val_cutoff: float) -> tuple[float, ConstTensor]:
    spaces = m0.spaces
    m0_choi = m0.choi(spaces)
    m1_choi = m1.choi(spaces)
    eigvals, eigvecs = eigh(m1_choi)

    sld = np.zeros_like(m1_choi)
    for (i, ei), (j, ej) in product(enumerate(eigvals), repeat=2):
        vi: np.ndarray = eigvecs[:, i]
        vj: np.ndarray = eigvecs[:, j]
        if ei + ej > sing_val_cutoff:
            sld_ij = 2 * vi.conjugate() @ (m0_choi @ vj) / (ei + ej)
            sld += sld_ij * ket_bra(vi, vj)
    
    qfi = np.real(np.trace(2 * m1_choi @ sld - m0_choi @ sld @ sld))

    return qfi, ConstTensor(spaces, choi=sld.T, sdict=m0.sdict)


def _get_b_vector(m: ConstTensor, physical_spaces: list[Hashable],
    bond_spaces: list[Hashable]) -> np.ndarray:
    tensor0: np.ndarray = np.copy(
        m.reorder(physical_spaces + bond_spaces).array
    )
    return np.ravel(tensor0)


def _get_a_matrix(m: ConstTensor, physical_spaces: list[Hashable],
    bond_spaces: list[Hashable]) -> np.ndarray:
    _sd = m.sdict

    _bond_spaces = [_sd.primed(s) for s in bond_spaces]
    physical_dim = prod(_sd[s] for s in physical_spaces)
    len_phys = len(physical_spaces)
    len_bond = len(bond_spaces)
    len_all = len_phys + len_bond

    array1_spaces = physical_spaces + bond_spaces + _bond_spaces
    array1: np.ndarray = np.copy(m.reorder(array1_spaces).array)

    array1_new_phys_dims: list[int] = []
    array1_new_bond_dims: list[int] = []
    non_con_i = -1
    new_array1_phys_order: list[int] = []
    mat_prod_term_order: list[int] = []

    for space in physical_spaces:
        array1_new_phys_dims += [_sd[space], _sd[space]]
        new_array1_phys_order += [
            non_con_i, non_con_i - 2*len_phys - len_bond - 1
        ]
        mat_prod_term_order += [non_con_i - 1]
        non_con_i -= 2

    new_array1_bond_order: list[int] = []
    for space in bond_spaces:
        array1_new_bond_dims += [_sd[space]]
        new_array1_bond_order += [non_con_i]
        non_con_i -= 1

    array1_new_dims = array1_new_phys_dims + 2 * array1_new_bond_dims
    array1 = array1.reshape(array1_new_dims)

    new_array1_bond_order += [
        i - (2*len_phys+len_bond) for i in new_array1_bond_order
    ]
    new_array1_order = new_array1_phys_order + new_array1_bond_order

    mat_prod_term_order += [
        i - (2*len_phys+len_bond-1) for i in mat_prod_term_order
    ]

    mat_prod_term = np.identity(physical_dim, dtype=np.complex128)
    mat_prod_term = mat_prod_term.reshape(
        2 * [_sd[s] for s in physical_spaces]
    )

    new_array1 = ncon(
        [array1, mat_prod_term],
        [new_array1_order, mat_prod_term_order]
    )

    single_bond_range = int(prod(_sd[s] for s in bond_spaces))
    return new_array1.reshape([
        single_bond_range * physical_dim**2,
        single_bond_range * physical_dim**2
    ])


def _mps_optimize(m: ConstTensor, norm_tensor: ConstTensor, eps: float
    ) -> tuple[float, ConstTensor]:
    _sd = m.sdict
    bond_spaces = m.bond_spaces
    physical_spaces = m.physical_spaces.copy()
    len_phys = len(m.physical_spaces)
    len_bond = len(m.bond_spaces)
    len_all = len_phys + len_bond

    f_mat = _get_f_mat(m) # It is F operator from [1] multiplied by -1

    norm_tensor.reorder(bond_spaces)
    proto_n = norm_tensor.array.copy()
    proto_n = proto_n.reshape([
        int(np.sqrt(d)) for s in bond_spaces for d in repeat(_sd[s], 2)
    ])

    inner_prod_term = np.identity(m.physical_dim, dtype=np.complex128)
    inner_prod_term = inner_prod_term.reshape(
        2 * [_sd[s] for s in physical_spaces]
    )

    inner_prod_indices = [-1 - i for i in range(len_phys)]
    inner_prod_indices += [-1 - len_all - i for i in range(len_phys)]

    proto_n_indices = [
        -1 - len_phys - ind
        for i in range(len_bond) for ind in (i, i + len_all)
    ]

    n_arr = ncon(
        [inner_prod_term, proto_n],
        [inner_prod_indices, proto_n_indices]
    )
    single_bond_range = int(sqrt(prod(_sd[s] for s in bond_spaces)))
    n_dim = m.physical_dim * single_bond_range
    n_mat = n_arr.reshape(n_dim, n_dim)

    VERSION = 0
    if VERSION == 0:
        n_eigs = eig(n_mat)[0]
        min_n_eig = np.min(np.abs(np.real(n_eigs)))
        if min_n_eig < eps:
            raise NormMatZeroEigenval(min_n_eig)

        eigs, eigvs = eig(f_mat, n_mat)
        i, _ = max(enumerate(np.real(eigs)), key=snd)
        solution_vec: np.ndarray = eigvs[:, i]
    elif VERSION == 1:
        # This version requires bigger eps (1e-7 vs 1e-5) so it might be
        # more prone to errors. On the other hand, it can proceed when
        # n_mat has eigenvalues smaller than eps while version 0 can't.
        # In practive they give similar results.
        n_eigs, U_hc = eig(n_mat)
        _n_pinv_sqrt_eigs = []
        for eigval in n_eigs:
            sqrt_eig = np.sqrt(eigval + 0j) # n_mat is PSD by construction.

            if abs(eigval) > 1e-5:
                _n_pinv_sqrt_eigs.append(1 / sqrt_eig)
            else:
                _n_pinv_sqrt_eigs.append(0.0)

        _n_pinv_sqrt = U_hc @ np.diag(_n_pinv_sqrt_eigs) @ hc(U_hc)

        eigs, eigvs = eig(_n_pinv_sqrt @ f_mat @ _n_pinv_sqrt)
        i, _ = max(enumerate(np.real(eigs)), key=snd)
        print(_)
        solution_vec: np.ndarray = _n_pinv_sqrt @ eigvs[:, i]

    norm = solution_vec.conjugate() @ (n_mat @ solution_vec)
    qfi = solution_vec.conjugate() @ (f_mat @ solution_vec) / norm
    solution_vec /= np.sqrt(norm)
    solution_arr = solution_vec.reshape(
        [_sd[s] for s in physical_spaces]
        + [int(sqrt(_sd[s])) for s in bond_spaces]
    )

    return np.real(qfi), ConstTensor.from_mps(
        solution_arr, physical_spaces + bond_spaces, _sd,
        physical_spaces
    )


def _get_f_mat(m: ConstTensor) -> np.ndarray:
    _sd = m.sdict
    bond_spaces = m.bond_spaces
    physical_spaces = m.physical_spaces.copy()
    spaces = m.physical_spaces + m.bond_spaces
    len_phys = len(m.physical_spaces)
    len_bond = len(m.bond_spaces)
    len_all = len_phys + len_bond

    m.reorder(spaces)
    proto_f_arr = m.array.reshape(
        [d for s in physical_spaces for d in repeat(_sd[s], 2)]
        + [d for s in bond_spaces for d in repeat(int(sqrt(_sd[s])), 2)]
    )
    tmp = [-1 - ind for i in range(len_all) for ind in (i, i + len_all)]
    f_arr = ncon(
        [proto_f_arr],
        [tmp]
    )
    f_dim = m.physical_dim * int(sqrt(prod(_sd[s] for s in bond_spaces)))
    return f_arr.reshape(f_dim, f_dim)


def _comb_optimize(m: ConstTensor,
    comb_structure: list[tuple[list[Hashable], list[Hashable]]]
    ) -> tuple[float, ConstTensor]:
    _sd = m.sdict
    dims = []
    for tooth_inp, tooth_out in comb_structure:
        inp_dims = [_sd[space] for space in tooth_inp]
        out_dims = [_sd[space] for space in tooth_out]

        inp_dim = prod(inp_dims)
        out_dim = prod(out_dims)

        dims.append(inp_dim)
        dims.append(out_dim)

    T_vars, constraints, _ = comb_variables(tuple(dims))
    T = T_vars[-1] # H_2i+2 (x) ... (x) H_2 (x) H_2i+1 (x) ... H_1
    constraints.append(T >> 0)

    even_spaces = []
    odd_spaces = []
    for tooth_inp, tooth_out in comb_structure[::-1]:
        even_spaces += tooth_out
        odd_spaces += tooth_inp
    spaces = even_spaces + odd_spaces
    m_matrix = m.choi(spaces).T

    objective = cp.Maximize(cp.real(cp.trace(m_matrix @ T)))
    problem = cp.Problem(objective, constraints)
    qfi = problem.solve()

    return qfi, ConstTensor(
        spaces, choi=T.value, sdict=_sd, output_spaces=even_spaces,
        comb_structure=comb_structure
    )


NORM_TRIES_STEP = 1
NORM_TRIES_MAX = 8
