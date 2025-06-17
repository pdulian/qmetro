from __future__ import annotations

import copy
import uuid

from collections import OrderedDict, namedtuple
from collections.abc import Hashable
from itertools import chain
from queue import Queue
from warnings import warn

import numpy as np

from ..consts import LINE_WIDTH
from ..qmtensor import TensorNetwork, ParamTensor, ConstTensor, contr
from ..qmtensor.operations import (
    is_var, is_measurement, is_cptp_var, is_mps, is_comb_var
)

from ..utils import fst, limited_print

from .artnoise import ArtNoise
from .errors import (
    MaxIterExceededError, SolverError, NonHermitianError, SingleIterError,
    NormMatZeroEigenval
)
from .iss_config import IssConfig
from .optimize import optimize_var




def iss_opt(tn: TensorNetwork, name: str | None = None,
    max_error_iterations: int = 10, max_iterations: int = 500,
    min_iterations: int = 10, eps: float = 1e-4,
    init_tn: TensorNetwork | None = None,
    print_messages: bool | str = False,
    var_iterations: int = 1, sld_iterations: int = 1,
    art_noise_spaces: list[list[Hashable]] | None = None,
    art_noise_params: tuple[float, float] = (0.5, 0.1),
    contraction_order: list[str] | None = None,
    adaptive_art_noise: bool = True) -> tuple[float, list[list[float]],
    TensorNetwork, bool]:
    """
    Computes quantum Fisher information (QFI) for strategy provided in
    a form of a tensor network using iterative see-saw algorithm [1]_,
    [2]_.

    Function takes as an arguemnt an object of QNetwork class then
    maximizes its QFI optimizing over nodes which are variable Choi
    matrices (members of VarChoiMatrix) or variable measurements (members
    of MeasureChoi) called also variable semi-logarithmic derivatives
    (SLD).

    To make the algorithm more stable it adds an additional artificial
    depolarizing noise [2]_ which decays exponentially with each
    iteration.
    

    Parameters
    ----------
    tn : QNetwork
        Network that is being optimized.
    name : str | None, optional
        Name of returned network. If None then the name will be
        tn.name + ' optimized', by default None.
    max_error_iterations : int, optional
        Maximal number of times the function will restart
        computation after error occurence, by default 10.
    max_iterations : int, optional
        Maximal number of algorithm iterations, by default 500.
    min_iterations : int, optional
        Minimal number of algorithm iterations, by default 10.
    eps : float, optional
        The algorithm stops when QFI in 5 consecutive iterations changes
        by less then eps, by default 1e-4.
    init_tn : QNetwork | None, optional
        Network from which initial values of variables will be taken.
        If None then initial values will be random, by default None
    print_messages : bool | str, optional
        Wheter to print messages about progress it made. Possible options:
        - True: all messages will be printed,
        - False: no messages will be printed,
        - 'partial': print only iteration summary.
        By default False.
    var_iterations : int, optional
        Number of iterations done solely over non measure variables (CPTP
        and MPS) before proceeding to measure (SLD) variables, by default
        1.
    sld_iterations : int, optional
        Number of iterations done solely over variable SLDs before
        proceeding to other variables, by default 1.
    art_noise_spaces : list[list[Hashable]] | None, optional
        Spaces on which artificial noise
        will act. For a list: [[s00, s01, ...], [s10, s11, ...], ...]
        noise will be applied to the whole list of spaces [s00, s01, ...]
        combined. If set to None then only spaces of ParamTensor elements
        will be noisy (separately), by default None.
    art_noise_params : tuple[float, float], optional
        For a tupple (a, l) noise will take a form:

        rho -> p * rho + (1 - p) * Id,
        
        where p = 1 - a * exp(-l * i) and i is the iteration number, by
        default (0.5, 0.1).
    contraction_order : list[str] | None, optional
        Names of tensors in the required order of contraction and
        therfore also the order of optimization. If None then the program
        will use BFS starting from a smallest variable without input
        spaces, by default None.
    adaptive_art_noise : bool, optional
        If True then when in a given iteration the increase of QFI coming
        from variable optimization is smaller then that coming from the
        artificial noise decay the l parameter of the noise will be
        doubled, by default True.

    Returns
    -------
    qfi : float
        Quantum Fisher information.
    qfiss : list[list[float]]
        QFI by connected component by iteration number.
    tn_opt : TensorNetwork
        Copy of tn with variables substituted with computed optimal
        values.
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
    config = IssConfig(
        name, max_error_iterations, max_iterations, min_iterations, eps,
        init_tn, print_messages, var_iterations, sld_iterations,
        art_noise_spaces, art_noise_params, contraction_order,
        adaptive_art_noise
    )
    config.check(tn)

    _tn, art_noise = _copy_network(tn, config)
    qfi = 0.0
    qfiss = []

    Result = namedtuple('Result', ['qfi', 'qfis', 'tn', 'status'])
    result = Result(-1, [], _tn, False)

    components = _tn.connected_components()
    for i, component in enumerate(components):
        if print_messages:
            print('-' * LINE_WIDTH)
            print(f'Component {i}')
            limited_print(component)
        for j in range(max_error_iterations):
            if print_messages:
                print('-' * LINE_WIDTH)
                print(f'Try number: {j + 1}')
            try:
                qfis = _connected_iss(
                    tn, _tn, component, art_noise, config
                )
                result = Result(qfis[-1], qfis, _tn, True)
                break
            except (SingleIterError, MaxIterExceededError) as e:
                if e.qfi > result.qfi:
                    result = Result(
                        e.qfi, e.qfis.copy(), _tn.copy(), False
                    )

                message = ''
                message = f"Solver failed in iteration {e.iteration}.\n"
                message += f"QFI: {e.qfi:.10f}.\n"
                message += str(e.cause)

                if j < max_error_iterations - 1:
                    message += "\nI will try again with another random"\
                        " initial data..."
                    warn(message)
                elif len(components) == 1:
                    break
                else:
                    raise RuntimeError('To many errors.') from e

        qfi += result.qfi
        qfiss.append(result.qfis)
    clean_tn = _clean_network(tn, _tn, art_noise, config)

    return qfi, qfiss, clean_tn, result.status


def _copy_network(tn: TensorNetwork, config: IssConfig
    ) -> tuple[TensorNetwork, ArtNoise]:
    _sd = copy.copy(tn.sdict)
    _sd.name = f'{tn.sdict.name} working copy'
    art_noise = ArtNoise(
        config.art_noise_spaces, config.art_noise_params, tn, _sd
    )
    chois = art_noise.new_tensors(tn)
    name = tn.name + ' working copy'
    _tn = TensorNetwork(tensors=chois, name=name, sdict=_sd)
    if config.contraction_order is not None:
        if _tn.free_spaces:
            raise ValueError(
                'Network has free spaces that could not be traced out:'\
                f' {_tn.free_spaces}.'
            )
    else:
        _tn: TensorNetwork = _tn.choi_trace(*_tn.free_spaces)

    return _tn, art_noise


def _clean_network(tn: TensorNetwork, _tn: TensorNetwork,
    art_noise: ArtNoise, config: IssConfig) -> TensorNetwork:
    tensors = []
    sd = tn.sdict
    for name, tensor in tn.tensors.items():
        if name in _tn.tensors and is_var(tensor):
            _tensor = _tn.tensors[name].respace(
                space_map=art_noise.unprimed, sdict=sd
            )
            if is_measurement(tensor):
                _tensor = _tensor.choi_T(*tensor.physical_spaces)
                _tensor.name = name
        else:
            _tensor = tensor
        tensors.append(_tensor)

    if config.name is None:
        name = tn.name + ' optimized'
    else:
        name = config.name
    return TensorNetwork(tensors=tensors, name=name, sdict=sd)


def _connected_iss(tn: TensorNetwork, _tn: TensorNetwork,
    component: list[str], art_noise: ArtNoise, config: IssConfig
    ) -> list[float]:
    """
    Performs ISS algorithm on a connected component of _qn defined by
    names provided in component argument.
    """
    names = _get_contr_order(tn, _tn, component, art_noise, config)

    cptp_vs, slds, mpss, mps_vs = _initialize_variables(
        tn, _tn, names, art_noise, config
    )

    qfis: list[float] = [0.0]
    noise_i = 0
    for i in range(config.max_iterations):
        last_qfi = qfis[-1]

        if i == 0: # _single_iteration updates slds2
            slds2 = {name: LT.square_without(LT.bond_spaces)
                     for name, LT in slds.items()}

        if config.print_full:
            print('-' * LINE_WIDTH)
            print(f'Iteration no. {i}')
            if config.art_noise_spaces:
                a = art_noise.a
                l = art_noise.l
                v = a * np.exp(-l * noise_i)
                print(
                    f'Art. noise strength: 1-p = {a}*exp(-{l}*{noise_i})'\
                    f' = {v:.5f}'
                )


        art_noise.update(noise_i, _tn)
        noise_i += art_noise.decay_step

        _qfis: list[float] = []
        try:
            if i == 0:
                for _ in range(config.sld_iterations):
                    _qfis += _single_iteration(
                        tn, _tn, names, slds, cptp_vs, slds,
                        slds2, mpss, mps_vs, config
                    )

            if config.var_iterations == 1 and config.sld_iterations == 1:
                _qfis += _single_iteration(
                    tn, _tn, names, {**cptp_vs, **slds, **mps_vs},
                    cptp_vs, slds, slds2, mpss, mps_vs, config
                )
            else:
                for _ in range(config.var_iterations):
                    _qfis += _single_iteration(
                        tn, _tn, names, {**cptp_vs, **mps_vs},
                        cptp_vs, slds, slds2, mpss, mps_vs, config
                    )

                for _ in range(config.sld_iterations):
                    _qfis += _single_iteration(
                        tn, _tn, names, slds, cptp_vs,
                        slds, slds2, mpss, mps_vs, config
                    )
        except (SolverError, NonHermitianError) as e:
            raise SingleIterError(e, i, last_qfi, qfis) from e

        noise_decay_inc = _qfis[0] - last_qfi
        iteration_inc = _qfis[-1] - _qfis[0]
        if config.adaptive_art_noise and noise_decay_inc > iteration_inc:
            art_noise.decay_step *= 2

        qfi = _qfis[-1]
        qfis.append(qfi)
        if config.print_messages:
            max_name_len = max(
                len(str(name)) for name in chain(cptp_vs, slds, mps_vs)
            )
            if config.print_full:
                spaces = max_name_len - 3
                print(
                    'QFI:' + ' ' * spaces,
                    f'{last_qfi:.10f} -> {qfi:.10f},',
                    f'inc: {qfi - last_qfi:.10f}'
                )
            else:
                spaces = len(str(config.max_iterations)) - len(str(i))
                print(
                    f'It. {i}' + ' ' * spaces,
                    f'QFI: {last_qfi:.10f} -> {qfi:.10f},',
                    f'inc: {qfi - last_qfi:.10f}'
                )

        if i >= config.min_iterations - 1:
            k = min(5, config.min_iterations)
            try:
                if abs((qfi - qfis[-k]) / qfis[-k]) < config.eps:
                    if config.print_messages:
                        print(f'Number of iterations: {i + 1}.')
                    return qfis
            except ZeroDivisionError:
                if config.print_messages:
                    print(f'Number of iterations: {i + 1}.')
                return qfis

    raise MaxIterExceededError(config.max_iterations, qfis[-1], qfis)


def _initialize_variables(tn: TensorNetwork, _tn: TensorNetwork,
    names: list[str], art_noise: ArtNoise, config: IssConfig
    ) -> tuple[
        OrderedDict[str, ConstTensor], OrderedDict[str, ConstTensor],
        OrderedDict[str, ConstTensor], OrderedDict[str, ConstTensor]
    ]:
    cptp_vs: OrderedDict[str, ConstTensor] = OrderedDict()
    slds: OrderedDict[str, ConstTensor] = OrderedDict()
    mpss: OrderedDict[str, ConstTensor] = OrderedDict()
    mps_vs: OrderedDict[str, ConstTensor] = OrderedDict()

    for name in names:
        if name not in tn.tensors:
            continue

        tensor = tn.tensors[name]
        if is_var(tensor):
            init_val = None
            init_tn = config.init_tn
            if init_tn and tensor.name in init_tn.tensors:
                init_val = init_tn.tensors[tensor.name]

            if is_measurement(tensor):
                if init_val:
                    new_tensor = init_val.choi_T(
                        *init_val.physical_spaces
                    )
                else:
                    new_tensor = tensor.random_sld(name)
            elif is_cptp_var(tensor):
                if init_val:
                    new_tensor = init_val
                else:
                    if is_comb_var(tensor):
                        new_tensor = tensor.random_comb(name)
                    else:
                        new_tensor = tensor.random_choi(name)
            else:
                if tensor.input_spaces:
                    raise ValueError(
                        'MPS type variable cannot have input spaces but '\
                        f'{tensor.input_spaces} were provided.'
                    )
                if init_val:
                    new_tensor = init_val
                else:
                    new_tensor = tensor.random_mps_element(name)
        else:
            new_tensor = tensor

        old_spaces = set(_tn.tensors[name].spaces)
        new_tensor = new_tensor.respace(
            spaces=[s if s in old_spaces else art_noise.primed(s)
                    for s in new_tensor.spaces],
            sdict=_tn.sdict, name=name
        )

        _tn.tensors[name] = new_tensor
        if is_measurement(tensor):
            slds[name] = new_tensor
        elif is_cptp_var(tensor):
            cptp_vs[name] = new_tensor
        elif is_mps(tensor):
            mpss[name] = new_tensor
            if is_var(tensor):
                mps_vs[name] = new_tensor
            else:
                mpss[name] = new_tensor

    if mps_vs:
        norm: complex = contr(
            *(
                mps.choi_trace(*mps.physical_spaces)
                for mps in mpss.values()
            )
        ).array[0]
        factor = norm**(1/len(mps_vs))
        for tensor in mps_vs.values():
            tensor.array /= factor

    return cptp_vs, slds, mpss, mps_vs


def _single_iteration(tn: TensorNetwork, _tn: TensorNetwork,
    names: list[str], to_opt: dict[str, ConstTensor],
    cptp_vs: dict[str, ConstTensor], slds: dict[str, ConstTensor],
    slds2: dict[str, ConstTensor], mpss: dict[str, ConstTensor],
    mps_vs: dict[str, ConstTensor], config: IssConfig) -> list[float]:
    """
    Makes one round of optimization of variables in to_opt.
    """
    _sd = _tn.sdict
    max_name_len = max(len(str(name)) for name in to_opt)

    rights_L: dict[str, ParamTensor] = {}
    rights_L2: dict[str, ConstTensor] = {}
    to_link_rl = [ParamTensor.from_const(_sd.choi_identity())]
    to_link_rl2 = [_sd.choi_identity()]
    # Not an identifier. Just to make sure it doesn't coincide with
    # existing names.
    last_name = uuid.uuid4().hex

    mps_rights: dict[str, ConstTensor] = {}
    last_mps = _sd.choi_identity()

    for name in names[::-1]:
        if name in to_opt:
            if last_name in names:
                rights_L[name] = contr(rights_L[last_name], *to_link_rl)
                rights_L2[name] = contr(rights_L2[last_name], *to_link_rl2)
            else:
                rights_L[name] = contr(*to_link_rl)
                rights_L2[name] = contr(*to_link_rl2)
            last_name = name
            to_link_rl = []
            to_link_rl2 = []

        to_link_rl.append(_tn.tensors[name])
        if name in slds:
            to_link_rl2.append(slds2[name])
        else:
            to_link_rl2.append(ParamTensor.to_const(_tn.tensors[name]))

        if name in mpss:
            mps_rights[name] = last_mps
            last_mps *= mpss[name].choi_trace(*mpss[name].physical_spaces)

    left_L = ParamTensor.from_const(_sd.choi_identity())
    left_L2 = _sd.choi_identity()
    sld_in_left_L2 = False
    # ^ As long as no sld was contracted left_L == left_L2.
    # It significantly improves error.
    to_link_ll = []
    to_link_ll2 = []
    mps_left = _sd.choi_identity()

    qfis = []
    first_to_opt = True
    for name in names:
        if name in to_opt:
            left_L = contr(left_L, *to_link_ll)
            if sld_in_left_L2:
                left_L2 = contr(left_L2, *to_link_ll2)
            else:
                left_L2 = ParamTensor.to_const(left_L)
            to_link_ll = []
            to_link_ll2 = []

            right_L = rights_L[name]
            right_L2 = rights_L2[name]
            m0: ConstTensor = (left_L * right_L).dtensor
            m1 = left_L2 * right_L2
            m: ConstTensor = _sd.choi_identity()
            # ^ m = 2*m0 - m1 will be computed later if needed.

            old_t = _tn.tensors[name]
            if name in cptp_vs:
                m = 2 * m0 - m1
                pre_qfi = np.real((m * old_t).array[0])
            elif name in slds:
                _x = m0 * slds[name]
                y = m1 * slds2[name]
                pre_qfi = np.real(
                    (2 * m0 * slds[name] - m1 * slds2[name]).array[0]
                )
            elif name in mps_vs:
                m = 2 * m0 - m1
                pre_qfi = np.real((m * old_t).array[0])
            else:
                raise ValueError(
                    f'Name {name} in to_opt but not in cptp, sld or mps_'\
                    'vs'
                )

            try:
                qfi, x = optimize_var(
                    tn, _tn, name, m0, m1, m, cptp_vs, slds, mps_vs,
                    mps_left, mps_rights, config
                )
            except NormMatZeroEigenval:
                # It will set increased = update = False.
                qfi = -float('inf')
                x = ConstTensor(m.spaces, sdict=_sd) # Dummy.

            increased = qfi > pre_qfi
            update = False
            bond_spaces = old_t.bond_spaces
            if name in cptp_vs:
                update = True
            elif name in slds:
                update = not bond_spaces or increased
            elif name in mps_vs:
                update = increased

            # Catch infinity
            if pre_qfi > 1 and abs(qfi / pre_qfi) > 1e5:
                warn(
                    'Optimizer returned unprobable result '
                    f'QFI: {pre_qfi:.10f} -> {qfi:.10f}. '
                    'Proceeding without node update.'
                )
                update = False

            if update:
                _tn.tensors[name] = x

                if name in cptp_vs:
                    cptp_vs[name] = x
                elif name in slds:
                    slds[name] = x
                    slds2[name] = x.square_without(bond_spaces)
                elif name in mps_vs:
                    mps_vs[name] = x
                    mpss[name] = x
            else:
                qfi = pre_qfi

            if first_to_opt:
                qfis.append(pre_qfi)
                first_to_opt = False
            qfis.append(qfi)

            if config.print_full:
                spaces = max_name_len - len(str(name))
                print(
                    f'{name}:' + ' ' * spaces,
                    f'{pre_qfi:.10f} -> {qfi:.10f},',
                    f'inc: {qfi - pre_qfi:.10f}'
                )

        to_link_ll.append(_tn.tensors[name])
        if name in slds:
            to_link_ll2.append(slds2[name])
            sld_in_left_L2 = True
        else:
            to_link_ll2.append(ParamTensor.to_const(_tn.tensors[name]))

        if name in mpss:
            mps = mpss[name]
            mps_left *= mps.choi_trace(*mps.physical_spaces)

    return qfis


def _get_contr_order(tn: TensorNetwork, _tn: TensorNetwork,
    component: list[str], art_noise: ArtNoise, config: IssConfig
    ) -> list[str]:
    if config.contraction_order is None:
        return _bfs(tn, _tn, component)

    component_set = set(component)
    was_added = set()
    names = []
    for name in config.contraction_order:
        if name not in component_set:
            continue

        names.append(name)

        for neighbor in _tn.neighbors(name):
            if (
                neighbor in art_noise.tensor_info
                and neighbor not in was_added
            ):
                names.append(neighbor)
                was_added.add(neighbor)

    return names


def _bfs(tn: TensorNetwork, _tn: TensorNetwork, component: list[str]) -> list[str]:
    min_dim_inp = float('inf')
    min_dim = float('inf')
    min_ch_inp = ''
    min_ch = ''
    for name in component:
        ch = _tn.tensors[name]

        if not ch.input_spaces and ch.dimension < min_dim_inp:
            min_dim_inp = ch.dimension
            min_ch_inp = name

        if ch.dimension < min_dim:
            min_dim = ch.dimension
            min_ch = name

    q: Queue[str] = Queue()
    checked = {name: False for name in component}
    if min_dim_inp < float('inf'):
        name0 = min_ch_inp
    else:
        name0 = min_ch
    q.put(name0)
    names = []

    while not q.empty():
        name = q.get()
        if checked[name]:
            continue
        names.append(name)
        checked[name] = True

        neighbors = []
        for _name in _tn.neighbors(name):
            if checked[_name]:
                continue
            i = 0
            if _name in tn.tensors and is_measurement(tn.tensors[_name]):
                i = 1
            neighbors.append((i, _name))

        for _, _name in sorted(neighbors, key=fst):
            q.put(_name)
    return names
