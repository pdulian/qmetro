from __future__ import annotations

from collections.abc import Hashable
from copy import copy

import numpy as np

from ...utils import flatten

from ..classes import (
    SpaceDict, ConstTensor, VarTensor, TensorNetwork, DEFAULT_SDICT
)




def mps_var_tnet(output_spaces: list[Hashable], name: str,
    sdict: SpaceDict = DEFAULT_SDICT, mps_bond_dim: int = 1, **kwargs
    ) -> tuple[TensorNetwork, list[str], list[Hashable]]:
    """
    Creates a variable density matrix of a pure matrix product state
    (MPS). This density matrix is a matrix product operaotr (MPO).

    The result is a network of variable tensors: 
    
        ``T_0``, ``T_1``, ..., ``T_n-1``.
    
    The i-th tensor has inidices called: ``O_i``, ``B_i-1``, ``B_i`` where
    - ``O_i`` represent Hilbert spaces on which the state acts (output
    spaces),
    - ``B_i`` are bond spaces,
    - ``B_-1`` and ``B_n-1`` do not exist.
    Then the contraction of all tensors by the bond spaces with the same
    name yields a density matrix of a pure state on
    
        ``O_0 (x) ... (x) O_n-1``.

    Bond spaces are defined for MPO elements and thus their dimension is
    the square of the MPS bond dimension.

    Parameters
    ----------
    output_spaces : list[Hashable]
        Spaces of the state, i-th element is the space ``O_i``.
    name : str
        Name of the result. Tensors will have names in format
        f"{name}, {i}" and connecting spaces ``B_i`` = (name, 'BOND', i).
    sdict : SpaceDict, optional
        Space dictionary of the result, by default DEFAULT_SDICT.
    mps_bond_dim : int, optional
        Dimension of the MPS connecting (bond) spaces, by default 1.
    **kwargs :
        Arguments passed to VarTensor constructor.

    Returns
    -------
    mps : TensorNetwork
        Tensor network of variable tensors representing MPO of the density
        matrix.
    tensor_names : list[str]
        List of tensors' names. In order from i=0 to i=n-1.
    bonds : list[Hashable]
        List of bond spaces' names. In order from i=0 to i=n-2.
    """
    n = len(output_spaces)
    sd = sdict

    bonds = sd.arrange_bonds(n - 1, mps_bond_dim**2, (name, 'BOND'))

    mps_elements = []
    names = []
    for i in range(n):
        loc_bonds = []
        if i > 0:
            loc_bonds.append(bonds[i - 1])
        if i < n - 1:
            loc_bonds.append(bonds[i])

        el_name = f'{name}, {i}'
        names.append(el_name)

        spaces = [output_spaces[i]] + loc_bonds
        mps = VarTensor(spaces, sd, el_name, [output_spaces[i]], **kwargs)
        mps_elements.append(mps)

    return TensorNetwork(mps_elements, sd, name), names, bonds


def input_state_var(output_spaces: list[Hashable], name: str,
    sdict: SpaceDict = DEFAULT_SDICT, mps_bond_dim: int | None = None,
    **kwargs) -> VarTensor | TensorNetwork:
    """
    Creates a state variable.

    Parameters
    ----------
    output_spaces : list[Hashable]
        Spaces of the state.
    name : str
        Name of the result. If `mps_bond_dim` is provided it will use the
        naming convention of :func:`mps_var_tnet`.
    sdict : SpaceDict, optional
        Space dictionary of the result, by default `DEFAULT_SDICT`.
    mps_bond_dim : int, optional
        Dimension of the MPS connecting (bond) spaces. If None then it will
        return the state on the whole product Hilbert space (as if bond
        dimension was infinite). By default None.
    **kwargs :
        Arguments passed to VarTensor constructor.

    Returns
    -------
    state : VarTensor | TensorNetwork
        Tesnor representing the density matrix.

    Notes
    -----
    In case one needs names of the MPS elements and bond spaces use
    `mps_var_tnet`.
    """
    if mps_bond_dim is None:
        return VarTensor(
            output_spaces, sdict, name, output_spaces, **kwargs
        )

    if mps_bond_dim == 1:
        vts = []
        sd = sdict
        for i, space in enumerate(output_spaces):
            vt = VarTensor(
                [space], sd, f'{name}, {i}', output_spaces=[space],
                **kwargs
            )
            vts.append(vt)
        return TensorNetwork(vts, sd, name)

    return mps_var_tnet(
        output_spaces, name, sdict, mps_bond_dim, **kwargs
    )[0]


def mpo_measure_var_tnet(input_spaces: list[Hashable], name: str,
    sdict: SpaceDict = DEFAULT_SDICT, bond_dim: int = 1, **kwargs
    ) -> tuple[TensorNetwork, list[str], list[Hashable]]:
    """
    Creates a variable measurement (SLD) matrix product operator (MPO).

    The result is a network of variable tensors: ``T_0``, ``T_1``, ...,
    ``T_n-1``.
    The i-th tensor has inidices called: ``I_i``, ``B_i-1``, ``B_i`` where
    - ``I_i`` represent Hilbert spaces which are measured (input spaces),
    - ``B_i`` are bond spaces,
    - ``B_-1`` and ``B_n-1`` do not exist.
    Then the contraction of all tensors by the bond spaces with the same
    name yields a variable of a measurement on ``I_0 (x) ... (x) I_n-1``.

    Parameters
    ----------
    input_spaces : list[Hashable]
        Measured spaces, i-th element is the space ``I_i``.
    name : str
        Name of the result. Elements will have names in format
        f"{name}, {i}" and connecting spaces in (name, 'BOND', i),
        where i is the element/space number.
    sdict : SpaceDict, optional
        Space dictionary of the result, by default DEFAULT_SDICT.
    bond_dim : int, optional
        Dimension of the connecting (bond) spaces, by default 1.
    **kwargs :
        Arguments passed to VarTensor constructor.

    Returns
    -------
    sld : TensorNetwork
        Tensor network representing MPO of the measurement (SLD).
    elements : list[str]
        List of elements' names. In order from i=0 to i=n-1.
    bonds : list[Hashable]
        List of bond spaces' names. In order from i=0 to i=n-2.
    """
    sd = sdict
    n = len(input_spaces)
    bonds = sd.arrange_bonds(n - 1, bond_dim, (name, 'BOND'))

    measures = []
    names = []
    for i in range(n):
        loc_bonds = []
        if i > 0:
            loc_bonds.append(bonds[i - 1])
        if i < n - 1:
            loc_bonds.append(bonds[i])

        el_name = f'{name}, {i}'
        names.append(el_name)

        m = VarTensor(
            [input_spaces[i]] + loc_bonds, sdict=sd, name=el_name,
            is_measurement=True, **kwargs
        )
        measures.append(m)

    return TensorNetwork(measures, sd, name), names, bonds


def measure_var(input_spaces: list[Hashable], name: str,
    sdict: SpaceDict = DEFAULT_SDICT, bond_dim: int | None = None,
    **kwargs) -> VarTensor | TensorNetwork:
    """
    Creates a measurement variable.

    Parameters
    ----------
    input_spaces : list[Hashable]
        Measured spaces.
    name : str
        Name of the result. If mps_bond_dim is provided it will use the
        naming convention of mpo_measure_var_tnet.
    sdict : SpaceDict
        Space dictionary of the result, by default DEFAULT_SDICT.
    bond_dim : int | None, optional
        Dimension of the connecting (bond) spaces. If None then it will
        return the measurement on the whole product Hilbert space (as if
        bond dimension was infinite). By default None.
    **kwargs :
        Arguments passed to VarTensor constructor.

    Returns
    -------
    VarTensor | TensorNetwork
        Tensor representing the measurement.

    Notes
    -----
    In case one needs names of the MPS elements and bond spaces use
    `mpo_measure_var_tnet`.
    """
    if bond_dim is None:
        return VarTensor(
            input_spaces, sdict, name, is_measurement=True, **kwargs
        )

    if bond_dim == 1:
        vts = []
        sd = sdict
        for i, space in enumerate(input_spaces):
            vt = VarTensor(
                [space], sd, f'{name}, {i}', is_measurement=True, **kwargs
            )
            vts.append(vt)
        return TensorNetwork(vts, sd, name)

    return mpo_measure_var_tnet(
        input_spaces, name, sdict, bond_dim, **kwargs
    )[0]


def cmarkov_param_tnet(krauses: list[np.ndarray],
    dkrauses: list[np.ndarray], input_spaces: list[list[Hashable]],
    output_spaces: list[list[Hashable]], transition_mat: np.ndarray,
    name: str, sdict : SpaceDict = DEFAULT_SDICT,
    initial_prob: np.ndarray | None = None, trace_end: bool = True,
    **kwargs) -> tuple[TensorNetwork, list[str], list[Hashable]]:
    """
    Returns Choi matrices of channels whith classical Markovian
    correlation of their Kraus operators. The number of repetitions is
    derived from the number of lists in input_spaces.

    Parameters
    ----------
    krauses : list[np.ndarray]
        Kraus operators.
    dkrauses : list[np.ndarray]
        Derivatives of Kraus operators.
    input_spaces : list[list[Hashable]]
        List of lists of input spaces. The ith list is a list of input
        spaces of the ith channel.
    output_spaces : list[list[Hashable]]
        List of lists of output spaces. The ith list is a list of output
        spaces of the ith channel.
    transition_mat : np.ndarray
        Matrix M of conditional probabilities. The coefficient M[i, j] is
        equal to ``P(i|j)`` - the conditional probability of ith Kraus
        operator given the occurence of jth operator in the previous step.
    name : str
        Name of the result. Elements have names in format f"{name}, {i}"
        where i is the element number and its input and output correlation
        spaces are (name, 'ENVIRONMENT', i) and (name, 'ENVIRONMENT', i+1)
        respectively.
    sdict : SpaceDict, optional
        Space dictionary of the result, by default DEFAULT_SDICT.
    initial_prob : np.ndarray | None, optional
        Initial probabilities in a Markov chain. If None then the input of
        the correlation space - (name, 'ENVIRONMENT', 0) is left open, by
        default None.
    trace_end : bool, optional
        Whether to trace out the output of the correlation space -
        (name, 'ENVIRONMENT', n_of_repetitions), by default True.
    **kwargs :
        Arguments passed to ParamTensor constructor.

    Returns
    -------
    corrlated_channels : TensorNetwork
        Tensor network rerpresenting the Choi matrices of correlated
        channels.
    channel_names : list[str]
        List of channels' names. In order from i=0 to i=n-1.
    environments : list[Hashable]
        List of environment spaces' names. In order from i=0 to i=n-2.
    """
    raise NotImplementedError
    sd = sdict
    d_env= len(krauses)
    n = len(input_spaces)

    envs = sd.arrange_spaces(n + 1, d_env, (name, 'ENVIRONMENT'))

    corr_ks = cmarkov_krauses(krauses, transition_mat)
    corr_dks = cmarkov_krauses(dkrauses, transition_mat)

    pchs = []
    names = []
    for i in range(n):
        loc_inp = [envs[i], *input_spaces[i]]
        loc_out = [envs[i + 1], *output_spaces[i]]
        loc_name = f"{name}, {i}"

        pch = choi_tensor_from_krauses(
            corr_ks, loc_inp, loc_out, sd, corr_dks, **kwargs
        )

        if i == 0 and initial_prob is not None:
            pch *= ConstTensor(
                envs[0], choi=np.diag(initial_prob), sdict=sd
            )
            loc_inp.remove(envs[0])
        if i == n - 1 and trace_end:
            pch = pch.choi_trace(envs[i + 1])
            loc_out.remove(envs[i + 1])

        pch.name = loc_name
        pch.input_spaces = loc_inp

        pchs.append(pch)
        names.append(loc_name)

    return TensorNetwork(pchs, sd, name), names, envs


def comb_var_tnet(structure: list[tuple[list[Hashable], list[Hashable]]],
    name: str, sdict: SpaceDict = DEFAULT_SDICT, ancilla_dim : int = 1,
    **kwargs) -> tuple[TensorNetwork, list[str], list[Hashable]]:
    """
    Returns a network of variable tensors representing channels in a
    sequence such that part of the output of one channel is a part of the
    input of the next one.

    More precisely, it returns a network of n variable tensor representing
    channels called teeth:

        ``T_i: L(I_i (x) A_i-1) -> L(O_i (x) A_i)``,

    where:
    - ``I_i``, ``O_i`` and ``A_i`` are input, output and ancilla Hilbert
    spaces,
    - ``(x)`` is a tensor product,
    - ``L(H)`` is a set of linear operators on H,
    - ``A_(-1)`` and ``A_n`` are trivial.

    Parameters
    ----------
    structure : list[tuple[list[Hashable], list[Hashable]]]
        List defining the spaces of every tooth. For i-th tooth and
        ``inp_i, out_i = structure[i]`` where ``inp_i`` (``out_i``)
        is a list of input (output) spaces of the i-th tooth excluding
        the ancilla.
    name : str
        Name of the result. Elements have names in format f"{name}, {i}"
        where i is the element number and its input and output ancilla
        spaces are (name, 'ANCILLA', i-1) and (name, 'ANCILLA', i)
        respectively.
    sdict : SpaceDict, optional
        Space dictionary, by default DEFAULT_SDICT.
    ancilla_dim : int, optional
        Dimension of ancilla spaces, by default 1.
    **kwargs :
        Arguments passed to VarTensor constructor.

    Returns
    -------
    network : TensorNetwork
        Tensor network of a comb.
    channel_names : list[str]
        List of teeth's names: ``[T_0, T_1, ..., T_n-1]``.
    ancillas : list[Hashable]
        List of ancilla spaces' names: ``[A_0, A_1, ..., A_n-2]``.
    """
    n = len(structure)
    ancillas = sdict.arrange_spaces(n - 1, ancilla_dim, (name, 'ANCILLA'))

    teeth = []
    names = []
    for i, (exter_tooth_inp, exter_tooth_out) in enumerate(structure):
        tooth_name = f'{name}, tooth: {i}'
        tooth_inputs = copy(exter_tooth_inp)
        tooth_outputs = copy(exter_tooth_out)

        if i > 0:
            tooth_inputs.append(ancillas[i - 1])

        if i < n - 1:
            tooth_outputs.append(ancillas[i])

        tooth = VarTensor(
            tooth_inputs + tooth_outputs, sdict, tooth_name, tooth_outputs,
            **kwargs
        )
        teeth.append(tooth)
        names.append(tooth_name)

    return TensorNetwork(teeth, sdict, name), names, ancillas


def comb_var(structure: list[tuple[list[Hashable], list[Hashable]]],
    name: str, sdict: SpaceDict = DEFAULT_SDICT,
    ancilla_dim : int | None = None, **kwargs
    ) -> VarTensor | TensorNetwork:
    """
    Creates a comb variable.

    Parameters
    ----------
    structure : list[tuple[list[Hashable], list[Hashable]]]
        List defining the spaces of every tooth. For i-th tooth and
        ``inp_i, out_i = structure[i]`` where ``inp_i`` (``out_i``)
        is a list of input (output) spaces of the i-th tooth excluding
        the ancilla.
    name : str
        Name of the result. Elements have names in format f"{name}, {i}"
        where i is the element number and its input and output ancilla
        spaces are (name, 'ANCILLA', i-1) and (name, 'ANCILLA', i)
        respectively.
    sdict : SpaceDict, optional
        Space dictionary, by default DEFAULT_SDICT.
    ancilla_dim : int | None, optional
        Dimension of ancilla spaces. When this argument is:
        - a number: a tensor network of control operations will be
        created,
        - None: one comb-like variable tensor will be created,
        by default None.
    **kwargs :
        Arguments passed to VarTensor constructor.

    Returns
    -------
    VarTensor | TensorNetwork
        Tensor of comb variable.
    """
    if ancilla_dim is None:
        return VarTensor(
            flatten(structure, 2), sdict=sdict, comb_structure=structure,
            name=name, **kwargs
        )

    return comb_var_tnet(
        structure, name, sdict, ancilla_dim, **kwargs
    )[0]
