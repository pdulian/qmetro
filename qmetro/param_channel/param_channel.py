from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Callable, Any
from itertools import repeat
from warnings import warn

import numpy as np

from ..qmtensor import ParamTensor, DEFAULT_SDICT, SpaceDict, ConstTensor
from ..qtools import (
    choi_from_krauses, dchoi_from_krauses, ket_bra, hc,
    krauses_from_choi, dkrauses_from_choi, krauses_kron,
    krauses_sequential, choi_dchoi_from_lindblad
)




class ParamChannel():
    """
    Class representing parametrized channels which can additionally act
    on an "environment" space that is used to construct correlations
    between multiple channels [1]_.
    
    More precisely it is representing a CPTP map parametrized by t:

        F_t: L(E_i (x) I) -> L(E_o (x) O),
    
    where L(X) is the space of linear operators on Hilbert space X and
    - (x) is tensor product of spaces,
    - I and O are spaces of input and output respectively,
    - E_i and E_o are input and output environment spaces respectively.
    
    Note that formally the input space of F_t is E_i (x) I. Thus to avoid
    confusion E_i (x) I is called 'total input space' and I is called
    'input space'. Anlogously, E_o (x) O is called 'total output space' 
    and O is called 'output space'.

    Objects can be initialized in three different ways:
    1) By providing kraus operators and their derivatives
    2) By providing choi matrix and its derivative
    3) By providing lindbladian function, its derivative over parameter,
    time of evolution and dimension of Hilber space. In this mode, choi
    and dchoi are calculated firstly.
    
    Class constructor takes keyword-only arguments.

    Parameters
    ----------
    krauses : list[np.ndarray] | None, optional
        Kraus operators, by default None. Each operator is assumed to act
        from E_i (x) I to E_o (x) O.
    choi : np.ndarray | None, optional
        Choi matrix, by default None. It is assumed to be a matrix on
        E_o (x) O (x) E_i (x) I.
    dkrauses : list[np.ndarray] | None, optional
        Derivatives of Kraus operators, by default None. Each operator is
        assumed to act from E_i (x) I to E_o (x) O.
    dchoi : np.ndarray | None, optional
        Derivative of Choi matrix, by default None. It is assumed to be a
        matrix on E_o (x) O (x) E_i (x) I.
    lindblad: Callable[[np.ndarray], np.ndarray], optional
        Lindbladian: function, that for input density matrix returns its 
        derivative over time; lindblad(rho) = drho/dt. It is assumed that
        Lindbladian is constant over time. rho is matrix on E_i (x) I,
        drho/dt is matrix on E_o (x) O, dim(rho) = dim(drho/dt)
    dlindblad: Callable[[np.ndarray], np.ndarray], optional
        Function that for input density matrix returns derivative of
        Lindbladian over some parameter. For example 
        dlindblad(rho) =  0.5j * (rho@sigma_z - sigma_z@rho) when parameter
        is angular velocity around z, sigma_z is a z Pauli matrix.
    time: float, optional
        Time of evolution with a given lindbladian
    env_dim : int | tuple[int,int], optional
        Environment dimension, that is the value of dimE_i = dimE_o or a
        tuple (E_i, E_o). By default 1.
    sdict : SpaceDict, optional
        Space distionary of the tensor, by default DEFAULT_SDICT.
    input_dim : int | None, optional
        Dimenion of input space I or for I = I_0 (x) ... (x) I_n-1 it
        should be a list [dimI_0, ..., dimI_n-1]. If None then this
        dimension is derived from other arguments. Must be provided in
        lindbladian input mode.
    output_dim : int | None, optional
        Dimenion of output space O or for O = O_0 (x) ... (x) O_n-1 it
        should be a list [dimO_0, ..., dimO_n-1]. If None then this
        dimension is derived from other arguments.
    **kwargs: Any
        Keyword arguments passed to lindblad and dlindblad

    Bibliography
    ------
    [1] S. Kurdziałek et al. Quantum metrology using quantum combs and
    tensor network formalism


    Raises
    ------
    ValueError
        Raised when 2 out of 3 possible input modes are mixed or
        when `env_dim` and/or `channel_out_dim` are incompatible with the
        shape of Kraus operators/Choi matrix.
    """
    name_prefix = 'PARAM CHANNEL '
    counter = 0


    def __init__(self, *, krauses: list[np.ndarray] | None = None,
        choi: np.ndarray | None = None,
        dkrauses: list[np.ndarray] | None = None,
        dchoi: np.ndarray | None = None,
        lindblad: Callable[[np.ndarray], np.ndarray] | None = None,
        dlindblad: Callable[[np.ndarray], np.ndarray] | None = None,
        time: float | None = None,
        env_dim: int | tuple[int, int] = 1,
        sdict: SpaceDict = DEFAULT_SDICT,
        input_dim: int | list[int] | None = None,
        output_dim: int | list[int] | None = None, **kwargs: Any):

        kraus_like_inp = krauses is not None or dkrauses is not None
        if krauses is not None and dkrauses is None:
            dkrauses = [np.zeros_like(k) for k in krauses]

        choi_like_inp = choi is not None or dchoi is not None
        if choi is not None and dchoi is None:
            dchoi = np.zeros_like(choi)
        
        lindblad_like_inp = (
            (lindblad is not None or dlindblad is not None)
            and time is not None and input_dim is not None
        )
        if lindblad is not None and dlindblad is None:
            def _dlindblad(rho):
                return np.zeros_like(rho)
            dlindblad = _dlindblad
        
        self.kraus_like_inp = kraus_like_inp
        #when lindblad is given, choi will be computed
        self.choi_like_inp = choi_like_inp or lindblad_like_inp

        not_enough_inputs = (not kraus_like_inp and not choi_like_inp
                            and not lindblad_like_inp)
        
        too_many_inputs = ((kraus_like_inp and choi_like_inp) or
                           (kraus_like_inp and lindblad_like_inp) or
                           (choi_like_inp and lindblad_like_inp))

        if not_enough_inputs:
            raise ValueError(
                'Either krauses, dkrauses or choi, dchoi or lindblad, '\
                'dlindblad, time, input_dim sets of arguments must be '\
                'provided.'
            )
        if too_many_inputs:
            raise ValueError(
                'Only one of krauses, dkrauses or choi, dchoi or '\
                'lindblad, dlindblad, time, input_dim sets of arguments '\
                'must be provided.'
            )

        if isinstance(env_dim, Iterable):
            self.env_inp_dim, self.env_out_dim = env_dim
        else:
            self.env_inp_dim = self.env_out_dim = env_dim

        if kraus_like_inp:
            total_out_dim, total_inp_dim = krauses[0].shape
            if total_inp_dim % self.env_inp_dim != 0:
                raise ValueError(
                    f'Kraus operator input dimension ({total_inp_dim}) '\
                    'must be divisible by env_inp_dim ('\
                    f'{self.env_inp_dim}).'
                )
            if total_out_dim % self.env_out_dim != 0:
                raise ValueError(
                    f'Kraus operator output dimension ({total_out_dim}) '\
                    f'must be divisible by env_out_dim ('\
                    f'{self.env_out_dim}).'
                )

            self.output_dim = total_out_dim // self.env_out_dim
            self.input_dim = total_inp_dim // self.env_inp_dim

            if output_dim is None:
                self.output_dims = [self.output_dim]
            elif isinstance(output_dim, Iterable):
                if self.output_dim != np.prod(output_dim):
                    raise ValueError(
                        f'Output tensor structure {tuple(output_dim)} is'\
                        ' incompatible with output dimension derived '\
                        f'from Kraus operators ({self.output_dim}).'
                    )
                self.output_dims = list(output_dim)
            else:
                self.output_dims = [self.output_dim]
                warn(
                    'When Kraus operators are provided output dimension '\
                    'is derived from Kruas operators.'
                )
            if input_dim is None:
                self.input_dims = [self.input_dim]
            elif isinstance(input_dim, Iterable):
                if self.input_dim != np.prod(input_dim):
                    raise ValueError(
                        f'input tensor structure {tuple(input_dim)} is'\
                        ' incompatible with input dimension derived '\
                        f'from Kraus operators ({self.input_dim}).'
                    )
                self.input_dims = list(input_dim)
            else:
                self.input_dims = [self.input_dim]
                warn(
                    'When Kraus operators are provided input dimension '\
                    'is derived from Kruas operators.'
                )
        else:
            #choi or lindblad like input
            #for lindbladian input, first calculate choi, dchoi
            #then proceed like for choi input
            if lindblad_like_inp:
                if isinstance(input_dim, Iterable):
                    raise ValueError(
                        'In lindblad input mode input_dim cannot be a '
                        'list, it must be an integer.'
                    )
                #TODO: maybe allow input_dim to be a list

                choi, dchoi = choi_dchoi_from_lindblad(
                    lindblad, dlindblad, input_dim * self.env_inp_dim,
                    time, **kwargs
                )
            
            choi_dim = len(choi)
            env_inp_out_dim = self.env_inp_dim * self.env_out_dim

            if choi_dim % (env_inp_out_dim) != 0:
                raise ValueError(
                    f'Choi matrix dimension ({choi_dim}) must be '\
                    'divisible by the product of env_inp_dim and '\
                    f'env_out_dim ({self.env_inp_dim} * '\
                    f'{self.env_out_dim} = {env_inp_out_dim}).'
                )
            inp_out_dim = choi_dim // env_inp_out_dim

            if output_dim is None and input_dim is None:
                x = int(np.sqrt(inp_out_dim))
                self.output_dim = self.input_dim = x
                self.output_dims = [x]
                self.input_dims = [x]
                if x**2 != inp_out_dim:
                    raise ValueError(
                        'If output_dim and input_dim are not provided '\
                        'input and output dimensions are assumed to be '\
                        'equal which requires that Choi matrix dimension'\
                        'divided by dimensions of environments ('\
                        f'{choi_dim} / {env_inp_out_dim} = {inp_out_dim}'\
                        ') is a square of natural number.'
                    )
            elif output_dim is not None:
                if isinstance(output_dim, Iterable):
                    self.output_dim = np.prod(output_dim)
                    self.output_dims = list(output_dim)
                else:
                    self.output_dim = output_dim
                    self.output_dims = [output_dim]

                if input_dim is not None:
                    if isinstance(input_dim, Iterable):
                        self.input_dim = np.prod(input_dim)
                        self.input_dims = list(input_dim)
                    else:
                        self.input_dim = input_dim
                        self.input_dims = [input_dim]
                else:
                    self.input_dim = inp_out_dim // self.output_dim
                    self.input_dims = [self.input_dim]
            else: # input_dim is not None
                if isinstance(input_dim, Iterable):
                    self.input_dim = np.prod(input_dim)
                    self.input_dims = list(input_dim)
                else:
                    self.input_dim = input_dim
                    self.input_dims = [input_dim]

                self.output_dim = inp_out_dim // self.input_dim
                self.output_dims = [self.output_dim]

            if self.input_dim * self.output_dim != inp_out_dim:
                raise ValueError(
                    f'Provided input_dim ({input_dim}), output_dim ('\
                    f'{output_dim}) and env_dim ({env_dim}) are not '\
                    'compatible with the dimension of the Choi '\
                    f'matrix ({choi_dim}).'
                )

        self._choi = choi
        self._dchoi = dchoi
        self._krauses = krauses
        self._dkrauses = dkrauses
        self._tensor = None

        self.input_dim: int
        self.output_dim: int
        self.input_dims: list[int]
        self.output_dims: list[int]

        if len(self.input_dims) != len(self.output_dims):
            raise ValueError(
                f'Arguments input_dim ({input_dim}) and output_dim ('\
                f'{output_dim}) should define the same number of spaces. '\
                'Note that one can alwyays define a trivial i.e. one-'\
                'dimensional space.'
            )

        #removing trivial teeth (with input and output 1)
        input_dims_new = []
        output_dims_new = []
        for inp_dim, out_dim in zip(self.input_dims, self.output_dims):
            if inp_dim != 1 or out_dim != 1:
                input_dims_new.append(inp_dim)
                output_dims_new.append(out_dim)
        if not input_dims_new: #trivial input and output
            self.input_dims = self.output_dims = [1]
        else:
            self.input_dims = input_dims_new
            self.output_dims = output_dims_new

        self.id = ParamChannel.counter
        ParamChannel.counter += 1
        self.sdict = sdict

        self.input_spaces = []
        self.output_spaces = []

        for i, d in enumerate(self.input_dims):
            space = 'INPUT', i, self.id
            self.input_spaces.append(space)
            self.sdict[space] = d

        for i, d in enumerate(self.output_dims):
            space = 'OUTPUT', i, self.id
            self.output_spaces.append(space)
            self.sdict[space] = d

        self.env_inp = 'ENV INP', self.id
        self.env_out = 'ENV OUT', self.id
        self.sdict[self.env_inp] = self.env_inp_dim
        self.sdict[self.env_out] = self.env_out_dim


    @property
    def single_tooth(self) -> bool:
        """
        Tells whether the channel is a single-tooth comb
        (i.e. it has only one input and one output space).

        Returns
        -------
        bool
            `len(self.input_spaces) == 1`
        """
        return len(self.input_spaces) == 1


    @property
    def trivial_env_inp(self) -> bool:
        """
        Tells whether the input environment space E_i is trivial.

        Returns
        -------
        bool
            `self.env_inp_dim == 1`
        """
        return self.env_inp_dim == 1


    @property
    def trivial_env_out(self) -> bool:
        """
        Tells whether the input environment space E_o is trivial.

        Returns
        -------
        bool
            `self.env_out_dim == 1`
        """
        return self.env_out_dim == 1


    @property
    def trivial_env(self) -> bool:
        """
        Tells whether the channel has trivial environment.

        Returns
        -------
        bool
            `self.trivial_env_inp and self.trivial_env_out`.
        """
        return self.trivial_env_inp and self.trivial_env_out


    @property
    def total_input_dim(self) -> int:
        """
        Returns
        -------
        int
            Dimension of the total input space, dim E_i (x) I.
        """
        return self.input_dim * self.env_inp_dim


    @property
    def total_output_dim(self) -> int:
        """
        Returns
        -------
        int
            Dimension of the total output space, dim E_o (x) O.
        """
        return self.output_dim * self.env_out_dim


    @property
    def total_input_dims(self) -> list[int]:
        """
        Returns
        -------
        int
            Dimensions of the total input space, [dimE_i, I_0, ...].
        """
        return [self.env_inp_dim] + self.input_dims


    @property
    def total_output_dims(self) -> list[int]:
        """
        Returns
        -------
        int
            Dimensions of the total output space, [dimE_o, O_0, ...].
        """
        return [self.env_out_dim] + self.output_dims


    @property
    def total_input_spaces(self) -> list[Hashable]:
        """
        Returns
        -------
        int
            Names of factors in the total input space, [E_i, I_0, ...].
        """
        return [self.env_inp] + self.input_spaces


    @property
    def total_output_spaces(self) -> list[Hashable]:
        """
        Returns
        -------
        int
            Names of factors in the total output space, [E_o, O_0, ...].
        """
        return [self.env_out] + self.output_spaces
    

    @property
    def is_comb(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the object represent a comb (True) or a single
            channel (False).
        """
        return len(self.input_spaces) > 1
    
    
    def tensor(self, input_spaces: list[Hashable] | None = None,
        output_spaces: list[Hashable] | None = None,
        env_inp: Hashable | None = None, env_out: Hashable | None = None,
        sdict: SpaceDict | None = None, **kwargs) -> ParamTensor:
        """
        Returns parametrized tensor of self's Choi matrix. Names of its
        inidices are in attributes of self:
        - input_spaces, output_spaces,
        - env_inp_space, env_out_space,
        - total_input_spaces which is equal to [env_inp_space,
        *input_spaces],
        - total_output_spaces which is equal to [env_out_space,
        *output_spaces].

        or can be provided in arguments.

        Parameters
        ----------
        input_spaces : list[Hashable] | None, optional
            Tensor's input spaces names. They should be in the causal order
            (as in self.input dims), by default None.
        output_spaces : list[Hashable] | None, optional
            Tensor's output spaces names. They should be in the causal
            order (as in self.output dims), by default None.
        env_inp : Hashable | None, optional
            Name of the environment input space. If the environment input
            is trivial this space will be omitted. By default None.
        env_out : Hashable | None, optional
            Name of the environment output space. If the environment output
            is trivial this space will be omitted. By default None.
        sdict : SpaceDict, optional
            Tensor's space dicitionary, by default `self.sdict`.

        Returns
        -------
        ParamTensor
            Parametrized tensor of channel's Choi matrix.
        """
        provided_spaces = any([
            input_spaces, output_spaces, env_inp, env_out, sdict
        ])
        sd = sdict or self.sdict
        
        if not provided_spaces:
            if self._tensor is None:
                self._tensor = ParamTensor(
                    [self.env_out, *self.output_spaces, self.env_inp,
                    *self.input_spaces], sdict=self.sdict,
                    output_spaces=[self.env_out] + self.output_spaces,
                    choi=self.choi(), dchoi=self.dchoi(), **kwargs
                )
            return self._tensor
        
        # Check if necessary spaces are provided.
        if None in (input_spaces, output_spaces):
            raise ValueError(
                'Both input and output spaces must be provided'
            )
        env_inps = [env_inp]
        if env_inp is None:
            if self.trivial_env_inp:
                env_inps = []
            else:
                raise ValueError(
                    'For channel with non-trivial environment input'
                    'env_inp must be provided.'
                )
        env_outs = [env_out]
        if env_out is None:
            if self.trivial_env_out:
                env_outs = []
            else:
                raise ValueError(
                    'For channel with non-trivial environment output'
                    'env_out must be provided.'
                )
        
        # Check if spaces are in sdict and have correct dimensions.
        spaces = env_outs + output_spaces + env_inps + input_spaces

        dims = []
        if env_outs:
            dims.append(self.env_out_dim)
        dims += self.output_dims
        if env_inps:
            dims.append(self.env_inp_dim)
        dims += self.input_dims

        for space, dim in zip(spaces, dims):
            if space not in sd.spaces:
                raise ValueError(
                    f'Space {space} not in space dictionary {sd}'
                )
            if sd[space] != dim:
                raise ValueError(
                    f'Space dimension sdict[{space}] = {sd[space]} is '
                    f'differnt than apropriate dimension in self ({dim}).'
                )
        
        comb_str = []
        n = len(input_spaces)
        for i in range(n):
            tooth_inp = [input_spaces[i]]
            tooth_out = [output_spaces[i]]
            
            if i == 0:
                tooth_inp = env_inps + tooth_inp
            
            if i == n - 1:
                tooth_out = env_outs + tooth_out
            
            comb_str.append((tooth_inp, tooth_out))
        
        return ParamTensor(
            spaces, self.choi(), sd, dchoi=self.dchoi(),
            output_spaces=env_outs + output_spaces,
            comb_structure=comb_str, **kwargs
        )


    def krauses(self, force_computation: bool = False
        ) -> list[np.ndarray]:
        """
        Computes Kraus operators of the channel.

        If the object was defined using Kraus operators it will
        return the same Kraus operators it was defined with unless
        `force_computation` is true.

        Parameters
        ----------
        force_computation : bool, optional
            Whether to force computation of new Kraus operators.

        Returns
        -------
        krauses : list[np.ndarray]
            List of Kraus operators. Each acting from
            E_i (x) I to E_o (x) O.
        """
        if force_computation or self._krauses is None:
            krauses = krauses_from_choi(
                self.choi(), (self.total_input_dim, self.total_output_dim)
            )
            if not force_computation:
                self._krauses = krauses

        return self._krauses


    def dkrauses(self, force_computation: bool = False) -> tuple[
            list[np.ndarray], list[np.ndarray]
        ]:
        """
        Computes Kraus operatiors of the channel.

        In case the object was defined using Kraus operators it will
        return the same Kraus operators it was defined with unless
        `force_computation` is true.

        Parameters
        ----------
        force_computation : bool, optional
            Whether to force computation of new Kraus operators.

        Returns
        -------
        krauses : list[np.ndarray]
            List of Kraus operators.
        dkrauses : list[np.ndarray]
            List of derivatives of Kraus operators.
        """
        if force_computation or self._dkrauses is None:
            krauses, dkrauses = dkrauses_from_choi(
                self.choi(), self.dchoi(),
                (self.total_input_dim, self.total_output_dim)
            )
            if not force_computation:
                self._krauses = krauses
                self._dkrauses = dkrauses

        return self._krauses, self._dkrauses


    def choi(self) -> np.ndarray:
        """
        Compute Choi matrix of the channel.

        Returns
        -------
        matrix : np.ndarray
            Choi matrix.

        """
        if self._choi is None:
            self._choi = choi_from_krauses(self._krauses)
        return self._choi


    def dchoi(self) -> np.ndarray:
        """
        Compute derivative of the Choi matrix of the channel.

        Returns
        -------
        matrix : np.ndarray
            Derivative of the Choi matrix.

        """
        if self._dchoi is None:
            self._dchoi = dchoi_from_krauses(
                self._krauses, self._dkrauses
            )
        return self._dchoi


    def duplicate(self) -> ParamChannel:
        """
        Create new ParamChannel object representing the same channel
        but with different names of spaces.

        Returns
        -------
        ParamChannel
            New ParamChannel.
        """
        if self.kraus_like_inp:
            krauses, dkrauses = self.dkrauses()
            inp = {'krauses': krauses, 'dkrauses': dkrauses}
        else:
            inp = {
                'choi': self.choi(), 'dchoi': self.dchoi()
            }

        new =  ParamChannel(
            **inp, env_dim=(self.env_inp_dim, self.env_out_dim),
            sdict=self.sdict, input_dim=self.input_dims,
            output_dim=self.output_dims
        )

        if self._tensor is not None:
            old_spaces = self.total_output_spaces + self.total_input_spaces
            new_spaces = new.total_output_spaces + new.total_input_spaces
            duplicate_space = dict(zip(old_spaces, new_spaces))
            new._tensor = self._tensor.respace(
                space_map=lambda s: duplicate_space[s]
            )

        return new


    def merge_spaces(self) -> ParamChannel:
        """
        Merges spaces of the channel to make it single-tooth.

        When input channel is from E_i (x) I_1 (x) ... (x) I_n to
        E_o (x) O_1 (x) ... (x) O_n it will be transformed to a channel
        from E_i (x) I to E_o (x) O, where I = I_1 (x) ... (x) I_n and
        O = O_1 (x) ... (x) O_n are single, high-dimensional spaces.

        Returns
        -------
        ParamChannel
            New single-tooth ParamChannel.
        """
        if self.kraus_like_inp:
            krauses, dkrauses = self.dkrauses()
            return ParamChannel(
                krauses=krauses, dkrauses=dkrauses, sdict=self.sdict,
                env_dim=(self.env_inp_dim, self.env_out_dim)
            )
        
        return ParamChannel(
            choi=self.choi(), dchoi=self.dchoi(), sdict=self.sdict,
            env_dim=(self.env_inp_dim, self.env_out_dim),
            input_dim=self.input_dim, output_dim=self.output_dim
        )


    def _kron_one(self, other: ParamChannel) -> ParamChannel:
        """
        Computes Kronecker product of two elementary channels.

        Both channels A and B must be elementary i.e. they must have one
        input and one output space (single-tooth comb). The corresponding 
        spaces are of A and B are merged, so the resulting channel is from
        E_iAB (x) IAB to E_oAB (x) OAB, AB denotes merged spaces of A and
        B.

        Parameters
        ----------
        other : ParamChannel
            Channel to compute Kronecker product with.

        Returns
        -------
        ParamChannel
            Kronecker product of two channels (single-tooth channel).

        Raises
        ------
        ValueError
            When channels are not single-tooth.
        """
        if not (self.single_tooth and other.single_tooth):
            raise ValueError(
                'Kronecker product is possible only between single-tooth '\
                'channels (each has to have one input and one output)'
            )

        res_env_out = self.env_out_dim * other.env_out_dim
        res_env_inp = self.env_inp_dim * other.env_inp_dim
        res_out_dim = self.output_dim * other.output_dim
        res_inp_dim = self.input_dim * other.input_dim

        if (self.kraus_like_inp and other.kraus_like_inp and
            other.trivial_env):

            krauses, dkrauses = krauses_kron(
                *self.dkrauses(), *other.dkrauses()
            )
            return ParamChannel(  
                krauses=krauses, dkrauses=dkrauses, sdict=self.sdict,  
                env_dim=(res_env_inp, res_env_out)  
            )

        #else calculate the result based on choi
        self_spaces = {
            *self.total_input_spaces, *self.total_output_spaces
        }
        other_spaces = {
            *other.total_input_spaces, *other.total_output_spaces
        }
        if not self_spaces.isdisjoint(other_spaces):
            other = other.duplicate()
        tensor: ParamTensor = self.tensor() * other.tensor()
        spaces = (
            [self.env_out, other.env_out]
            + self.output_spaces + other.output_spaces
            + [self.env_inp, other.env_inp]
            + self.input_spaces + other.input_spaces
        )
        choi = tensor.choi(spaces)
        dchoi = tensor.dchoi(spaces)
        return ParamChannel(
            choi=choi, dchoi=dchoi, sdict=self.sdict,
            input_dim=res_inp_dim,
            output_dim=res_out_dim,
            env_dim=(res_env_inp, res_env_out)
        )


    def kron(self, *others: ParamChannel) -> ParamChannel:
        """
        Computes Kronecker product of channels A (x) B (x) C (x) ...

        Each channel must be elementary i.e. they must have one
        input and one output space (single-tooth comb). The corresponding
        spaces of two channels are merged, so the resulting channel has a 
        single input, single env_input, single output, single env_output.

        Parameters
        ----------
        others[0...*] : ParamChannel
            Channels to compute Kronecker product with.

        Returns
        -------
        ParamChannel
            Kronecker product of channels (single-tooth channel).

        Raises
        ------
        ValueError
            When channels are not single-tooth.
        """
        new = self
        for other in others:
            new = new._kron_one(other)
        return new


    def kron_pow(self, n: int) -> ParamChannel:
        """
        Creates a n-th Kronecker power of channel with trivial environment
        space.

        Parameters
        ----------
        n : int
            Exponent.

        Returns
        -------
        ParamChannel
            n-th Kronecker power

        Raises
        ------
        ValueError
            If channels are not single-tooth
        """
        if n > 1:
            return self.kron(*repeat(self, n - 1))
        if n == 1:
            return self.duplicate()
        raise ValueError(f'Kronecker power is undefined for n = {n}.')


    def _link_env_one(self, other: ParamChannel) -> ParamChannel:
        """
        Computes link product of two channels connecting them by their 
        environment.

        For a product of channels A and B all spaces of A will be before
        spaces of B e.g. A._kron_one(B).input_spaces == A.input_spaces + 
        B.input_spaces

        Link produckt is possible only between channels A and B such
        that `A.env_out_dim == B.env_inp_dim`.

        Parameters
        ----------
        other : ParamChannel
            Channel to compute link product with.

        Returns
        -------
        ParamChannel
            Link product of two channels.

        Raises
        ------
        ValueError
            When the appropriate environment spaces have different
            dimensions.
        """
        if self.env_out_dim != other.env_inp_dim:
            raise ValueError(
                f'self.env_out_dim = {self.env_out_dim} and '\
                f'other.env_inp_dim = {other.env_inp_dim} do not match.'
            )

        self_spaces = {
            *self.total_input_spaces, *self.total_output_spaces
        }
        other_spaces = {
            *other.total_input_spaces, *other.total_output_spaces
        }
        if not self_spaces.isdisjoint(other_spaces):
            other = other.duplicate()

        if (other.trivial_env
            and self.kraus_like_inp and other.kraus_like_inp):

            krauses, dkrauses = krauses_kron(
                *self.dkrauses(), *other.dkrauses()
            )
            return ParamChannel(
                krauses=krauses, dkrauses=dkrauses, sdict=self.sdict,
                input_dim=self.input_dims + other.input_dims,
                output_dim=self.output_dims + other.output_dims
            )

        tensor0 = self.tensor()
        tensor1 = other.tensor().respace(
            space_map=lambda s: s if s != other.env_inp else self.env_out
        )
        tensor = tensor0 * tensor1
        spaces = (
            [other.env_out] + self.output_spaces + other.output_spaces
            + self.total_input_spaces + other.input_spaces
        )
        choi = tensor.choi(spaces)
        dchoi = tensor.dchoi(spaces)
        return ParamChannel(
            choi=choi, dchoi=dchoi, sdict=self.sdict,
            input_dim=self.input_dims + other.input_dims,
            output_dim=self.output_dims + other.output_dims,
            env_dim=(self.env_inp_dim, other.env_out_dim)
        )


    def link_env(self, *others: ParamChannel) -> ParamChannel:
        """
        Computes link product of channels A * B * C * ...  Connecting
        them by their environment.

        For a product of channels A and B all spaces of A will be before
        spaces of B e.g. A._kron_one(B).input_spaces == A.input_spaces + 
        B.input_spaces

        Link produckt is possible only between channels A and B such
        that `A.env_out_dim == B.env_inp_dim`.

        Parameters
        ----------
        others[0...*] : ParamChannel
            Channels to compute link product with.

        Returns
        -------
        ParamChannel
            Link product of channels.

        Raises
        ------
        ValueError
            When the appropriate environment spaces have different
            dimensions.
        """
        new = self
        for other in others:
            new = new._link_env_one(other)
        return new


    def scalar_mul(self, factor: float) -> ParamChannel:
        """
        Multiplies the channel by scalar by multiplying its choi and dchoi.

        Breaks trace preservation, useful to construct convex combinations
        of channels (together with add_channel). 

        Parameters
        ----------
        factor: float
            The factor by which a channel is multiplied
        
        Returns
        -------
        ParamChannel
            Input channel multiplied by a scalar.
        """
        choi_new, dchoi_new = factor * self.choi(), factor * self.dchoi()
        return ParamChannel(
            choi=choi_new, dchoi=dchoi_new,
            env_dim=(self.env_inp_dim, self.env_out_dim), sdict=self.sdict,
            input_dim=self.input_dims, output_dim=self.output_dims
        )


    def __mul__(self, other: ParamChannel | float) -> ParamChannel:
        if isinstance(other, ParamChannel):
            return self.link_env(other)

        return self.scalar_mul(other)


    def __rmul__(self, other: float) -> ParamChannel:
        return self.scalar_mul(other)


    def add(self, other: ParamChannel) -> ParamChannel:
        """
        Adds two ParamChannels by adding their chois and dchois.

        Breaks trace preservation, useful to construct convex combinations
        of channels (together with scalar_mul)

        Parameters
        ----------
        other: ParamChannel
            channel to be added to self
        
        Returns
        -------
        ParamChannel
            Sum of two ParamChannels.
        
        Raises
        ------
        ValueError
            When two channels do not act on spaces of the same dimensions.
        """
        if (self.total_input_dims != other.total_input_dims or
            self.total_output_dims != other.total_output_dims):
            raise ValueError(
                'Two channels must act on spaces of the same dimensions.'
            )

        choi_sum = self.choi() + other.choi()
        dchoi_sum = self.dchoi() + other.dchoi()

        return ParamChannel(
            choi=choi_sum, dchoi=dchoi_sum,
            env_dim=(self.env_inp_dim, self.env_out_dim), sdict=self.sdict,
            input_dim=self.input_dims, output_dim=self.output_dims
        )


    def __add__(self, other: ParamChannel) -> ParamChannel:
        return self.add(other)
    

    def __sub__(self, other: ParamChannel) -> ParamChannel:
        return self.add(other.scalar_mul(-1))


    def _compose_one(self, other: ParamChannel,
        simplify_krauses: bool = True) -> ParamChannel:
        """
        Returns the sequential composition of self and other.

        The returned channel represents a map rho -> self(other(rho)),
        which is a result of concatenation of output of other with input
        of self.

        Parameters
        ----------
        other: ParamChannel
            Channel to be composed with self
        simplify_krauses: bool, optional
            If True, then output is always computed based on chois to
            ensure the minimal possible Kraus represantation of output.
            By default True.

        Returns
        -------
        ParamChannel
            Sequential composition of channels

        Raises
        ------
        ValueError
            When channels are not single-tooth or dimensions of connected
            input and output do not match.
        """

        if not (self.single_tooth and other.single_tooth):
            raise ValueError(
                'Sequential composition works for single-tooth channels '\
                'only'
            )
        if self.input_dim != other.output_dim:
            raise ValueError(
                f'self input dimension {self.input_dim} and other output '\
                f'dimension {other.output_dim} do not match.'
            )
        if self.env_inp_dim != other.env_out_dim:
            raise ValueError(
                f'self environment input dimension {self.env_inp_dim} and'\
                f' other environment output dimension {other.env_out_dim}'\
                'do not match.'
            )
        
        if (self.kraus_like_inp and other.kraus_like_inp and 
            not simplify_krauses):

            krauses12, dkrauses12 = krauses_sequential(
                *self.dkrauses(), *other.dkrauses()
            )
            return ParamChannel(
                krauses=krauses12, dkrauses = dkrauses12,
                env_dim=(other.env_inp_dim, self.env_out_dim),
                sdict=self.sdict
            )
        
        self_spaces = {
            *self.total_input_spaces, *self.total_output_spaces
        }
        other_spaces = {
            *other.total_input_spaces, *other.total_output_spaces
        }
        if not self_spaces.isdisjoint(other_spaces):
            other = other.duplicate()

        def space_map_other(s):
            if s == other.env_out:
                return self.env_inp
            if s == other.output_spaces[0]:
                return self.input_spaces[0]
            return s

        tensor0 = self.tensor()
        tensor1 = other.tensor().respace(space_map=space_map_other)
        tensor = tensor1 * tensor0
        spaces = self.total_output_spaces + other.total_input_spaces

        choi = tensor.choi(spaces)
        dchoi = tensor.dchoi(spaces)
        return ParamChannel(
            choi=choi, dchoi=dchoi, sdict=self.sdict,
            input_dim=other.input_dim, output_dim=self.output_dim,
            env_dim=(other.env_inp_dim, self.env_out_dim)
        )
    
    
    def compose(self, *others: ParamChannel,  
        simplify_krauses: bool = True) -> ParamChannel:
        """
        Returns the sequential composition of self and others.

        The returned channel represents a map 
        rho -> self( others[0]( others[1] ( ... (others[-1]( rho ))...),
        which is a result of concatenation of output of others[i] with
        input of other[i-1] and output of others[0] with input of self.

        Parameters
        ----------
        others[0...*]: ParamChannel
            Channels to be composed with self
        simplify_krauses: bool, optional
            If True, then output is always computed based on chois to
            ensure the minimal possible Kraus represantation of output.
            By default True.

        Returns
        -------
        ParamChannel
            Sequential composition of channels

        Raises
        ------
        ValueError
            When channels are not single-tooth or dimensions of connected
            inputs and outputs do not match.
        """
        new = self
        for other in others:
            new = new._compose_one(other, simplify_krauses)
        return new
    

    def act(self, state: np.ndarray | tuple[np.ndarray, np.ndarray]
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Act on a state with the channel.

        Parameters
        ----------
        state: np.ndarray | tuple[np.ndarray, np.ndarray]
            Input state density matrix (when np.ndarray)
            Input state density matrix and its derivative 
            (when tuple[np.ndarray, np.ndarray])

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Density matrix of the output state and its derivative 
            after action of ParamChannel on input state.
        Raises
        ------
        ValueError
            When state is neither 2d np.ndarray nor tuple of 2 2d
            np.ndarrays
        """
        if isinstance(state, np.ndarray) and state.ndim == 2:
            rho = state
            drho = np.zeros(state.shape)
            correct_input = True
        else:
            try:
                rho, drho = state
                correct_input = rho.ndim == 2 and drho.ndim == 2
            
            except (ValueError, AttributeError, TypeError):
                correct_input = False

        if not correct_input:
            raise ValueError(
                'Input argument state must be either 2d np.array or tuple'\
                'of two 2d np.arrays.'
            )

        if self.kraus_like_inp:
            krauses, dkrauses = self.dkrauses()
            rho_out = np.sum([K@rho@hc(K) for K in krauses], axis = 0)
            drho_out_1 = np.sum([K@drho@hc(K) for K in krauses], axis = 0)
            drho_out_2 = np.sum(
                [
                    dK@rho@hc(K) + K@rho@hc(dK)
                    for K, dK in zip(krauses, dkrauses)
                ],
                axis=0
            )
            drho_out = drho_out_1 + drho_out_2
            return rho_out, drho_out


        state_chan = ParamChannel(  
            choi=rho, dchoi=drho, input_dim=1, 
            env_dim=(1, self.env_inp_dim)  
        )
        state_out = self.compose(state_chan)
        return state_out.choi(), state_out.dchoi()

    
    def __matmul__(self,
        other:ParamChannel | np.ndarray | tuple[np.ndarray, np.ndarray]
        ) -> ParamChannel | tuple[np.ndarray, np.ndarray]:

        if isinstance(other, ParamChannel):
            return self.compose(other)
        return self.act(other)


    def __call__(self, state) -> tuple[np.ndarray, np.ndarray]:
        return self.act(state)


    def trace_env_inp(self, env_inp_state: np.ndarray | None = None
        ) -> ParamChannel:
        """
        Trace environment input space using provided input state.

        Parameters
        ----------
        env_inp_state : np.ndarray | None, optional
            Density matrix of the state. If None then maximally mixed
            state is used. By default None.

        Returns
        -------
        ParamChannel
            New channel with traced out environment input space.

        Raises
        ------
        ValueError
            When input state dimension does not match environment input
            space dimension.
        """
        env_inp_dim = self.env_inp_dim
        if env_inp_state is None:
            env_inp_state = np.identity(env_inp_dim) / env_inp_dim

        if len(env_inp_state.shape) == 1:
            env_inp_state = ket_bra(env_inp_state, env_inp_state)

        if env_inp_state.shape[0] != env_inp_dim:
            raise ValueError(
                f'Provided density matrix {env_inp_state} does not match'\
                f' channel environment dimension {env_inp_dim}.'
            )

        if self.trivial_env_inp:
            return self.duplicate()

        inp_state_ten = ConstTensor(
            [self.env_inp], env_inp_state, self.sdict
        )
        new_tensor = self.tensor() * inp_state_ten

        spaces = self.total_output_spaces + self.input_spaces
        return ParamChannel(
            choi=new_tensor.choi(spaces),
            dchoi=new_tensor.dchoi(spaces), sdict=self.sdict,
            input_dim=self.input_dims, output_dim=self.output_dims,
            env_dim=(1, self.env_out_dim)
        )


    def trace_env_out(self) -> ParamChannel:
        """
        Trace environment output space.

        Returns
        -------
        ParamChannel
            New channel with traced out environment output space.
        """
        if self.trivial_env_out:
            return self.duplicate()

        new_tensor = self.tensor().choi_trace(self.env_out)

        spaces = self.output_spaces + self.total_input_spaces
        return ParamChannel(
            choi=new_tensor.choi(spaces),
            dchoi=new_tensor.dchoi(spaces), sdict=self.sdict,
            input_dim=self.input_dims, output_dim=self.output_dims,
            env_dim=(self.env_inp_dim, 1)
        )


    def trace_env(self, env_inp_state: np.ndarray | None = None
        ) -> ParamChannel:
        """
        Trace environment input space using provided input state and then
        trace environment output space.

        Parameters
        ----------
        env_inp_state : np.ndarray | None, optional
            Density matrix of the state. If None then maximally mixed
            state is used. By default None.

        Returns
        -------
        ParamChannel
            New channel with traced out environment spaces.

        Raises
        ------
        ValueError
            When input state dimension does not match environment input
            space dimension.
        """
        env_inp_dim = self.env_inp_dim
        if env_inp_state is None:
            env_inp_state = np.identity(env_inp_dim) / env_inp_dim

        if len(env_inp_state.shape) == 1:
            env_inp_state = ket_bra(env_inp_state, env_inp_state)

        if env_inp_state.shape[0] != env_inp_dim:
            raise ValueError(
                f'Provided density matrix {env_inp_state} does not match'\
                f' channel environment dimension {env_inp_dim}.'
            )

        if self.trivial_env_inp:
            return self.duplicate()

        inp_state_ten = ConstTensor(
            [self.env_inp], choi=env_inp_state, sdict=self.sdict
        )
        new_tensor = self.tensor() * inp_state_ten
        new_tensor = new_tensor.choi_trace(self.env_out)

        spaces = self.output_spaces + self.input_spaces
        return ParamChannel(
            choi=new_tensor.choi(spaces),
            dchoi=new_tensor.dchoi(spaces), sdict=self.sdict,
            input_dim=self.input_dims, output_dim=self.output_dims
        )


    def markov_series(self: ParamChannel, n: int) -> ParamChannel:
        """
        Computes link product of n channel copies each connected by their
        environment space.

        Roughly equivalent to `self.link_env(*repeat(self, n - 1))`.

        Parameters
        ----------
        n : int
            Number of copies.

        Returns
        -------
        ParamChannel
            Channel representing correlated channels.
        """
        if n > 1:
            return self.link_env(*repeat(self, n - 1))
        if n == 1:
            return self.duplicate()
        raise ValueError(f'Markov series is undefined for n = {n}.')
