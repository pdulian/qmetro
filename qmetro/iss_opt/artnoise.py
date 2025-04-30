from __future__ import annotations

from collections.abc import Hashable

import numpy as np

from ..qmtensor import (
    ConstTensor, TensorNetwork, SpaceDict,
    const_tensor_from_fun,
    Tensor
)
from ..qmtensor.operations import is_param




class ArtNoise:
    def __init__(self, space_packs: list[list[Hashable]] | None,
        params: tuple[float, float], tn: TensorNetwork, sdict: SpaceDict):

        if space_packs is None:
            _spaces = set()
            for tensor in tn.tensors.values():
                if is_param(tensor):
                    _spaces.update(
                        s for s in tensor.spaces
                        if s not in sdict.bond_spaces
                    )
            space_packs = [[space] for space in _spaces]
        else:
            occured = set()
            for space_pack in space_packs:
                for space in space_pack:
                    if space in occured:
                        raise ValueError(
                            f'Space: {space} appears more than once in'\
                            ' art_noise_spaces.'
                        )
                    occured.add(space)

        self.space_packs = space_packs
        self.spaces = [s for pack in self.space_packs for s in pack]

        self.a = params[0]
        self.l = params[1]

        if self.a <= 0 or self.a > 1:
            raise ValueError(
                'First parameter of artificial noise must be in ]0, 1]'\
                f' but {self.a} was given.'
            )
        if self.l <= 0:
            raise ValueError(
                'Second parameter of artificial noise must be positive'\
                f' but {self.l} was given.'
            )

        self.sdict = sdict

        self.prime: str = 'prime' #uuid.uuid4().hex
        self.primed_spaces: dict[tuple[Hashable, str], Hashable] = {}
        for space in self.spaces:
            self.sdict[self.primed(space)] = self.sdict[space]
            self.primed_spaces[self.primed(space)] = space

        self.tensor_info: dict[
            str, tuple[list[Hashable], tuple[int, ...]]
        ] = {}
        self.dim_types: dict[
            tuple[int, ...], tuple[int, ConstTensor]
        ] = {}
        self.current_it = -1
        self.prefix = f'ARTIFICIAL NOISE {self.prime} '
        for pack in space_packs:
            name = self.prefix + str(
                sorted(pack, key=lambda s: self.sdict[s])
            )
            dim_type = self.get_dim_type(pack)
            self.tensor_info[name] = pack, dim_type

        self.decay_step = 1


    def primed(self, space: Hashable) -> tuple[Hashable, str]:
        """
        Returns a tuple (space, 'prime')
        """
        return space, self.prime


    def unprimed(self, space: Hashable) -> Hashable:
        """
        Performs:
            (space, 'prime') -> space
            space -> space
        """
        if space in self.primed_spaces:
            return self.primed_spaces[space]
        return space


    def new_tensors(self, tn: TensorNetwork) -> list[Tensor]:
        """
        Creates a list of tensors from tn with added artificial noise.
        """
        to_cut = set()
        for space, space_tensors in tn.edges.items():
            if len(space_tensors) < 2 or space not in self.spaces:
                continue
            name0, _ = space_tensors
            to_cut.add((name0, space))

        tensors: list[Tensor] = []
        for name, tensor in tn.tensors.items():
            def space_map(space):
                if (name, space) in to_cut:
                    return self.primed(space)
                return space

            new_tensor = tensor.respace(
                space_map=space_map, sdict=self.sdict
            )
            new_tensor.name = name
            tensors.append(new_tensor)

        # Add tensors of artifitial noise
        self.current_it = -1
        for name, (pack, dim_type) in self.tensor_info.items():
            pack_ = list(map(self.primed, pack))
            ct = ConstTensor(
                pack + pack_, sdict=self.sdict, name=name,
                output_spaces=pack_
            )
            self.dim_types[dim_type] = self.current_it, ct
            tensors.append(ct)

        return tensors


    def get_dim_type(self, space_pack: list[Hashable]) -> tuple[int, ...]:
        return tuple(sorted(self.sdict[s] for s in space_pack))


    def channel(self, i: int, rho: np.ndarray) -> np.ndarray:
        p = 1 - self.a * np.exp(-self.l * i)
        d = len(rho)
        return p * rho + (1-p) * np.trace(rho) * np.identity(d)/d


    def update(self, iteration: int, _qn: TensorNetwork):
        max_dim = max(self.sdict.spaces.values())
        def primed_second(space):
            dim = self.sdict[space]
            if space in self.primed_spaces:
                return dim + max_dim
            return dim

        for name, (_, dim_type) in self.tensor_info.items():
            tensor = _qn.tensors[name]
            i, tensor_ = self.dim_types[dim_type]
            spaces = sorted(tensor.spaces, key=primed_second)
            tensor.reorder(spaces)
            if i == iteration:
                tensor_.reorder(sorted(tensor_.spaces, key=primed_second))
                tensor.array = tensor_.array
            else:
                tensor.array = const_tensor_from_fun(
                    lambda rho: self.channel(iteration, rho),
                    spaces[:len(spaces) // 2], spaces[len(spaces) // 2:],
                    self.sdict
                ).array
                self.dim_types[dim_type] = iteration, tensor
