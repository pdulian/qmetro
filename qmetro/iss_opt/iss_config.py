from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from ..qmtensor import TensorNetwork

from .consts import PARTIAL




@dataclass
class IssConfig:
    name: str | None
    max_error_iterations: int
    max_iterations: int
    min_iterations: int
    eps: float
    init_tn: TensorNetwork | None
    print_messages: bool | str
    var_iterations: int
    sld_iterations: int
    art_noise_spaces: list[list[Hashable]] | None
    art_noise_params: tuple[float, float]
    contraction_order: list[str] | None
    adaptive_art_noise: bool


    def check(self, tn: TensorNetwork):
        self.check_art_noise_params()
        self.check_contraction_order(tn)


    def check_art_noise_params(self):
        a, l = self.art_noise_params

        if not 0 < a < 1:
            raise ValueError(
                'First element of art_noise_params argument must be a '\
                f'number in the open interval ]0, 1[ but {a} was '\
                'provided.'
            )

        if not 0 < l:
            raise ValueError(
                'Second element of art_noise_params argument must be a '\
                f'positive number but {l} was provided.'
            )


    def check_contraction_order(self, tn: TensorNetwork):
        if self.contraction_order is None:
            return

        contr_order_set = set(self.contraction_order)
        tn_tensors_set = set(tn.tensors.keys())
        if contr_order_set != tn_tensors_set:
            not_in_contr = tn_tensors_set.difference(contr_order_set)
            if not_in_contr:
                raise ValueError(
                    'contraction_order must contain names of all tensors'\
                    ' present in the tensor netork tn. The following '\
                    f'tensors were not present: {not_in_contr}.'
                )

            not_in_tn = contr_order_set.difference(tn_tensors_set)
            raise ValueError(
                'contraction_order must contain only names of the '\
                'tensors that are in tn. The following elements are not '\
                f'names of tn tensors: {not_in_tn}.'
            )


    @property
    def print_full(self) -> bool:
        if isinstance(self.print_messages, str):
            return self.print_messages != PARTIAL
        return self.print_messages


    @property
    def print_partial(self) -> bool:
        return self.print_messages == PARTIAL
