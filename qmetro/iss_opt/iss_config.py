from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from ..qmtensor import TensorNetwork
from ..qmtensor.operations import is_var

from .consts import PARTIAL




@dataclass
class IssConfig:
    """
    Configuration parameters for the ISS optimization algorithm.

    Parameters
    ----------
    name : str | None, optional
        Name of the returned network. If None then the name will be
        `tn.name` + ' optimized', by default None.
    max_error_iterations : int, optional
        Maximal number of times the function will restart
        computation after an error occurence, by default 10.
    max_iterations : int, optional
        Maximal number of algorithm iterations, by default 500.
    min_iterations : int, optional
        Minimal number of algorithm iterations, by default 10.
    eps : float, optional
        The algorithm stops when the QFI in 5 consecutive iterations
        changes by less than eps, by default 1e-4.
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
        Number of iterations done solely over non measurement variables
        (CPTP, MPS and combs) before proceeding to the measurement
        (pre-SLD) variables, by default 1.
    sld_iterations : int, optional
        Number of iterations done solely over measurement (pre-SLD)
        variables before proceeding to other variables, by default 1.
    art_noise_spaces : list[list[Hashable]] | None, optional
        Spaces on which the artificial noise
        will act. For a list: `[[s00, s01, ...], [s10, s11, ...], ...]`
        noise will be applied to the whole list of spaces `[s00, s01, ...]`
        combined. If set to None then only spaces of ParamTensor elements
        will be noisy (separately), by default None.
    art_noise_params : tuple[float, float], optional
        For a tupple `(a, l)` noise will take a form of:

        rho -> p * rho + (1 - p) * Id,
        
        where p = 1 - a * exp(-l * i), i is the iteration number and Id is
        theidentity matrix, by default (0.5, 0.1).
        See also :ref:`the documentation <depolarization>`.
    contraction_order : list[str] | None, optional
        Names of tensors in the required order of contraction and
        therfore also the order of optimization. If None then the program
        will use BFS starting from the variable with smallest dimension
        and without input spaces, by default None.
    adaptive_art_noise : bool, optional
        If True then the program will track the increase of QFI coming
        from variable optimization and from the artificial noise decay.
        If the latter is bigger the artificial noise decay parameter, `l`,
        will be doubled increasing the decay speed. By default True.
    """
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
        """
        Check correctness of the parameters of the ISS algorithm.

        Parameters
        ----------
        tn : TensorNetwork
            Tensor network to be optimized.
        
        Raises
        ------
        ValueError
            If the parameters are not correct.
        """
        self.check_art_noise_params()
        self.check_contraction_order(tn)
        self.check_cptp_vars(tn)


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
        
    
    def check_cptp_vars(self, tn: TensorNetwork):
        for name, tensor in tn.tensors.items():
            if (
                is_var(tensor) and tensor.bond_spaces
                and tensor.input_spaces and tensor.output_spaces
            ):
                raise ValueError(
                    'The tensor network contains CPTP variable with bond '\
                    f'spaces ({name}). '
                )


    @property
    def print_full(self) -> bool:
        if isinstance(self.print_messages, str):
            return self.print_messages != PARTIAL
        return self.print_messages


    @property
    def print_partial(self) -> bool:
        return self.print_messages == PARTIAL
