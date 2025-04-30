def non_herimitan_message(delta: float) -> str:
    return 'Non-hermitian parameter encountered (non-hermiticity: '\
        f'{delta}). Proceeding with artificially enhanced parameter.'


def parameter_norm_message(n: int) -> str:
    return f'Solver failed for normalization factor: 2**{n}. '\
        f'Proceeding with 2**{n+1}.'
