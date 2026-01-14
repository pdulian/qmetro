class EnvDimsError(ValueError):
    """
    Raised when channel's environment input and output dimensions
    are not the same.

    Parameters
    ----------
    env_inp_dim : int
        Environment input dimension.
    env_out_dim : int
        Environment output dimension.
    """    
    def __init__(self, env_inp_dim: int, env_out_dim: int, *args,
        **kwargs):
        self.env_inp_dim = env_inp_dim
        self.env_out_dim = env_out_dim
        msg = f"Channel's environmnent input ({env_inp_dim}) and "\
            f"output ({env_out_dim}) dimensions must be the same."
        super().__init__(msg, *args, **kwargs)


class UnitalDimsError(ValueError):
    """
    Raised when comb's teeth are unital but channel input and output
    dimensions are not the same.

    Parameters
    ----------
    inp_dims : list[int]
        Input dimensions.
    out_dims : list[int]
        Output dimensions.
    """
    def __init__(self, inp_dims: list[int], out_dims: list[int], *args,
        **kwargs):
        self.inp_dims = inp_dims
        self.out_dims = out_dims
        msg = "Comb's teeth can be unital only if channel input "\
            f"dimensions ({inp_dims}) and output ({out_dims}) dimensions"\
            " are the same."
        super().__init__(msg, *args, **kwargs)
