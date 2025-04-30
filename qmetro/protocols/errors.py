class EnvDimsError(ValueError):
    def __init__(self, env_inp_dim: int, env_out_dim: int, *args,
        **kwargs):
        self.env_inp_dim = env_inp_dim
        self.env_out_dim = env_out_dim
        msg = f"Channel's environmnent input ({env_inp_dim}) and "\
            f"output ({env_out_dim}) dimensions must be the same."
        super().__init__(msg, *args, **kwargs)


class UnitalDimsError(ValueError):
    def __init__(self, inp_dims: list[int], out_dims: list[int], *args,
        **kwargs):
        self.inp_dims = inp_dims
        self.out_dims = out_dims
        msg = "Comb's teeth can be unital only if channel input "\
            f"dimensions ({inp_dims}) and output ({out_dims}) dimensions"\
            " are the same."
        super().__init__(msg, *args, **kwargs)
