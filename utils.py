from functools import partial


def make_partial_func(func,
                      window=None,  # time series operators
                      n=None,  # time series operators
                      a=None,  # time series operators
                      constant=None,  # cross sectional operators
                      mode=None,  # cross sectional operators、time series operators
                      alpha=None,  # cross sectional operators
                      rank=None  # cross sectional operators
                      ):
    """Add auxiliary functions for various operators to assist in registering function names"""
    ret_func = func
    name_parts = [func.__name__]

    if window is not None:
        ret_func = partial(ret_func, window=window)
        name_parts.append("window_" + str(window))

    if n is not None:
        ret_func = partial(ret_func, n=n)
        name_parts.append("n_" + str(n))

    if a is not None:
        ret_func = partial(ret_func, a=a)
        name_parts.append("a_" + str(a).replace('.', "__"))

    if mode is not None:
        ret_func = partial(ret_func, mode=mode)
        name_parts.append("mode_" + str(mode))

    if constant is not None:
        ret_func = partial(ret_func, constant=constant)
        name_parts.append("constant_" + str(constant).replace('.', "__").replace("-", "___"))

    if alpha is not None:
        ret_func = partial(ret_func, alpha=alpha)
        name_parts.append("alpha_" + str(alpha).replace('.', "__").replace("-", "___"))

    if rank is not None:
        ret_func = partial(ret_func, rank=rank)
        name_parts.append("rank_" + str(rank))

    ret_func.__name__ = "_".join(name_parts)
    return ret_func
