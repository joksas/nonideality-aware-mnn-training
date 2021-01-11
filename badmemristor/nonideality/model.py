import copy
import numpy as np
from badmemristor.nonideality import utils


def lognormal(G, lognormal_G, lognormal_mean, lognormal_sigma, lognormal_rate,
        eff=False):
    """Disturbs conductances by sampling from lognormal distribution.

    This type of modelling was inspired largely by [1]. The main difference is
    that the lognormal parameters are interpolated using linear interpolation
    between neighbouring points, instead of global linear fit.

    [1] Chai, Zheng, et al. "Impact of RTN on pattern recognition accuracy of
    RRAM-based synaptic neural network." IEEE Electron Device Letters 39.11
    (2018): 1652-1655.

    Parameters
    ----------
    G : ndarray
        Conductances (or effective conductances).
    lognormal_G : list of float
        A set of conductances at which mean and sigma parameters, as well as
        occurrence rates, were estimated.
    lognormal_mean : list of float
        A set of mean values of the underlying distribution corresponding to
        states in `lognormal_G`.
    lognormal_sigma : list of float
        A set of standard deviation values of the underlying distribution
        corresponding to states in `lognormal_G`.
    lognormal_rate : list of float
        A set of probabilities (normalized to 1) denoting the likelihood of
        the disturbance occurring for a device in a corresponding resistance
        state from `lognormal_G`.
    eff : bool, optional
        If True, it means that effective conductances have been passed. They,
        instead of conductances, will be disturbed directly by assuming that
        proportional mapping scheme is being used.

    Returns
    -------
    ndarray
        Disturbed conductances (or effective conductances).
    """
    G_disturbed = copy.deepcopy(G)

    length = len(lognormal_G)
    if any(len(lst) != length for lst in [
        lognormal_G, lognormal_mean, lognormal_sigma, lognormal_rate]):
        raise IndexError("Lists `lognormal_G`, `lognormal_mean`, "\
                "`lognormal_sigma` and `lognormal_rate` should all have the "\
                "same length!")

    if eff:
        lognormal_G = utils.symmetric_list(lognormal_G, negative=True)
        lognormal_mean = utils.symmetric_list(lognormal_mean, negative=False)
        lognormal_sigma = utils.symmetric_list(lognormal_sigma, negative=False)
        lognormal_rate = utils.symmetric_list(lognormal_rate, negative=False)

    # linearly interpolate between provided conductances
    means = np.interp(G_disturbed, lognormal_G, lognormal_mean)
    sigmas = np.interp(G_disturbed, lognormal_G, lognormal_sigma)
    rates = np.interp(G_disturbed, lognormal_G, lognormal_rate)

    lognormal_proportions = np.random.lognormal(means, sigmas)
    # determine in which devices the disturbance will occur
    occurrences = np.where(np.random.random_sample(G.shape) <= rates, 1, 0)

    scaling = 1 + occurrences * lognormal_proportions
    G_disturbed = scaling * G_disturbed

    return G_disturbed


def weibull(G, weibull_G, weibull_shape, weibull_scale, eff=False):
    """Disturbs conductances by sampling from Weibull distribution.

    This type of modelling was inspired by similar experimental data as in
    lognormal() function. Similarly, Weibull parameters are interpolated using
    linear interpolation between neighbouring points, instead of global linear
    fit.

    Parameters
    ----------
    G : ndarray
        Conductances (or effective conductances).
    weibull_G : list of float
        A set of conductances at which shape and scale parameters were
        estimated.
    weibull_shape : list of float
        A set of Weibull shape parameters corresponding to states in
        `weibull_G`.
    weibull_scale : list of float
        A set of Weibull scale parameters corresponding to states in
        `weibull_G`.
    eff : bool, optional
        If True, it means that effective conductances have been passed. They,
        instead of conductances, will be disturbed directly by assuming that
        proportional mapping scheme is being used.

    Returns
    -------
    ndarray
        Disturbed conductances (or effective conductances).
    """
    G_disturbed = copy.deepcopy(G)

    length = len(weibull_G)
    if any(len(lst) != length for lst in [
        weibull_G, weibull_shape, weibull_scale]):
        raise IndexError("Lists `weibull_G`, `weibull_shape` and "\
                "`weibull_scale` should all have the same length!")

    if eff:
        weibull_G = utils.symmetric_list(weibull_G, negative=True)
        weibull_shape = utils.symmetric_list(weibull_shape, negative=False)
        weibull_scale = utils.symmetric_list(weibull_scale, negative=False)

    # linearly interpolate between provided conductances
    shape_pars = np.interp(G_disturbed, weibull_G, weibull_shape)
    scale_pars = np.interp(G_disturbed, weibull_G, weibull_scale)

    weibull_proportions = scale_pars * np.random.weibull(shape_pars)

    # disturb in random direction
    signs = 2*np.random.randint(2, size=G_disturbed.shape) - 1
    # prevent conductances from becoming non-positive
    signs = np.where(weibull_proportions >= 1, 1, signs)

    scaling = 1 + signs * weibull_proportions
    G_disturbed = scaling * G_disturbed

    return G_disturbed

