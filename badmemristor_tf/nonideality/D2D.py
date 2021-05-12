import numpy as np
import copy


def faulty(G, proportion, G_min=None, G_max=None, type_='unelectroformed',
        eff=False):
    """Sets a proportion of random conductances to a certain state.

    Parameters
    ----------
    G : ndarray
        Conductances (or effective conductances).
    G_min : float, optional
        Minimum conductance of electroformed memristors.
    G_max : float, optional
        Maximum conductance of electroformed memristors.
    proportion : float, optional
        Proportion (normalized to 1) of conductances that are set to a certain
        state.
    type_ : {'unelectroformed', 'stuck_at_G_min', 'stuck_at_G_max'}, optional
        The state in which the proportion of the devices gets stuck. If type_
        is 'stuck_at_G_min' or 'stuck_at_G_max', then a fraction of
        *electroformed* devices is set to those conductances.
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
    if type_ == 'unelectroformed':
        num_changed = int(np.floor(proportion*G.size))
        device_indices = np.random.choice(
            G_disturbed.size, size=num_changed, replace=False)
        device_indices = np.unravel_index(device_indices, G_disturbed.shape)
        G_disturbed[device_indices] = 0
    else:
        if type_ == 'stuck_at_G_min':
            if G_min is None:
                raise ValueError("If `type_ == stuck_at_G_min`, then `G_min` " \
                "must be passed!")
            G_x = G_min
        elif type_ == 'stuck_at_G_max':
            if G_max is None:
                raise ValueError("If `type_ == stuck_at_G_max`, then `G_max` " \
                "must be passed!")
            G_x = G_max

        x, y = np.nonzero(G_disturbed)
        num_changed = int(np.floor(proportion*len(x)))
        indices = np.random.choice(len(x), size=num_changed, replace=False)
        if eff:
            G_disturbed[x[indices], y[indices]] = np.where(
                    G_disturbed[x[indices], y[indices]] > 0, G_x, -G_x)
        else:
            G_disturbed[x[indices], y[indices]] = G_x

    return G_disturbed
