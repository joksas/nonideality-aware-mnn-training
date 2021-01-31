from scipy import interpolate
import numpy as np
import copy
from badmemristor.nonideality import utils


def compute_I(V, G, V_ref, G_ref, I_ref, eff=False):
    """Computes output currents of a crossbar consisting of devices suffering
    from I/V non-linearities.

    Parameters
    ----------
    V : ndarray
        Voltages of shape (m x p).
    G : ndarray
        Conductances (or effective conductances) of shape (m x n).
    V_ref :
        Reference voltage values of length r.
    G_ref : ndarray
        Reference conductance values of length q.
    I_ref :
        Reference current values of shape (q x r) corresponding go G_ref and
        V_ref.
    eff : bool, optional
        If True, it means that effective conductances have been passed.

    Returns
    ----------
    I : ndarray
        Output currents of shape (p x n). If eff is True, then *total* output
        currents are returned.
    """
    I_ind = interpolate_I(G_ref, V_ref, I_ref, G, V, eff)
    I = add_I_BL(I_ind)
    return I


def interpolate_I(G_ref, V_ref, I_ref, G, V, eff):
    """Interpolates current values.

    Parameters
    ----------
    G_ref : ndarray
        Reference conductance values of length q.
    V_ref :
        Reference voltage values of length r.
    I_ref :
        Reference voltage values of shape (q x r) corresponding go G_ref and
        V_ref.
    G : ndarray
        Conductances (or effective conductances) of shape (m x n).
    V : ndarray
        Voltages of shape (m x p).
    eff : bool
        If True, it means that effective conductances have been passed.

    Returns
    ----------
    I : ndarray
        Interpolated currents of shape (p x m x n) produced by each of the
        conductances in the crossbar array.
    """
    if eff:
        G_ref = utils.symmetric_array(G_ref, negative=True)
        I_ref = np.concatenate((-I_ref[::-1, :], I_ref), axis=0)

    f = interpolate.interp2d(V_ref, G_ref, I_ref, kind='linear')

    num_WL = G.shape[0]

    I_seq = []
    for i_WL in range(num_WL):
        G_slice = G[i_WL, :]
        V_slice = V[:, i_WL]

        # For some reason, scipy interpolation returns results in increasing
        # order of the arguments. Therefore, we must make a note of the order.
        G_slice_idx = np.argsort(G_slice)
        V_slice_idx = np.argsort(V_slice)

        # Interpolation.
        I_slice_wrong = f(V_slice, G_slice)

        # Restoring original order.
        I_slice_wrong_V = copy.deepcopy(I_slice_wrong)
        I_slice_wrong_V[G_slice_idx, :] = I_slice_wrong
        I_slice = copy.deepcopy(I_slice_wrong_V)
        I_slice[:, V_slice_idx] = I_slice_wrong_V

        I_seq.append(I_slice)

    I = np.stack(I_seq, axis=2)
    I = np.swapaxes(I, 0, 2)
    I = np.swapaxes(I, 0, 1)

    return I


def add_I_BL(I_ind):
    """Adds currents along the bit lines.

    Parameters
    ----------
    I_ind : ndarray
        Currents of shape (p x m x n) produced by each of the conductances in
        the crossbar array.

    Returns
    ----------
    I : ndarray
        Output currents of shape (p x n).
    """
    I = np.sum(I_ind, axis=1)
    return I

