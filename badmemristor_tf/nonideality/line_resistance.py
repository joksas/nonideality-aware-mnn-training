import numpy as np
import copy
#import badcrossbar


def G_to_R(G):
    """Computes resistance values from conductance values.

    Parameters
    ----------
    G : ndarray
        Conductances.

    Returns
    -------
    R : ndarray
        Resistances.
    """
    with np.errstate(divide='ignore'):
        R = 1./G

    return R


def IA_reordering(V, R, ref_V):
    """Reorder voltages and resistances using intensity-aware reordering
    scheme.

    Parameters
    ----------
    V : ndarray
        Voltages to be applied. Each column corresponds to a different example.
    R : ndarray
        Resistances.
    ref_V : ndarray
        Reference voltages for predicting the nature of V.

    Returns
    -------
    tuple of ndarray
        Reordered inputs and resistances so that rows with highest expected
        average intensity (of the voltages) would be at the bottom.
    """
    new_V = copy.deepcopy(V)
    new_R = copy.deepcopy(R)

    # indices of sorted average values
    indices = np.argsort(np.mean(ref_V, axis=1))

    new_V[:, ] = new_V[indices, ]
    new_R[:, ] = new_R[indices, ]

    return new_V, new_R


def crossbar_config(num_WL, num_BL, num_inputs, num_outputs):
    """Extracts configuration of the crossbars that would accommodate all
    resistances.

    Parameters
    ----------
    num_WL : int
        Number of word lines.
    num_BL : int
        Number of bit lines.
    num_inputs : int
        Number of inputs (usually number of rows in resistances matrix).
    num_outputs : int
        Number of outputs (usually number of columns in resistances matrix).

    Returns
    -------
    num_vert : int
        Number of crossbars needed to fit the first dimension of resistances
        array.
    num_horiz : int
        Number of crossbars needed to fit the second dimension of resistances
        array. Total number of crossbar required is num_vert*num_horiz.
    """
    num_vert = int(np.ceil(num_inputs/num_WL))
    num_horiz = int(np.ceil(num_outputs/num_BL))
    return num_vert, num_horiz


def map_V_and_R(V, R, num_WL, num_BL, IA=False, ref_V=None):
    """Maps voltages and resistances onto multiple crossbars.

    ***Example***

    Suppose that we have resistances R and voltages V:

        ⌜ 11 12 13 14 15 16 17 ⌝
        | 21 22 23 24 25 26 27 |
    R = | 31 32 33 34 35 36 37 |  V =  ⌜ 1.1 1.2 1.3 1.4 1.5 ⌝
        | 41 42 43 44 45 46 47 |       ⌞ 2.1 2.2 2.3 2.4 2.5 ⌟
        ⌞ 51 52 53 54 55 56 57 ⌟

    We must allocate them on crossbars of size 3 x 2. Using this function
    (without IA reordering), it would be done in the following way:

            ⌜ 11 15 ⌝          ⌜ 12 16 ⌝          ⌜ 13 17 ⌝          ⌜ 14 ∞ ⌝
    R_0_0 = | 31 35 |  R_0_1 = | 32 36 |  R_0_2 = | 33 37 |  R_0_3 = | 34 ∞ |
            ⌞ 51 55 ⌟          ⌞ 52 56 ⌟          ⌞ 53 57 ⌟          ⌞ 54 ∞ ⌟

                                                       ⌜ 1.1 2.1 ⌝
    each of which would be applied with voltages V_0 = | 1.3 2.3 |
                                                       ⌞ 1.5 2.5 ⌟

            ⌜  ∞  ∞ ⌝          ⌜  ∞  ∞ ⌝          ⌜  ∞  ∞ ⌝          ⌜  ∞ ∞ ⌝
    R_1_0 = | 21 25 |  R_1_1 = | 22 26 |  R_1_2 = | 23 27 |  R_1_3 = | 24 ∞ |
            ⌞ 31 45 ⌟          ⌞ 42 46 ⌟          ⌞ 43 47 ⌟          ⌞ 44 ∞ ⌟

                                                       ⌜   0   0 ⌝
    each of which would be applied with voltages V_1 = | 1.2 2.2 |
                                                       ⌞ 1.4 2.4 ⌟

    The function would return:
    * V_list = [V_0, V_1]
    * R_list = [[R_0_0, R_0_1, R_0_2, R_0_3], [R_1_0, R_1_1, R_1_2, R_1_3]]

    ***

    Parameters
    ----------
    V : ndarray
        Voltages to be applied.
    R : ndarray
        Resistances.
    num_WL : int
        Number of word lines.
    num_BL : int
        Number of bit lines.
    IA : bool, optional
        If intensity-aware mapping should be used.
    ref_V : ndarray, optional
        Reference voltages for predicting the nature of V when intensity-aware
        mapping is used.

    Returns
    -------
    V_list : list of ndarray
        Voltages to be applied. Each row corresponds to a different example.
    R_list : list of list of ndarray
        Resistances in each crossbar array.
    """
    V = np.transpose(V)
    if IA:
        ref_V = np.transpose(ref_V)
        V, R = IA_reordering(V, R, ref_V)

    num_vert, num_horiz = crossbar_config(num_WL, num_BL, *R.shape)
    R_list = [[np.inf*np.ones((num_WL, num_BL)) for j in range(num_horiz)] for i in range(num_vert)]
    V_list = [np.zeros((num_WL, V.shape[1])) for i in range(num_vert)]

    for i in range(num_vert):
        R_sub  = R[i::num_vert, ]
        V_sub = V[i::num_vert, ]
        num_used_WL = R_sub.shape[0]
        V_list[i][-num_used_WL:, :] = V_sub
        for j in range(num_horiz):
            R_sub_sub = R_sub[:, j::num_horiz]
            num_used_BL = R_sub_sub.shape[1]
            R_list[i][j][-num_used_WL:, :num_used_BL] = R_sub_sub

    return V_list, R_list


def crossbar_output_I(V, R, R_WL, R_BL, verbose):
    """Computes output currents of a crossbar array.

    Parameters
    ----------
    V : ndarray
        Voltages of shape `m x p` applied to the rows of the crossbar array.
        Each column corresponds to a different example.
    R : ndarray
        Resistances of shape `m x 2n` of the crossbar array.
    R_WL : float
        Interconnect resistance of the word line segments.
    R_BL : float
        Interconnect resistance of the bit line segments.
    verbose : {1, 2, 0}
        If 1, all badcrossbar messages are shown. If 2, only warnings are
        shown. If 0, no messages are shown.

    Returns
    -------
    output_I : ndarray
        Output currents of shape `p x 2n`.
    """
    output_I = badcrossbar.compute(
            V, R, r_i_word_line=R_WL, r_i_bit_line=R_BL,
            all_currents=False, node_voltages=False,
            verbose=verbose).currents.output
    return output_I


def crossbar_output_Is(V_list, R_list, R_WL, R_BL, verbose):
    """Computes output currents for an array of crossbar arrays.

    Parameters
    ----------
    V_list : list of ndarray
        Voltages to be applied. Each row corresponds to a different example.
    R_list : list of list of ndarray
        Resistances in each crossbar array.
    R_WL : float
        Interconnect resistance of the word line segments.
    R_BL : float
        Interconnect resistance of the bit line segments.
    verbose : {1, 2, 0}
        If 1, all messages are shown. If 2, only warnings are shown. If
        0, no messages are shown.


    Returns
    -------
    I_list : list of list of ndarray
        Output currents for each crossbar array.
    """
    num_vert = len(R_list)
    num_horiz = len(R_list[0])
    num_WL, num_BL = R_list[0][0].shape
    I_list = [[np.zeros((num_WL, num_BL)) for j in range(num_horiz)] for i in range(num_vert)]
    for i, (crossbar_V, crossbar_row_R) in enumerate(zip(V_list, R_list)):
        for j, crossbar_R in enumerate(crossbar_row_R):
            I_list[i][j] = crossbar_output_I(
                    crossbar_V, crossbar_R, R_WL, R_BL, verbose)

    return I_list


def output_Is_to_I(I_list, num_outputs):
    """Deshuffles currents from multiple crossbars.

    This produces currents that would have been obtained if a single crossbar
    of shape (num_inputs, num_outputs) had been used.

    Parameters
    ----------
    I_list : list of list of ndarray
        Output currents for each crossbar array.
    num_outputs : int
        Number of outputs (usually number of columns in resistances matrix).

    Returns
    -------
    I : ndarray
        Output currents.
    """
    num_examples = I_list[0][0].shape[0]
    num_BL = I_list[0][0].shape[1]
    num_horiz = len(I_list[0])
    I = np.zeros((num_examples, num_outputs))
    for crossbar_I_row in I_list:
        for idx, crossbar_I in enumerate(crossbar_I_row):
            # We will change `I` by reference.
            I_sub = I[:, idx::num_horiz]
            I_sub += crossbar_I[:, :I_sub.shape[1]]

    return I


def compute_I(V, G, num_WL, num_BL, R_WL, R_BL, IA=False, ref_V=None, verbose=1):
    """Computes output currents of a dot-product engine.

    Parameters
    ----------
    V : ndarray
        Voltages of shape (p x m) to be applied. Each column corresponds to a
        different example.
    G : ndarray
        Conductances of shape (m x 2n).
    num_WL : int
        Number of word lines.
    num_BL : int
        Number of bit lines.
    R_WL : float
        Interconnect resistance of the word line segments.
    R_BL : float
        Interconnect resistance of the bit line segments.
    IA : bool, optional
        If intensity-aware mapping should be used.
    ref_V : ndarray, optional
        Reference voltages for predicting the nature of V when intensity-aware
        mapping is used.
    verbose : {1, 2, 0}, optional
        If 1, all badcrossbar messages are shown. If 2, only warnings are
        shown. If 0, no messages are shown.

    Returns
    -------
    I : ndarray
        Output currents of shape (p x 2n).
    """
    R = G_to_R(G)
    V_list, R_list = map_V_and_R(V, R, num_WL, num_BL, IA=IA, ref_V=ref_V)
    I_list = crossbar_output_Is(V_list, R_list, R_WL, R_BL, verbose)
    I = output_Is_to_I(I_list, R.shape[1])
    return I

