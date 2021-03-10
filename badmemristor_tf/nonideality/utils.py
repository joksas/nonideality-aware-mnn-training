import numpy as np


def symmetric_list(lst, negative=False):
    """Returns a list that contains a reflected copy of the original and the
    original.

    For example, if the original list is `[1, 2, 4]`, the function would return
    `[4, 2, 1, 1, 2, 4]`. If `negative` is True, then `[-4, -2, -1, 1, 2, 4]`
    would be returned.

    Parameters
    ----------
    lst : list of (float or int)
        List of numbers.
    negative : bool, optional
        Whether the reflected copy should be negative.

    Returns
    ----------
    symmetric_lst : list of (float or int)
        Symmetric list.
    """
    reflected_lst = lst.copy()
    reflected_lst.reverse()
    if negative:
        reflected_lst = [-i for i in reflected_lst]
    symmetric_lst = reflected_lst + lst
    return symmetric_lst


def symmetric_array(arr, negative=False):
    """Returns an ndarray that contains a reflected copy of the original and
    the original.

    For example, if the original array is `[1, 2, 4]`, the function would
    return `[4, 2, 1, 1, 2, 4]`. If `negative` is True, then `[-4, -2, -1, 1,
    2, 4]` would be returned.

    Parameters
    ----------
    arr : ndarray
        Array of numbers.
    negative : bool, optional
        Whether the reflected copy should be negative.

    Returns
    ----------
    symmetric_arr : ndarray
        Symmetric array.
    """
    reflected_arr = arr.copy()
    reflected_arr = reflected_arr[::-1]
    if negative:
        reflected_arr = -reflected_arr
    symmetric_arr = np.concatenate((reflected_arr, arr))
    return symmetric_arr

