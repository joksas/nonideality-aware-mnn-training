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

