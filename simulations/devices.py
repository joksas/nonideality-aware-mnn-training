from training.iterator import (D2DLognormal, IVNonlinearity, StuckAtGMax,
                               StuckAtGMin)


def ideal():
    return {"G_min": None, "G_max": None, "nonidealities": {}}


def low_R():
    return {
        **low_R_conductance(),
        "nonidealities": {"iv_nonlinearity": IVNonlinearity(2.132, 0.095)},
    }


def high_R():
    return {
        **high_R_conductance(),
        "nonidealities": {"iv_nonlinearity": IVNonlinearity(2.989, 0.369)},
    }


def stuck_low():
    return {
        **high_R_conductance(),
        "nonidealities": {
            "stuck_at_G_min": StuckAtGMin(0.05),
        },
    }


def high_R_and_stuck():
    return {
        **high_R_conductance(),
        "nonidealities": {
            "iv_nonlinearity": IVNonlinearity(2.989, 0.369),
            "stuck_at_G_max": StuckAtGMax(0.05),
        },
    }


def low_R_conductance():
    return {
        "G_min": 1 / 1003,
        "G_max": 1 / 284.6,
    }


def high_R_conductance():
    return {
        "G_min": 1 / 1295000,
        "G_max": 1 / 366200,
    }


def symmetric_d2d():
    return {
        **low_R_conductance(),
        "nonidealities": {"d2d_lognormal": D2DLognormal(0.25, 0.25)},
    }


def asymmetric_d2d():
    return {
        **low_R_conductance(),
        "nonidealities": {"d2d_lognormal": D2DLognormal(0.05, 0.5)},
    }


def symmetric_high_d2d():
    return {
        **low_R_conductance(),
        "nonidealities": {"d2d_lognormal": D2DLognormal(0.5, 0.5)},
    }
