from crossbar.nonidealities import (D2DLognormal, IVNonlinearity, StuckAt,
                                    StuckDistribution)

from . import utils


def ideal():
    return {"G_min": None, "G_max": None, "nonidealities": []}


def low_R():
    return {
        **low_R_conductance(),
        "nonidealities": [IVNonlinearity(2.132, 0.095)],
    }


def high_R():
    return {
        **high_R_conductance(),
        "nonidealities": [IVNonlinearity(2.989, 0.369)],
    }


def stuck_low():
    return {
        **high_R_conductance(),
        "nonidealities": [
            StuckAt(high_R_conductance()["G_min"], 0.05),
        ],
    }


def high_R_and_stuck():
    return {
        **high_R_conductance(),
        "nonidealities": [
            IVNonlinearity(2.989, 0.369),
            StuckAt(high_R_conductance()["G_max"], 0.05),
        ],
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
        "nonidealities": [D2DLognormal(0.25, 0.25)],
    }


def asymmetric_d2d():
    return {
        **low_R_conductance(),
        "nonidealities": [D2DLognormal(0.05, 0.5)],
    }


def symmetric_high_d2d():
    return {
        **low_R_conductance(),
        "nonidealities": [D2DLognormal(0.5, 0.5)],
    }


def HfO2():
    data = utils.load_cycling_data("G_reads_11_cycles.mat")
    G_min, G_max = utils.extract_G_min_and_G_max(data)
    vals, p = utils.extract_stuck(data, G_min, G_max)
    return {
        "G_min": G_min,
        "G_max": G_max,
        "nonidealities": [StuckDistribution(vals, p)],
    }
