from awarememristor.crossbar.nonidealities import (D2DLognormal,
                                                   IVNonlinearity, StuckAt,
                                                   StuckAtGMax, StuckAtGMin,
                                                   StuckDistribution)
from awarememristor.simulations import utils


def ideal():
    return {"G_min": None, "G_max": None, "V_ref": None, "nonidealities": []}


def _SiO_x_V_ref() -> dict[str, float]:
    return {"V_ref": 0.25}


def _SiO_x_G(is_high_nonlinearity: bool) -> dict[str, float]:
    if is_high_nonlinearity:
        return {
            "G_min": 1 / 1295000,
            "G_max": 1 / 366200,
        }
    return {
        "G_min": 1 / 1003,
        "G_max": 1 / 284.6,
    }


def _SiO_x_nonidealities(is_high_nonlinearity: bool):
    V_ref = _SiO_x_V_ref()["V_ref"]
    if is_high_nonlinearity:
        return {
            "nonidealities": [IVNonlinearity(V_ref, 2.989, 0.369)],
        }
    return {
        "nonidealities": [IVNonlinearity(V_ref, 2.132, 0.095)],
    }


def SiO_x(is_high_nonlinearity: bool):
    return {
        **_SiO_x_V_ref(),
        **_SiO_x_G(is_high_nonlinearity),
        **_SiO_x_nonidealities(is_high_nonlinearity),
    }


def stuck_off():
    G = _SiO_x_G(True)
    return {
        **_SiO_x_V_ref(),
        **G,
        "nonidealities": [
            StuckAtGMin(G["G_min"], 0.05),
        ],
    }


def SiO_x_high_nonlinearity_and_stuck_on():
    is_high_nonlinearity = True
    G = _SiO_x_G(is_high_nonlinearity)
    nonidealities = _SiO_x_nonidealities(is_high_nonlinearity)["nonidealities"] + [
        StuckAtGMax(G["G_max"], 0.05)
    ]
    return {
        **_SiO_x_V_ref(),
        **G,
        "nonidealities": nonidealities,
    }


def more_uniform_d2d():
    G = _SiO_x_G(True)
    return {
        **_SiO_x_V_ref(),
        **G,
        "nonidealities": [D2DLognormal(G["G_min"], G["G_max"], 0.25, 0.25)],
    }


def less_uniform_d2d():
    G = _SiO_x_G(True)
    return {
        **_SiO_x_V_ref(),
        **G,
        "nonidealities": [D2DLognormal(G["G_min"], G["G_max"], 0.05, 0.5)],
    }


def high_magnitude_more_uniform_d2d():
    G = _SiO_x_G(True)
    return {
        **_SiO_x_V_ref(),
        **G,
        "nonidealities": [D2DLognormal(G["G_min"], G["G_max"], 0.5, 0.5)],
    }


def HfO2():
    data = utils.load_cycling_data("data/HfO2-data.mat")
    G_min, G_max = utils.extract_G_min_and_G_max(data)
    vals, p = utils.extract_stuck(data, G_min, G_max)
    return {
        **_SiO_x_V_ref(),
        "G_min": G_min,
        "G_max": G_max,
        "nonidealities": [StuckDistribution(vals, p)],
    }
