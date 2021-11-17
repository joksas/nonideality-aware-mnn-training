from awarememristor.crossbar.nonidealities import (D2DLognormal,
                                                   IVNonlinearity, StuckAt,
                                                   StuckAtGMax, StuckAtGMin,
                                                   StuckDistribution)
from awarememristor.simulations import utils


def ideal():
    return {"G_off": None, "G_on": None, "V_ref": None, "nonidealities": []}


def _SiO_x_V_ref() -> dict[str, float]:
    return {"V_ref": 0.25}


def _SiO_x_G(is_high_nonlinearity: bool) -> dict[str, float]:
    if is_high_nonlinearity:
        return {
            "G_off": 1 / 1295000,
            "G_on": 1 / 366200,
        }
    return {
        "G_off": 1 / 1003,
        "G_on": 1 / 284.6,
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
            StuckAtGMin(G["G_off"], 0.05),
        ],
    }


def SiO_x_high_nonlinearity_and_stuck_on():
    is_high_nonlinearity = True
    G = _SiO_x_G(is_high_nonlinearity)
    nonidealities = _SiO_x_nonidealities(is_high_nonlinearity)["nonidealities"] + [
        StuckAtGMax(G["G_on"], 0.05)
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
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.25, 0.25)],
    }


def less_uniform_d2d():
    G = _SiO_x_G(True)
    return {
        **_SiO_x_V_ref(),
        **G,
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.05, 0.5)],
    }


def high_magnitude_more_uniform_d2d():
    G = _SiO_x_G(True)
    return {
        **_SiO_x_V_ref(),
        **G,
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.5, 0.5)],
    }


def HfO2():
    data = utils.load_cycling_data("data/HfO2-data.mat")
    G_off, G_on = utils.extract_G_off_and_G_on(data)
    vals, p = utils.extract_stuck(data, G_off, G_on)
    return {
        **_SiO_x_V_ref(),
        "G_off": G_off,
        "G_on": G_on,
        "nonidealities": [StuckDistribution(vals, p)],
    }
