from awarememristor.crossbar.nonidealities import (D2DLognormal,
                                                   IVNonlinearity, StuckAtGOff,
                                                   StuckAtGOn,
                                                   StuckDistribution)
from awarememristor.simulations import data


def ideal():
    return {"G_off": None, "G_on": None, "nonidealities": []}


def SiO_x_V_ref() -> dict[str, float]:
    exp_data = data.load_SiO_x()
    (voltages, _), (_, _) = data.low_high_n_SiO_x_curves(exp_data)
    V_ref = voltages[0][-1] / 2

    return {"V_ref": float(V_ref)}


def _SiO_x_G(is_high_nonlinearity: bool) -> dict[str, float]:
    exp_data = data.load_SiO_x()
    G_off, G_on, _, _ = data.low_high_n_SiO_x_vals(exp_data, is_high_nonlinearity)
    return {
        "G_off": float(G_off),
        "G_on": float(G_on),
    }


def _SiO_x_nonidealities(is_high_nonlinearity: bool):
    exp_data = data.load_SiO_x()
    _, _, n_avg, n_std = data.low_high_n_SiO_x_vals(exp_data, is_high_nonlinearity)
    V_ref = SiO_x_V_ref()["V_ref"]
    return {
        "nonidealities": [IVNonlinearity(V_ref, float(n_avg), float(n_std))],
    }


def SiO_x(is_high_nonlinearity: bool):
    return {
        **_SiO_x_G(is_high_nonlinearity),
        **_SiO_x_nonidealities(is_high_nonlinearity),
    }


def stuck_off():
    G = _SiO_x_G(True)
    return {
        **G,
        "nonidealities": [
            StuckAtGOff(G["G_off"], 0.05),
        ],
    }


def SiO_x_high_nonlinearity_and_stuck_on():
    is_high_nonlinearity = True
    G = _SiO_x_G(is_high_nonlinearity)
    nonidealities = _SiO_x_nonidealities(is_high_nonlinearity)["nonidealities"] + [
        StuckAtGOn(G["G_on"], 0.05)
    ]
    return {
        **G,
        "nonidealities": nonidealities,
    }


def more_uniform_d2d():
    G = _SiO_x_G(True)
    return {
        **G,
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.25, 0.25)],
    }


def less_uniform_d2d():
    G = _SiO_x_G(True)
    return {
        **G,
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.05, 0.5)],
    }


def high_magnitude_more_uniform_d2d():
    G = _SiO_x_G(True)
    return {
        **G,
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.5, 0.5)],
    }


def Ta_HfO2():
    exp_data = data.load_Ta_HfO2()
    G_off, G_on = data.extract_G_off_and_G_on(exp_data)
    G_off, G_on = float(G_off), float(G_on)
    vals, p = data.extract_stuck(exp_data, G_off, G_on)
    return {
        "G_off": G_off,
        "G_on": G_on,
        "nonidealities": [StuckDistribution(vals, p)],
    }
