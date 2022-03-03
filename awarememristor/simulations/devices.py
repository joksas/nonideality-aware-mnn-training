from typing import Any

from awarememristor.crossbar.nonidealities import (D2DLognormal,
                                                   IVNonlinearityPF,
                                                   Nonideality, StuckAtGOff,
                                                   StuckAtGOn,
                                                   StuckDistribution)
from awarememristor.simulations import data


def ideal() -> dict[str, Any]:
    return {"G_off": None, "G_on": None, "nonidealities": []}


def SiO_x_k_V() -> dict[str, float]:
    exp_data = data.load_SiO_x_multistate()
    voltages, _ = data.all_SiO_x_curves(exp_data)
    k_V = voltages[0][-1]

    return {"k_V": float(k_V)}


def _SiO_x_G(is_high_resistance: bool) -> dict[str, float]:
    exp_data = data.load_SiO_x_multistate()
    G_off, G_on, _, _, _ = data.pf_params(
        exp_data, is_high_resistance, data.SiO_x_G_on_G_off_ratio()
    )
    return {
        "G_off": float(G_off),
        "G_on": float(G_on),
    }


def _SiO_x_nonidealities(is_high_resistance: bool) -> dict[str, list[Nonideality]]:
    exp_data = data.load_SiO_x_multistate()
    _, _, slopes, intercepts, res_cov_matrix = data.pf_params(
        exp_data, is_high_resistance, data.SiO_x_G_on_G_off_ratio()
    )
    k_V = SiO_x_k_V()["k_V"]
    return {"nonidealities": [IVNonlinearityPF(k_V, slopes, intercepts, res_cov_matrix)]}


def SiO_x(is_high_nonlinearity: bool) -> dict[str, Any]:
    return {
        **_SiO_x_G(is_high_nonlinearity),
        **_SiO_x_nonidealities(is_high_nonlinearity),
    }


def stuck_off() -> dict[str, Any]:
    G = _SiO_x_G(True)
    return {
        **G,
        "nonidealities": [
            StuckAtGOff(G["G_off"], 0.05),
        ],
    }


def SiO_x_high_nonlinearity_and_stuck_on() -> dict[str, Any]:
    is_high_nonlinearity = True
    G = _SiO_x_G(is_high_nonlinearity)
    nonidealities = _SiO_x_nonidealities(is_high_nonlinearity)["nonidealities"] + [
        StuckAtGOn(G["G_on"], 0.05)
    ]
    return {
        **G,
        "nonidealities": nonidealities,
    }


def more_uniform_d2d() -> dict[str, Any]:
    G = _SiO_x_G(True)
    return {
        **G,
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.25, 0.25)],
    }


def less_uniform_d2d() -> dict[str, Any]:
    G = _SiO_x_G(True)
    return {
        **G,
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.05, 0.5)],
    }


def high_magnitude_more_uniform_d2d() -> dict[str, Any]:
    G = _SiO_x_G(True)
    return {
        **G,
        "nonidealities": [D2DLognormal(G["G_off"], G["G_on"], 0.5, 0.5)],
    }


def Ta_HfO2() -> dict[str, Any]:
    exp_data = data.load_Ta_HfO2()
    G_off, G_on = data.extract_G_off_and_G_on(exp_data)
    G_off, G_on = float(G_off), float(G_on)
    vals, p = data.extract_stuck(exp_data, G_off, G_on)
    return {
        "G_off": G_off,
        "G_on": G_on,
        "nonidealities": [StuckDistribution(vals, p)],
    }
