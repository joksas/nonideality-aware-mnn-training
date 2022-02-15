import os
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
import openpyxl
import pandas as pd
import requests
import scipy.constants as const
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.stats import levene, linregress, probplot

from awarememristor.crossbar import nonidealities
from awarememristor.simulations import utils


def load_SiO_x_multistate() -> np.ndarray:
    """Load SiO_x data from multiple conductance states.

    Returns:
        Array of shape `(2, num_states, num_points)`. The first dimension
            combines current and voltage values.
    """
    path = os.path.join(_create_and_get_data_dir(), "SiO_x-multistate-data.mat")
    _validate_data_path(path, url="https://zenodo.org/record/5762184/files/excelDataCombined.mat")
    data = loadmat(path)["data"]
    data = np.flip(data, axis=2)
    data = np.transpose(data, (1, 2, 0))
    data = data[:2, :, :]
    return data


def _workbook_sheet_to_ndarray(workbook, sheet_name: str):
    sheet = workbook[sheet_name]
    return pd.DataFrame(sheet.values).to_numpy()[1:, :]


def load_SiO_x_switching() -> np.ndarray:
    """Load SiO_x switching data.

    Returns:
        Array of shape `(num_points, 2, 2)`. The second dimension combines
            current and voltage values, while the third combined SET and RESET
            modes.
    """
    path = os.path.join(_create_and_get_data_dir(), "SiO_x-switching-data.xlsx")
    _validate_data_path(
        path,
        url="https://zenodo.org/record/5762184/files/Ordinary%20I-V%20data%20%28full%20cycle%29.xlsx",
    )
    worksheet = openpyxl.load_workbook(filename=path)
    set_data = _workbook_sheet_to_ndarray(worksheet, "SET")[:, :2]
    reset_data = _workbook_sheet_to_ndarray(worksheet, "RESET")[:, :2]
    data = np.stack([set_data, reset_data], axis=-1)
    return data


def all_SiO_x_curves(data, max_voltage=0.5, voltage_step=0.005, clean_data=True):
    num_points = int(max_voltage / voltage_step) + 1

    data = data[:, :, :num_points]
    if clean_data:
        data = _clean_iv_data(data)
    voltages = data[1, :, :]
    currents = data[0, :, :]

    return voltages, currents


def _clean_iv_data(
    data: npt.NDArray[np.float64], threshold: float = 0.1
) -> npt.NDArray[np.float64]:
    """Remove curves where there are unusual spikes in current."""
    ok_rows = []
    for idx in range(data.shape[1]):
        i = data[0, idx, :]
        # Second derivative; helpful to understand the smoothness of the curve.
        d2y_dx2 = np.gradient(np.gradient(i))
        # We care about the relative size of the spikes.
        ratio = d2y_dx2 / np.mean(i)
        if ratio.max() < threshold:
            ok_rows.append(idx)

    return data[:, ok_rows, :]


def low_high_n_SiO_x_curves(data):
    # Arbitrary, but 11 results in a similar G_on/G_off ratio.
    NUM_LOW_N_CURVES = 11

    voltages, currents = all_SiO_x_curves(data, clean_data=False)

    num_points = voltages.shape[1]
    half_voltage_idx = int(num_points / 2)
    resistances = voltages[:, half_voltage_idx] / currents[:, half_voltage_idx]
    indices = np.argsort(resistances)
    resistances = resistances[indices]
    voltages = voltages[indices, :]
    currents = currents[indices, :]

    low_n_ratio = resistances[NUM_LOW_N_CURVES - 1] / resistances[0]

    high_n_R_off = resistances[-1]
    idx = len(indices) - 2
    while True:
        # Stop whenever we exceed G_on/G_off ratio of low-nonlinearity region.
        if high_n_R_off / resistances[idx] > low_n_ratio:
            break
        idx -= 1

    low_n_voltages = voltages[:NUM_LOW_N_CURVES, :]
    low_n_currents = currents[:NUM_LOW_N_CURVES, :]
    high_n_voltages = voltages[idx:, :]
    high_n_currents = currents[idx:, :]

    return (low_n_voltages, low_n_currents), (high_n_voltages, high_n_currents)


def nonlinearity_parameter(current_curve):
    num_points = len(current_curve)
    half_voltage_idx = int(num_points / 2)
    return current_curve[-1] / current_curve[half_voltage_idx]


def G_at_half_voltage(voltage_curve, current_curve):
    num_points = len(current_curve)
    half_voltage_idx = int(num_points / 2)
    return current_curve[half_voltage_idx] / voltage_curve[half_voltage_idx]


def low_high_n_SiO_x_vals(data, is_high_nonlinearity):
    curves = low_high_n_SiO_x_curves(data)
    if is_high_nonlinearity:
        idx = 1
    else:
        idx = 0
    voltage_curves, current_curves = curves[idx]

    n = [nonlinearity_parameter(curve) for curve in current_curves]
    n_avg, n_std = np.mean(n), np.std(n, ddof=1)

    G_on = G_at_half_voltage(voltage_curves[0, :], current_curves[0, :])
    G_off = G_at_half_voltage(voltage_curves[-1, :], current_curves[-1, :])
    return G_off, G_on, n_avg, n_std


def linregress_params(x, y):
    result = linregress(x, y)
    y_pred = result.slope * x + result.intercept
    residuals = y - y_pred
    std = np.std(residuals, ddof=1)
    return result.slope, result.intercept, std


def pf_relationship(
    V, I
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    num_curves = V.shape[0]
    resistances = np.zeros(num_curves)
    c = np.zeros(num_curves)
    d_times_perm = np.zeros(num_curves)

    for idx in range(num_curves):
        v = V[idx, :]
        i = I[idx, :]

        r = v[20] / i[20]
        resistances[idx] = r

        popt, _ = curve_fit(nonidealities.IVNonlinearityPF.model_fitting, v, i, p0=[1e-5, 1e-16])
        c[idx] = popt[0]
        d_times_perm[idx] = popt[1]

    return resistances, c, d_times_perm


def pf_params(
    data, is_high_resistance: bool
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    V, I = all_SiO_x_curves(data, clean_data=True)
    resistances, c, d_times_perm = pf_relationship(V, I)

    resistances, c, d_times_perm = utils.sort_multiple(resistances, c, d_times_perm)

    ln_resistances = np.log(resistances)
    ln_d_times_perm = np.log(d_times_perm)
    ln_c = np.log(c)

    # ln(c) vs ln(resistances)
    ln_c_params = linregress_params(ln_resistances, ln_c)

    # Separate data into before and after the conductance quantum.
    sep_idx = np.searchsorted(
        resistances, const.physical_constants["inverse of conductance quantum"][0]
    )
    if is_high_resistance:
        x = ln_resistances[sep_idx:]
        y = ln_d_times_perm[sep_idx:]
    else:
        x = ln_resistances[:sep_idx]
        y = ln_d_times_perm[:sep_idx]

    # ln(d) vs ln(resistances)
    ln_d_times_perm_params = linregress_params(x, y)

    return ln_c_params, ln_d_times_perm_params


def load_Ta_HfO2():
    """Load Ta/HfO2 data.

    Returns:
        Array of shape `(num_cycles, num_pulses, num_bit_lines, num_word_lines)`.
            The first half of `num_pulses` denotes potentiation, while the second
            half denotes depression.
    """
    path = os.path.join(_create_and_get_data_dir(), "Ta_HfO2-data.mat")
    _validate_data_path(path)
    f = h5py.File(path, "r")
    data = f.get("G_reads")
    data = np.array(data)
    return data


def extract_G_off_and_G_on(data: np.ndarray) -> tuple[float, float]:
    shape = data.shape
    data = np.reshape(data, (shape[0] * shape[1], shape[2] * shape[3]))
    G_offs = np.min(data, axis=0)
    G_off = np.median(G_offs)
    G_ons = np.max(data, axis=0)
    G_on = np.median(G_ons)

    return G_off, G_on


def extract_stuck(data: np.ndarray, G_off: float, G_on: float) -> tuple[list[float], float]:
    median_range = G_on - G_off
    shape = data.shape
    data = np.reshape(data, (shape[0] * shape[1], shape[2] * shape[3]))
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    ranges = maxs - mins
    means = np.mean(data, axis=0)
    stuck_values = means[np.where(ranges < stuck_device_threshold(median_range))]
    probability_stuck = stuck_values.shape[0] / means.shape[0]
    return stuck_values.tolist(), probability_stuck


def stuck_device_threshold(median_range):
    return median_range / 2


def _validate_data_path(path: str, url: Optional[str] = None) -> None:
    if os.path.isfile(path):
        return

    if url is None:
        raise ValueError(f'Data file "{path}" does not exist and the URL has not been provided.')

    with open(path, "wb") as file:
        response = requests.get(url)
        file.write(response.content)


def _create_and_get_data_dir() -> str:
    dir_name = ".data"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name
