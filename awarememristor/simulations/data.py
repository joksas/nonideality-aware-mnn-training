import os
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
import openpyxl
import pandas as pd
import requests
import scipy.constants as const
import tensorflow as tf
import tensorflow_probability as tfp
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


def average_nonlinearity(voltage_curve, current_curve):
    conductance_curve = current_curve / voltage_curve
    nonlinearity = (
        conductance_curve[2::2] / conductance_curve[1 : int(len(conductance_curve) / 2) + 1]
    )
    return np.mean(nonlinearity)


def multivariate_linregress_params(
    x: tf.Tensor, *y: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    slopes: list[float] = []
    intercepts: list[float] = []
    residuals_lst: list[float] = []

    for y_i in y:
        result = linregress(x, y_i)
        slopes.append(result.slope)
        intercepts.append(result.intercept)

        y_pred = result.slope * x + result.intercept
        residuals = y_i - y_pred
        residuals_lst.append(residuals)

    slopes = tf.constant(slopes, dtype=tf.float32)
    intercepts = tf.constant(intercepts, dtype=tf.float32)
    residuals_lst = tf.constant(residuals_lst, dtype=tf.float32)
    res_cov_matrix = tfp.stats.covariance(residuals_lst, event_axis=0, sample_axis=1)

    return slopes, intercepts, res_cov_matrix


def SiO_x_G_on_G_off_ratio() -> float:
    return 5.0


def edge_state_idxs(
    sorted_resistances: npt.NDArray[np.float64], ratio: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    low_upper_idx = np.searchsorted(sorted_resistances, sorted_resistances[0] * ratio)
    low_idxs = np.arange(low_upper_idx)
    high_lower_idx = np.searchsorted(sorted_resistances, sorted_resistances[-1] / ratio)
    high_idxs = np.arange(high_lower_idx + 1, len(sorted_resistances))

    return low_idxs, high_idxs


def pf_relationship(
    V, I, voltage_step=0.005, ref_voltage=0.1
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    num_curves = V.shape[0]
    resistances = np.zeros(num_curves)
    c = np.zeros(num_curves)
    d_times_perm = np.zeros(num_curves)
    ref_idx = int(ref_voltage / voltage_step)

    for idx in range(num_curves):
        v = V[idx, :]
        i = I[idx, :]

        r = v[ref_idx] / i[ref_idx]
        resistances[idx] = r

        popt, _ = curve_fit(nonidealities.IVNonlinearityPF.model_fitting, v, i, p0=[1e-5, 1e-16])
        c[idx] = popt[0]
        d_times_perm[idx] = popt[1]

    resistances, c, d_times_perm, V, _ = utils.sort_multiple(resistances, c, d_times_perm, V, I)

    return resistances, c, d_times_perm, V, I


def pf_params(
    data, is_high_resistance: bool, ratio: float
) -> tuple[float, float, list[float], list[float], tf.Tensor]:
    V, I = all_SiO_x_curves(data, clean_data=True)
    resistances, c, d_times_perm, _, _ = pf_relationship(V, I)

    if is_high_resistance:
        G_min = 1 / resistances[-1]
        G_max = G_min * ratio
    else:
        G_max = 1 / resistances[0]
        G_min = G_max / ratio

    ln_resistances = tf.math.log(resistances)
    ln_d_times_perm = tf.math.log(d_times_perm)
    ln_c = tf.math.log(c)

    # Separate data into before and after the conductance quantum.
    sep_idx = np.searchsorted(
        resistances, const.physical_constants["inverse of conductance quantum"][0]
    )
    if is_high_resistance:
        x = ln_resistances[sep_idx:]
        y_1 = ln_c[sep_idx:]
        y_2 = ln_d_times_perm[sep_idx:]
    else:
        x = ln_resistances[:sep_idx]
        y_1 = ln_c[:sep_idx]
        y_2 = ln_d_times_perm[:sep_idx]

    slopes, intercepts, res_cov_marix = multivariate_linregress_params(x, y_1, y_2)

    return G_min, G_max, slopes, intercepts, res_cov_marix


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
