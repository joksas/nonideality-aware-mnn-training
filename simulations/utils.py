import h5py
import numpy as np
from scipy.io import loadmat


def get_training_params():
    return {
        "num_repeats": 5,
        "num_epochs": 1000,
        "batch_size": 64,
    }


def get_inference_params():
    return {
        "num_repeats": 25,
    }


def load_iv_data(path: str):
    data = loadmat(path)["data"]
    data = np.flip(data, axis=2)
    return data


def load_cycling_data(path: str, var_name: str = "G_reads"):
    """Load cycling data.

    Args:
        path: Filepath to `MAT` file.
        var_name: MATLAB variable name.

    Returns:
        Array of shape `(num_cycles, num_pulses, num_bit_lines, num_word_lines)`.
            The first half of `num_pulses` denotes potentiation, while the second
            half denotes depression.
    """
    f = h5py.File(path, "r")
    data = f.get(var_name)
    data = np.array(data)
    return data


def extract_G_min_and_G_max(data: np.ndarray) -> tuple[float, float]:
    shape = data.shape
    data = np.reshape(data, (shape[0] * shape[1], shape[2] * shape[3]))
    G_mins = np.min(data, axis=0)
    G_min = np.median(G_mins)
    G_maxs = np.max(data, axis=0)
    G_max = np.median(G_maxs)

    return G_min, G_max


def extract_stuck(data: np.ndarray, G_min: float, G_max: float) -> tuple[list[float], float]:
    median_range = G_max - G_min
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