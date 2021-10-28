import os
import pickle
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_datasets as tfds

from . import callbacks, network, utils


class D2DLognormal:
    def __init__(self, R_min_std: float, R_max_std: float) -> None:
        self.R_min_std = R_min_std
        self.R_max_std = R_max_std

    def label(self) -> str:
        return f"D2DLN:{self.R_min_std:.3g}_{self.R_max_std:.3g}"


class IVNonlinearity:
    def __init__(self, n_avg: float, n_std: float) -> None:
        self.n_avg = n_avg
        self.n_std = n_std

    def label(self) -> str:
        return f"IVNL:{self.n_avg:.3g}_{self.n_std:.3g}"


class Stuck:
    def __init__(self, p: float) -> None:
        self.p = p


class StuckAtGMin(Stuck):
    def __init__(self, p: float) -> None:
        Stuck.__init__(self, p)

    def label(self) -> str:
        return f"StuckMin:{self.p:.3g}"


class StuckAtGMax(Stuck):
    def __init__(self, p: float) -> None:
        Stuck.__init__(self, p)

    def label(self) -> str:
        return f"StuckMax:{self.p:.3g}"


class Nonideal:
    def __init__(
        self,
        G_min: float = None,
        G_max: float = None,
        iv_nonlinearity: IVNonlinearity = None,
        stuck_at_G_min: StuckAtGMin = None,
        stuck_at_G_max: StuckAtGMax = None,
        d2d_lognormal: D2DLognormal = None,
    ) -> None:
        self.G_min = G_min
        self.G_max = G_max
        self.iv_nonlinearity = iv_nonlinearity
        self.stuck_at_G_min = stuck_at_G_min
        self.stuck_at_G_max = stuck_at_G_max
        self.d2d_lognormal = d2d_lognormal

    def nonideality_list(
        self,
    ) -> list[Union[D2DLognormal, IVNonlinearity, StuckAtGMin, StuckAtGMax]]:
        nonidealities = []
        for nonideality in [
            self.iv_nonlinearity,
            self.stuck_at_G_min,
            self.stuck_at_G_max,
            self.d2d_lognormal,
        ]:
            if nonideality is not None:
                nonidealities.append(nonideality)

        return nonidealities

    def conductance_label(self) -> str:
        if self.G_min is None and self.G_max is None:
            return "none_none"

        return f"{self.G_min:.3g}_{self.G_max:.3g}"

    def nonideality_label(self) -> str:
        nonidealities = self.nonideality_list()
        if len(nonidealities) == 0:
            return "ideal"

        return "+".join(nonideality.label() for nonideality in nonidealities)

    def label(self) -> str:
        return f"{self.conductance_label()}__{self.nonideality_label()}"

    def is_nonideal(self) -> bool:
        if len(self.nonideality_list()) == 0:
            return False

        return True


class Iterable:
    def __init__(self) -> None:
        self.repeat_idx = 0


class Training(Nonideal, Iterable):
    def __init__(
        self,
        batch_size: int = 1,
        validation_split: float = 0.2,
        num_epochs: int = 1,
        is_regularized: bool = False,
        num_repeats: int = 0,
        G_min: float = None,
        G_max: float = None,
        nonidealities: dict[
            str, Union[D2DLognormal, IVNonlinearity, StuckAtGMin, StuckAtGMax]
        ] = {},
        force_regular_checkpoint: bool = False,
        memristive_validation_freq: int = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_repeats = num_repeats
        self.is_regularized = is_regularized
        self.validation_split = validation_split
        self.force_regular_checkpoint = force_regular_checkpoint
        self.memristive_validation_freq = memristive_validation_freq
        Nonideal.__init__(self, G_min=G_min, G_max=G_max, **nonidealities)
        Iterable.__init__(self)

    def regularized_label(self) -> str:
        if self.is_regularized:
            return "reg"
        else:
            return "nonreg"

    def label(self) -> str:
        l = f"{self.regularized_label()}__{self.batch_size}__{Nonideal.label(self)}"
        if self.force_regular_checkpoint:
            l += "__rc"
        if self.memristive_validation_freq is not None:
            l += f"__val_freq_{self.memristive_validation_freq}"
        return l

    def is_aware(self) -> bool:
        return self.is_nonideal()

    def network_label(self) -> str:
        return "network-{}".format(self.repeat_idx)


class Inference(Nonideal, Iterable):
    def __init__(
        self,
        num_repeats: int = 0,
        G_min: float = None,
        G_max: float = None,
        nonidealities={},
    ) -> None:
        self.num_repeats = num_repeats
        Nonideal.__init__(self, G_min=G_min, G_max=G_max, **nonidealities)
        Iterable.__init__(self)

    def repeat_label(self) -> str:
        return "repeat-{}".format(self.repeat_idx)


class Iterator:
    def __init__(
        self, dataset: str, training: Training, inferences: list[Inference]
    ) -> None:
        self.dataset = dataset
        self.training = training
        self.inferences = inferences
        self.is_callback = False
        self.is_training = False
        self.inference_idx = None
        self.test_batch_size = 100  # divisor of the size of the test set
        self.__training_data = None
        self.__validation_data = None
        self.__testing_data = None
        self.__train_split_boundary = int(100 * (1 - self.training.validation_split))

    def data(self, subset: str) -> tf.data.Dataset:
        if subset == "training":
            if self.__training_data is not None:
                return self.__training_data
            split = f"train[:{self.__train_split_boundary}%]"
        elif subset == "validation":
            if self.__validation_data is not None:
                return self.__validation_data
            split = f"train[{self.__train_split_boundary}%:]"
        elif subset == "testing":
            if self.__testing_data is not None:
                return self.__testing_data
            split = "test"
        else:
            raise ValueError(f'Subset "{subset}" is not recognised!')

        ds = tfds.load(
            self.dataset,
            split=split,
            as_supervised=True,
            shuffle_files=True,
        )
        size = ds.cardinality().numpy()

        ds = ds.map(utils.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        if subset == "testing":
            ds = ds.batch(self.test_batch_size)
            ds = ds.cache()
        else:
            ds = ds.cache()
            ds = ds.shuffle(size)
            ds = ds.batch(self.training.batch_size)
            if self.dataset == "cifar10" and subset == "training":
                data_augmentation = tf.keras.Sequential(
                    [
                        tf.keras.layers.RandomTranslation(0.1, 0.1),
                        tf.keras.layers.RandomFlip("horizontal"),
                    ]
                )
                ds = ds.map(
                    lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
        ds = ds.prefetch(tf.data.AUTOTUNE)

        if subset == "training":
            self.__training_data = ds
        elif subset == "validation":
            self.__validation_data = ds
        elif subset == "testing":
            self.__testing_data = ds

        print(f'Loaded dataset "{self.dataset}" ({subset}): {size} examples.')

        return ds

    def training_dir(self) -> str:
        return os.path.join(os.getcwd(), "models", self.dataset, self.training.label())

    def network_dir(self) -> str:
        return os.path.join(self.training_dir(), self.training.network_label())

    def weights_path(self) -> str:
        return os.path.join(self.network_dir(), "model.h5")

    def info_path(self) -> str:
        return os.path.join(self.network_dir(), "info.pkl")

    def inference_nonideality_dir(self) -> str:
        return os.path.join(
            self.network_dir(), self.inferences[self.inference_idx].label()
        )

    def inference_repeat_dir(self) -> str:
        return os.path.join(
            self.inference_nonideality_dir(),
            self.inferences[self.inference_idx].repeat_label(),
        )

    def power_path(self) -> str:
        return os.path.join(self.inference_repeat_dir(), "power.csv")

    def loss_path(self) -> str:
        return os.path.join(self.inference_repeat_dir(), "loss.csv")

    def accuracy_path(self) -> str:
        return os.path.join(self.inference_repeat_dir(), "accuracy.csv")

    def info(self) -> dict[str, Any]:
        with open(self.info_path(), "rb") as pickle_file:
            return pickle.load(pickle_file)

    def current_stage(self) -> Union[Training, Inference]:
        if self.is_training:
            return self.training
        else:
            return self.inferences[self.inference_idx]

    def avg_power(self) -> list[npt.NDArray]:
        average_powers = []
        for inference_idx in range(len(self.inferences)):
            self.inference_idx = inference_idx
            inference = self.inferences[self.inference_idx]
            average_power = np.zeros((self.training.num_repeats, inference.num_repeats))

            for i in range(self.training.num_repeats):
                for j in range(inference.num_repeats):
                    filename = self.power_path()
                    csv = np.genfromtxt(filename)
                    power = np.mean(csv)
                    # Two synaptic layers.
                    power = 2 * power
                    average_power[i, j] = power

                    inference.repeat_idx += 1

                inference.repeat_idx = 0
                self.training.repeat_idx += 1

            self.training.repeat_idx = 0
            average_powers.append(average_power)

        self.inference_idx = None

        return average_powers

    def train_epochs_and_metric(self, metric: str) -> tuple[npt.NDArray, npt.NDArray]:
        metric = np.array(self.info()["history"][metric])
        num_epochs = len(metric)
        epochs = np.arange(1, num_epochs + 1)
        return epochs, metric

    def train_epochs_and_accuracy(self) -> tuple[npt.NDArray, npt.NDArray]:
        return self.train_epochs_and_metric("accuracy")

    def train_epochs_and_loss(self) -> tuple[npt.NDArray, npt.NDArray]:
        return self.train_epochs_and_metric("loss")

    def validation_epochs_and_metric(
        self, metric: str
    ) -> tuple[npt.NDArray, npt.NDArray]:
        try:
            metric = self.info()["history"]["val_" + metric]
            num_epochs = len(metric)
            epochs = np.arange(1, num_epochs + 1)
        except KeyError:
            epochs = self.info()["callback_infos"]["memristive_checkpoint"]["history"][
                "epoch_no"
            ]
            metric = self.info()["callback_infos"]["memristive_checkpoint"]["history"][
                metric
            ]
        epochs = np.array(epochs)
        metric = np.array(metric)
        return epochs, metric

    def validation_epochs_and_accuracy(self) -> tuple[npt.NDArray, npt.NDArray]:
        return self.validation_epochs_and_metric("accuracy")

    def validation_epochs_and_loss(self) -> tuple[npt.NDArray, npt.NDArray]:
        return self.validation_epochs_and_metric("loss")

    def train_test_histories(self) -> dict[str, Any]:
        return self.info()["callback_infos"]["memristive_test"]["history"]

    def test_metric(self, metric_name: str) -> list[npt.NDAarray]:
        metrics = []
        for inference_idx in range(len(self.inferences)):
            self.inference_idx = inference_idx
            inference = self.inferences[self.inference_idx]
            metric = np.zeros((self.training.num_repeats, inference.num_repeats))

            for i in range(self.training.num_repeats):
                for j in range(inference.num_repeats):
                    if metric_name == "accuracy":
                        filename = self.accuracy_path()
                    elif metric_name == "loss":
                        filename = self.loss_path()
                    csv = np.genfromtxt(filename)
                    metric[i, j] = csv

                    inference.repeat_idx += 1

                inference.repeat_idx = 0
                self.training.repeat_idx += 1

            self.training.repeat_idx = 0
            metrics.append(metric)

        self.inference_idx = None

        return metrics

    def test_accuracy(self) -> list[npt.NDArray]:
        return self.test_metric("accuracy")

    def test_loss(self) -> list[npt.NDArray]:
        return self.test_metric("loss")

    def test_error(self) -> list[npt.NDArray]:
        return [1 - accuracy for accuracy in self.test_accuracy()]

    def train(self, use_test_callback: bool = False) -> None:
        self.is_training = True

        for _ in range(self.training.num_repeats):
            # New callbacks in each iteration because iterator changes.
            train_callbacks = []
            if use_test_callback:
                train_callbacks.append(callbacks.TestCallback(self))
            if not self.training.is_aware() or self.training.force_regular_checkpoint:
                train_callbacks.append(callbacks.RegularCheckpoint(self))
            else:
                train_callbacks.append(callbacks.MemristiveCheckpoint(self))

            network.train(self, callbacks=train_callbacks)
            self.training.repeat_idx += 1

        self.training.repeat_idx = 0

    def infer(self) -> None:
        self.is_training = False
        for idx in range(len(self.inferences)):
            self.inference_idx = idx
            inference = self.inferences[self.inference_idx]
            for _ in range(self.training.num_repeats):
                for _ in range(inference.num_repeats):
                    network.infer(self)
                    inference.repeat_idx += 1

                inference.repeat_idx = 0
                self.training.repeat_idx += 1

            self.training.repeat_idx = 0
        self.inference_idx = None
