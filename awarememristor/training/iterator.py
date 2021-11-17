import os
import pickle
import warnings
from typing import Any, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from awarememristor.crossbar.nonidealities import Nonideality
from awarememristor.training import callbacks, network, utils

warnings.simplefilter("default")


class Nonideal:
    def __init__(
        self,
        G_min: float = None,
        G_max: float = None,
        nonidealities: list[Nonideality] = [],
    ) -> None:
        self.G_min = G_min
        self.G_max = G_max
        self.nonidealities = nonidealities

    def __eq__(self, other):
        return (
            self.G_min == other.G_min
            and self.G_max == other.G_max
            and self.nonidealities == other.nonidealities
        )

    def conductance_label(self) -> str:
        if self.G_min is None and self.G_max is None:
            return "none_none"

        return f"{self.G_min:.3g}_{self.G_max:.3g}"

    def nonideality_label(self) -> str:
        if len(self.nonidealities) == 0:
            return "ideal"

        return "+".join(nonideality.label() for nonideality in self.nonidealities)

    def label(self) -> str:
        return f"{self.conductance_label()}__{self.nonideality_label()}"

    def is_nonideal(self) -> bool:
        if len(self.nonidealities) == 0:
            return False

        return True

    def is_aware(self) -> bool:
        return self.is_nonideal()


class Iterable:
    def __init__(self) -> None:
        self.repeat_idx = 0

    def __eq__(self, other):
        return self.repeat_idx == other.repeat_idx


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
        nonidealities: list[Nonideality] = [],
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
        Nonideal.__init__(self, G_min=G_min, G_max=G_max, nonidealities=nonidealities)
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

    def network_label(self) -> str:
        return f"network-{self.repeat_idx}"


class Inference(Nonideal, Iterable):
    def __init__(
        self,
        num_repeats: int = 0,
        G_min: float = None,
        G_max: float = None,
        nonidealities: list[Nonideality] = [],
    ) -> None:
        self.num_repeats = num_repeats
        Nonideal.__init__(self, G_min=G_min, G_max=G_max, nonidealities=nonidealities)
        Iterable.__init__(self)

    def repeat_label(self) -> str:
        return f"repeat-{self.repeat_idx}"

    def __eq__(self, other):
        return (
            self.num_repeats == other.num_repeats
            and Nonideal.__eq__(self, other)
            and Iterable.__eq__(self, other)
        )


class Iterator:
    def __init__(self, dataset: str, training: Training, inferences: list[Inference]) -> None:
        self.dataset = dataset
        self.training = training
        self.inferences = inferences
        self.is_callback = False
        self.is_training = False
        self.inference_idx = 0
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
        return os.path.join(self.network_dir(), self.inferences[self.inference_idx].label())

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

    def training_curves(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        if metric == "error":
            y = self.info()["history"]["accuracy"]
        else:
            y = self.info()["history"][metric]

        num_epochs = len(y)
        x = np.arange(1, num_epochs + 1)

        y = np.array(y)
        if metric == "error":
            y = 1 - y

        return x, y

    def validation_curves(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            if metric == "error":
                y = self.info()["history"]["val_accuracy"]
            else:
                y = self.info()["history"]["val_" + metric]
            num_epochs = len(y)
            x = np.arange(1, num_epochs + 1)
        except KeyError:
            history = self.info()["callback_infos"]["memristive_checkpoint"]["history"]
            x = history["epoch_no"]
            x = np.array(x)
            if metric == "error":
                y = history["accuracy"]
            else:
                y = history[metric]

        y = np.array(y)
        if metric == "error":
            y = 1 - y

        return x, y

    def training_testing_curves(self, metric: str, inference: Inference):
        """Data from test callbacks during training."""

        history = self.info()["callback_infos"]["memristive_test"]["history"][
            self._memristive_test_callback_idx(inference)
        ]

        if metric == "error":
            y = history["accuracy"]
        else:
            y = history[metric]

        x = np.array(history["epoch_no"])

        y = np.array(y)
        if metric == "error":
            y = 1 - y

        return x, y

    def _memristive_test_callback_idx(self, inference: Inference) -> int:
        """Number of inferences might not equal the number of memristive test callbacks."""
        # New method
        label = inference.label()
        for idx, history in enumerate(self.info()["callback_infos"]["memristive_test"]["history"]):
            try:
                if history["label"] == label:
                    return idx
            except KeyError:
                break

        # Old method
        nonideality_label = inference.nonideality_label()
        for idx, history in enumerate(self.info()["callback_infos"]["memristive_test"]["history"]):
            if history["nonideality_label"] == nonideality_label:
                warnings.warn(
                    "Using the old method of storing memristive test results.",
                    PendingDeprecationWarning,
                )
                return idx

        raise ValueError("Index not found.")

    def _test_metric_existing(self, inference_idx: int, metric: str = "accuracy") -> np.ndarray:
        """Return test metric for which we already have data."""
        self.inference_idx = inference_idx
        inference = self.inferences[self.inference_idx]
        y = np.zeros((self.training.num_repeats, inference.num_repeats))
        for i in range(self.training.num_repeats):
            for j in range(inference.num_repeats):
                if metric == "accuracy":
                    filename = self.accuracy_path()
                elif metric == "loss":
                    filename = self.loss_path()
                elif metric == "avg_power":
                    filename = self.power_path()
                val = np.genfromtxt(filename)
                if metric == "avg_power":
                    val = np.mean(val)
                    # Two synaptic layers.
                    val = 2 * val

                y[i, j] = val

                inference.repeat_idx += 1

            inference.repeat_idx = 0
            self.training.repeat_idx += 1

        self.training.repeat_idx = 0
        self.inference_idx = 0
        return y

    def test_metric(self, metric: str, inference_idx: int = 0) -> np.ndarray:
        if metric in "error":
            values = self._test_metric_existing(inference_idx, metric="accuracy")
        else:
            values = self._test_metric_existing(inference_idx, metric=metric)

        if metric == "error":
            values = 1 - values

        return values

    def train(self, use_test_callback: bool = False) -> None:
        self.is_training = True

        for _ in range(self.training.num_repeats):
            if os.path.isdir(self.network_dir()):
                warnings.warn(
                    f'Training directory "{self.network_dir()}" already exists. Skipping...'
                )
                self.training.repeat_idx += 1
                continue
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
            if os.path.isdir(self.inference_nonideality_dir()):
                warnings.warn(
                    f'Inference directory "{self.inference_nonideality_dir()}" already exists. Skipping...'
                )
                continue
            inference = self.inferences[self.inference_idx]
            for _ in range(self.training.num_repeats):
                for _ in range(inference.num_repeats):
                    network.infer(self)
                    inference.repeat_idx += 1

                inference.repeat_idx = 0
                self.training.repeat_idx += 1

            self.training.repeat_idx = 0
        self.inference_idx = 0
