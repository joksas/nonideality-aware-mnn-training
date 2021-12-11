import os
import pickle
import warnings
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from awarememristor.crossbar.nonidealities import (LinearityNonpreserving,
                                                   LinearityPreserving,
                                                   Nonideality)
from awarememristor.simulations import devices
from awarememristor.training import callbacks, network, utils

warnings.simplefilter("default")


class Iterable:
    def __init__(self, num_repeats: int) -> None:
        self.repeat_idx = 0
        self.num_repeats = num_repeats

    def __eq__(self, other):
        return self.repeat_idx == other.repeat_idx and self.num_repeats == other.num_repeats


class Stage(Iterable):
    def __init__(
        self,
        G_off: float = None,
        G_on: float = None,
        nonidealities: list[Nonideality] = None,
        mapping_rule: str = "default",
        num_repeats: int = 0,
    ) -> None:
        self.G_off = G_off
        self.G_on = G_on
        if nonidealities is None:
            nonidealities = []
        self.nonidealities = nonidealities
        self.mapping_rule = mapping_rule
        self.validate_nonidealities()
        Iterable.__init__(self, num_repeats)

    def __eq__(self, other):
        return (
            self.G_off == other.G_off
            and self.G_on == other.G_on
            and self.nonidealities == other.nonidealities
            and self.mapping_rule == other.mapping_rule
            and Iterable.__eq__(self, other)
        )

    def conductance_label(self) -> str:
        if self.G_off is None and self.G_on is None:
            return "none_none"

        return f"{self.G_off:.3g}_{self.G_on:.3g}"

    def nonideality_label(self) -> str:
        if len(self.nonidealities) == 0:
            return "ideal"

        l = "+".join(nonideality.label() for nonideality in self.nonidealities)
        if self.mapping_rule != "default":
            l += f"__{self.mapping_rule}"
        return l

    def label(self) -> str:
        return f"{self.conductance_label()}__{self.nonideality_label()}"

    def is_nonideal(self) -> bool:
        return len(self.nonidealities) > 0

    def k_V(self) -> float:
        for nonideality in self.nonidealities:
            if isinstance(nonideality, LinearityNonpreserving):
                return nonideality.k_V()

        # Except for power consumption, `k_V` makes no difference for
        # linearity-preserving nonidealities, thus using the same value as for
        # SiO_x devices.
        return 2 * devices.SiO_x_V_ref()["V_ref"]

    def validate_nonidealities(self) -> None:
        num_linearity_preserving = 0
        num_linearity_nonpreserving = 0
        for nonideality in self.nonidealities:
            if isinstance(nonideality, LinearityPreserving):
                num_linearity_preserving += 1
            elif isinstance(nonideality, LinearityNonpreserving):
                num_linearity_nonpreserving += 1

        for num, nonideality_type in zip(
            [num_linearity_preserving, num_linearity_nonpreserving],
            ["linearity-preserving", "linearity-nonpreserving"],
        ):
            if num > 1:
                raise ValueError(
                    f"Current implementation does not support more than one {nonideality_type} nonideality."
                )


class Training(Stage, Iterable):
    def __init__(
        self,
        batch_size: int = 1,
        validation_split: float = 0.2,
        num_epochs: int = 1,
        is_regularized: bool = False,
        num_repeats: int = 0,
        G_off: float = None,
        G_on: float = None,
        nonidealities: list[Nonideality] = None,
        use_combined_validation: bool = False,
        memristive_validation_freq: int = None,
        mapping_rule: str = "default",
        force_standard_w: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.is_regularized = is_regularized
        self.validation_split = validation_split
        self.use_combined_validation = use_combined_validation
        self.is_standard_validation_mode = False
        self.memristive_validation_freq = memristive_validation_freq
        self.force_standard_w = force_standard_w
        Stage.__init__(
            self,
            G_off=G_off,
            G_on=G_on,
            nonidealities=nonidealities,
            mapping_rule=mapping_rule,
            num_repeats=num_repeats,
        )

    def regularized_label(self) -> str:
        if self.is_regularized:
            return "reg"
        else:
            return "nonreg"

    def label(self) -> str:
        l = f"{self.regularized_label()}__{self.batch_size}__{Stage.label(self)}"
        if self.memristive_validation_freq is not None:
            l += f"__val_freq_{self.memristive_validation_freq}"
        if self.force_standard_w:
            l += "__standard_w"
        return l

    def network_label(self) -> str:
        return f"network-{self.repeat_idx}"

    def uses_double_weights(self) -> bool:
        return self.is_nonideal() and not self.force_standard_w


class Inference(Stage):
    def __init__(
        self,
        num_repeats: int = 0,
        G_off: float = None,
        G_on: float = None,
        nonidealities: list[Nonideality] = None,
        mapping_rule: str = "default",
    ) -> None:
        self.num_repeats = num_repeats
        Stage.__init__(
            self,
            G_off=G_off,
            G_on=G_on,
            nonidealities=nonidealities,
            mapping_rule=mapping_rule,
            num_repeats=num_repeats,
        )

    def repeat_label(self) -> str:
        return f"repeat-{self.repeat_idx}"


class Iterator:
    def __init__(self, dataset: str, training: Training, inferences: list[Inference]) -> None:
        self.dataset = dataset
        self.training = training
        self.inferences = inferences
        self.compute_power = False
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

    def weights_dir(self) -> str:
        if self.training.use_combined_validation:
            if self.training.is_standard_validation_mode:
                return os.path.join(self.network_dir(), "standard-validation")
            return os.path.join(self.network_dir(), "memristive-validation")
        return self.network_dir()

    def weights_path(self) -> str:
        filename = "model.h5"
        return os.path.join(self.weights_dir(), filename)

    def info_path(self) -> str:
        return os.path.join(self.network_dir(), "info.pkl")

    def inference_nonideality_dir(self) -> str:
        return os.path.join(self.weights_dir(), self.inferences[self.inference_idx].label())

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

    def current_stage(self) -> Stage:
        if self.is_training:
            return self.training
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

    def _checkpoint_from_info(self) -> str:
        if not self.training.is_nonideal():
            return callbacks.StandardCheckpoint.name()
        if self.training.use_combined_validation:
            return callbacks.CombinedCheckpoint.name()
        return callbacks.MemristiveCheckpoint.name()

    def validation_curves(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        checkpoint_name = self._checkpoint_from_info()
        info = self.info()
        if checkpoint_name == callbacks.StandardCheckpoint.name():
            if metric == "error":
                y = info["history"]["val_accuracy"]
            else:
                y = info["history"]["val_" + metric]
            num_epochs = len(y)
            x = np.arange(1, num_epochs + 1)
        elif checkpoint_name == callbacks.MemristiveCheckpoint.name():
            history = info["callback_infos"][callbacks.MemristiveCheckpoint.name()]["history"]
            x = history["epoch_no"]
            x = np.array(x)
            if metric == "error":
                y = history["accuracy"]
            else:
                y = history[metric]
        elif checkpoint_name == callbacks.CombinedCheckpoint.name():
            prepend = ""
            if self.training.is_standard_validation_mode:
                prepend = "standard_"
            history = info["callback_infos"][callbacks.CombinedCheckpoint.name()]["history"]
            x = history[f"{prepend}epoch_no"]
            x = np.array(x)
            if metric == "error":
                y = history[f"{prepend}accuracy"]
            else:
                y = history[f"{prepend}{metric}"]

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
        label = inference.label()
        for idx, history in enumerate(self.info()["callback_infos"]["memristive_test"]["history"]):
            try:
                if history["label"] == label:
                    return idx
            except KeyError:
                break

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
            train_callbacks: list[callbacks.Callback] = []

            if use_test_callback:
                train_callbacks.append(callbacks.TestCallback(self))

            if self.training.use_combined_validation:
                train_callbacks.append(callbacks.CombinedCheckpoint(self))
            elif not self.training.is_nonideal():
                train_callbacks.append(callbacks.StandardCheckpoint(self))
            else:
                train_callbacks.append(callbacks.MemristiveCheckpoint(self))

            network.train(self, train_callbacks)
            self.training.repeat_idx += 1

        self.training.repeat_idx = 0

    def infer(self) -> None:
        self.is_training = False
        self.compute_power = True
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
        self.compute_power = False
