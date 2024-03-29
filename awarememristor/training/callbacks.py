import copy
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import tensorflow as tf

from awarememristor.training import architecture


class Callback(ABC):
    """Abstract class that requires implementation of callback name."""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Returns name of the callback."""


class Checkpoint:
    """Used only to mark certain callbacks as checkpoint callbacks."""


class MemristiveCallback(tf.keras.callbacks.Callback):
    """Computes a metric multiple times in order to take the stochastic nature
    of memristive devices into account."""

    def __init__(self, iterator, history=None) -> None:
        self.iterator = copy.copy(iterator)
        self.validation_freq = 20
        if iterator.training.memristive_validation_freq is not None:
            self.validation_freq = iterator.training.memristive_validation_freq
        self.testing_freq = 20
        self.num_repeats = 20
        self.history = history

    def should_skip_epoch(self, epoch, is_validation=False) -> bool:
        freq = self.testing_freq
        if is_validation:
            freq = self.validation_freq
        # Will evaluate on first epoch and then every `freq` epochs.
        if epoch != 0 and (epoch + 1) % freq != 0:
            return True
        return False

    def evaluate(self, model, data, num_repeats: int = None):
        if num_repeats is None:
            num_repeats = self.num_repeats

        accuracy = []
        loss = []

        start_time = time.time()
        for _ in range(num_repeats):
            single_loss, single_accuracy = model.evaluate(data, verbose=0)
            loss.append(single_loss)
            accuracy.append(single_accuracy)
        num_total_batches = data.cardinality().numpy() * num_repeats

        end_time = time.time()
        duration = int(end_time - start_time)
        if num_repeats == 1:
            loss = loss[0]
            accuracy = accuracy[0]

        return loss, accuracy, duration, num_total_batches

    def evaluation_results_str(
        self,
        num_total_batches: int,
        duration: int,
        loss: float,
        accuracy: float,
        prepend_metrics: str = None,
        label: str = None,
    ) -> str:
        str_ = f"{num_total_batches}/{num_total_batches}"
        str_ += f" - {duration}s - "
        if prepend_metrics:
            str_ += f"{prepend_metrics}_"
        str_ += f"loss: {loss:.4f} - "
        if prepend_metrics:
            str_ += f"{prepend_metrics}_"
        str_ += f"accuracy: {accuracy:.4f}"
        if label is not None:
            str_ += f" [{label}]"

        return str_

    def saving_weights_str(
        self, accuracy: float, previous_best_accuracy: float, prepend: str = None
    ) -> str:
        str_ = ""
        if prepend is not None:
            str_ += f"{prepend}_"
        str_ += f"accuracy ({accuracy:.4f}) improved over previous best result ({previous_best_accuracy:.4f}). Saving weights..."
        return str_

    def info(self) -> dict[str, Any]:
        return {
            "history": self.history,
        }


class TestCallback(MemristiveCallback, Callback):
    """Compute test accuracy for all inference setups during training."""

    def __init__(self, iterator) -> None:
        MemristiveCallback.__init__(self, iterator)
        self.iterator.is_training = False
        self.history = [
            {
                "label": inference.label(),
                "epoch_no": [],
                "loss": [],
                "accuracy": [],
            }
            for inference in self.iterator.inferences
        ]

    def on_epoch_end(self, epoch, logs=None):
        if self.should_skip_epoch(epoch, is_validation=False):
            return

        model_weights = self.model.get_weights()

        for inference_idx, inference in enumerate(self.iterator.inferences):
            self.iterator.inference_idx = inference_idx
            data = self.iterator.data("testing")
            callback_model = architecture.get_model(self.iterator, custom_weights=model_weights)
            loss, accuracy, duration, num_total_batches = self.evaluate(callback_model, data)
            self.history[inference_idx]["loss"].append(loss)
            self.history[inference_idx]["accuracy"].append(accuracy)
            self.history[inference_idx]["epoch_no"].append(epoch + 1)
            results_str = self.evaluation_results_str(
                num_total_batches,
                duration,
                np.median(loss),
                np.median(accuracy),
                prepend_metrics="median_test",
                label=inference.nonideality_label(),
            )
            print(results_str)

        self.iterator.inference_idx = 0

    @staticmethod
    def name():
        return "memristive_test"


class MemristiveCheckpoint(MemristiveCallback, Callback, Checkpoint):
    """Evaluate accuracy on validation set multiple times to provide a more reliable measure of
    learning progress.
    """

    def __init__(self, iterator) -> None:
        MemristiveCallback.__init__(self, iterator)
        self.iterator.is_training = True
        self.best_median_val_accuracy = 0.0
        self.history = {"epoch_no": [], "loss": [], "accuracy": []}

    def on_epoch_end(self, epoch, logs=None):
        if self.should_skip_epoch(epoch, is_validation=True):
            return

        data = self.iterator.data("validation")
        loss, accuracy, duration, num_total_batches = self.evaluate(
            self.model,
            data,
            num_repeats=self.iterator.training.memristive_validation_num_repeats,
        )

        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)
        self.history["epoch_no"].append(epoch + 1)

        median_val_loss = np.median(loss)
        median_val_accuracy = np.median(accuracy)
        print(
            self.evaluation_results_str(
                num_total_batches,
                duration,
                median_val_loss,
                median_val_accuracy,
                prepend_metrics="median_val",
            )
        )

        if median_val_accuracy > self.best_median_val_accuracy:
            print(
                self.saving_weights_str(
                    median_val_accuracy, self.best_median_val_accuracy, prepend="median_val"
                )
            )
            self.best_median_val_accuracy = median_val_accuracy
            self.model.save_weights(self.iterator.weights_path())

    @staticmethod
    def name():
        return "memristive_checkpoint"


class StandardCheckpoint(tf.keras.callbacks.ModelCheckpoint, Callback, Checkpoint):
    """Same as `tf.keras.callbacks.ModelCheckpoint`, but with a `name()`."""

    def __init__(self, iterator) -> None:
        tf.keras.callbacks.ModelCheckpoint.__init__(
            self,
            iterator.weights_path(),
            monitor="val_accuracy",
            save_best_only=True,
        )

    @staticmethod
    def name():
        return "standard_checkpoint"


class CombinedCheckpoint(MemristiveCallback, Callback, Checkpoint):
    """Used to test the effectiveness of memristive validation.

    Two validation techniques (standard and memristive) are applied at the same
    time during training.
    """

    def __init__(self, iterator) -> None:
        MemristiveCallback.__init__(self, iterator)
        self.iterator.is_training = True
        self.best_median_val_accuracy = 0.0
        self.best_standard_val_accuracy = 0.0
        self.history = {
            "epoch_no": [],
            "loss": [],
            "accuracy": [],
            "standard_epoch_no": [],
            "standard_loss": [],
            "standard_accuracy": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        data = self.iterator.data("validation")

        if self.should_skip_epoch(epoch, is_validation=True):
            single_loss, single_accuracy, duration, num_total_batches = self.evaluate(
                self.model,
                data,
                num_repeats=1,
            )
            self.history["standard_loss"].append(single_loss)
            self.history["standard_accuracy"].append(single_accuracy)
            self.history["standard_epoch_no"].append(epoch + 1)
            print(
                self.evaluation_results_str(
                    num_total_batches, duration, single_loss, single_accuracy, prepend_metrics="val"
                )
            )
        else:
            loss, accuracy, duration, num_total_batches = self.evaluate(
                self.model,
                data,
                num_repeats=self.iterator.training.memristive_validation_num_repeats,
            )

            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)
            self.history["epoch_no"].append(epoch + 1)

            median_val_loss = np.median(loss)
            median_val_accuracy = np.median(accuracy)
            print(
                self.evaluation_results_str(
                    num_total_batches,
                    duration,
                    median_val_loss,
                    median_val_accuracy,
                    prepend_metrics="median_val",
                )
            )

            single_loss = loss[0]
            single_accuracy = accuracy[0]
            self.history["standard_loss"].append(single_loss)
            self.history["standard_accuracy"].append(single_accuracy)
            self.history["standard_epoch_no"].append(epoch + 1)
            print(
                self.evaluation_results_str(
                    0, 0, single_loss, single_accuracy, prepend_metrics="val"
                )
            )

            if median_val_accuracy > self.best_median_val_accuracy:
                self.iterator.training.is_standard_validation_mode = True
                os.makedirs(self.iterator.weights_dir(), exist_ok=True)
                print(
                    self.saving_weights_str(
                        median_val_accuracy, self.best_median_val_accuracy, prepend="median_val"
                    )
                )
                self.best_median_val_accuracy = median_val_accuracy
                self.model.save_weights(self.iterator.weights_path())

        if single_accuracy > self.best_standard_val_accuracy:
            self.iterator.training.is_standard_validation_mode = False
            os.makedirs(self.iterator.weights_dir(), exist_ok=True)
            print(
                self.saving_weights_str(
                    single_accuracy, self.best_standard_val_accuracy, prepend="val"
                )
            )
            self.best_standard_val_accuracy = single_accuracy
            self.model.save_weights(self.iterator.weights_path())

    @staticmethod
    def name():
        return "combined_checkpoint"
