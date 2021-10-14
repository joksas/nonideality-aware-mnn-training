import copy
import time
import tensorflow as tf
import numpy as np
from . import architecture


class MemristiveCallback(tf.keras.callbacks.Callback):
    def __init__(self, iterator):
        self.iterator = copy.deepcopy(iterator)
        self.iterator.is_callback = True
        self.every = 20
        self.num_repeats = 20
        self.history = None

    def should_skip_epoch(self, epoch):
        # Will evaluate on first epoch and then every `self.every` epochs.
        if epoch != 0 and (epoch+1)%self.every != 0:
            return True


    def info(self):
        return {
                "history": self.history,
                }


class TestCallback(MemristiveCallback):
    """Computes test accuracy for all inference setups during training.
    """
    def __init__(self, iterator):
        MemristiveCallback.__init__(self, iterator)
        self.iterator.is_training = False
        self.reset_history()

    def reset_history(self):
        self.history = [{
            "nonideality_label": inference.nonideality_label(),
            "epoch_no": [],
            "loss": [],
            "accuracy": []
            } for inference in self.iterator.inferences]

    def on_epoch_end(self, epoch, logs=None):
        if self.should_skip_epoch(epoch):
            return

        model_weights = self.model.get_weights()

        for inference_idx, inference in enumerate(self.iterator.inferences):
            start_time = time.time()
            self.iterator.inference_idx = inference_idx
            callback_model = architecture.get_model(self.iterator, custom_weights=model_weights)
            accuracy = []
            loss = []
            data = self.iterator.data("testing")
            for _ in range(self.num_repeats):
                score = callback_model.evaluate(data, verbose=0)
                loss.append(score[0])
                accuracy.append(score[1])
            self.history[inference_idx]["loss"].append(loss)
            self.history[inference_idx]["accuracy"].append(accuracy)
            self.history[inference_idx]["epoch_no"].append(epoch+1)
            num_total_batches = data.cardinality().numpy() * self.num_repeats
            end_time = time.time()
            duration = int(end_time - start_time)
            print(f"{num_total_batches}/{num_total_batches} - "\
                    f"{duration}s - "\
                    f"median_test_loss: {np.median(loss):.4f} - "\
                    f"median_test_accuracy: {np.median(accuracy):.4f} "\
                    f"[{inference.nonideality_label()}]")

        self.iterator.inference_idx = None

    def name(self):
        return "memristive_test"


class MemristiveCheckpoint(MemristiveCallback):
    """Evaluates accuracy on validation set multiple times to provide a more reliable measure of
    learning progress.
    """
    def __init__(self, iterator):
        MemristiveCallback.__init__(self, iterator)
        self.iterator.is_training = True
        self.reset_history()
        self.best_median_val_accuracy = 0.0

    def reset_history(self):
        self.history = {
                "epoch_no": [],
                "loss": [],
                "accuracy": []
                }

    def on_epoch_end(self, epoch, logs=None):
        if self.should_skip_epoch(epoch):
            return

        accuracy = []
        loss = []
        start_time = time.time()
        data = self.iterator.data("validation")
        for _ in range(self.num_repeats):
            score = self.model.evaluate(data, verbose=0)
            loss.append(score[0])
            accuracy.append(score[1])
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)
        self.history["epoch_no"].append(epoch+1)

        median_val_accuracy = np.median(accuracy)
        median_val_loss = np.median(loss)
        num_total_batches = data.cardinality().numpy() * self.num_repeats
        end_time = time.time()
        duration = int(end_time - start_time)
        print(f"{num_total_batches}/{num_total_batches} - "\
                f"{duration}s - "\
                f"median_val_loss: {np.median(loss):.4f} - "\
                f"median_val_accuracy: {np.median(accuracy):.4f}")
        if median_val_accuracy > self.best_median_val_accuracy:
            print(f"median_val_accuracy ({median_val_accuracy:.4f}) improved over "\
                    f"previous best result ({self.best_median_val_accuracy:.4f}). Saving weights...")
            self.best_median_val_accuracy = median_val_accuracy
            self.model.save_weights(self.iterator.weights_path())

    def name(self):
        return "memristive_checkpoint"


class RegularCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, iterator):
        tf.keras.callbacks.ModelCheckpoint.__init__(
                self,
                iterator.weights_path(),
                monitor="val_accuracy",
                save_best_only=True,
                )

    def name(self):
        return "regular_checkpoint"
