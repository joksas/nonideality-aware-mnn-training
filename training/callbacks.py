import copy
import tensorflow as tf
import numpy as np
from . import architecture


class MemristiveCallback(tf.keras.callbacks.Callback):
    def __init__(self, iterator):
        self.iterator = copy.deepcopy(iterator)
        self.iterator.is_callback = True
        self.iterator.is_training = False
        self.every = 20
        self.num_repeats = 20
        self.history = None
        self.reset_history()

    def reset_history(self):
        self.history = [{
            "nonideality_label": inference.nonideality_label(),
            "epoch_no": [],
            "loss": [],
            "accuracy": []
            } for inference in self.iterator.inferences]


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

    def on_epoch_end(self, epoch, logs=None):
        if self.should_skip_epoch(epoch):
            return

        model_weights = self.model.get_weights()

        for inference_idx, inference in enumerate(self.iterator.inferences):
            self.iterator.inference_idx = inference_idx
            callback_model = architecture.get_model(self.iterator, custom_weights=model_weights)
            accuracy = []
            loss = []
            for _ in range(self.num_repeats):
                score = callback_model.evaluate(self.iterator.x_test, self.iterator.y_test, verbose=0, batch_size=128)
                loss.append(score[0])
                accuracy.append(score[1])
            self.history[inference_idx]["loss"].append(loss)
            self.history[inference_idx]["accuracy"].append(accuracy)
            self.history[inference_idx]["epoch_no"].append(epoch+1)
            print(
                    inference.nonideality_label(),
                    "median loss:", f"{np.median(loss):.4f}",
                    "median accuracy:", f"{np.median(accuracy):.4f}",
                    )

        self.iterator.inference_idx = None

    def name(self):
        return "memristive_test"

