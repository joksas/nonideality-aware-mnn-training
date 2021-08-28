from typing import NamedTuple, List
import os
from . import network, utils


class IVNonlinearity:
    def __init__(self, label: str, n_avg: float, n_std: float):
        self.label = label
        self.n_avg = n_avg
        self.n_std = n_std


class Nonideal:
    def __init__(self, iv_nonlinearity: IVNonlinearity = None) -> None:
        self.iv_nonlinearity = iv_nonlinearity

    def nonideality_label(self) -> str:
        if self.iv_nonlinearity is not None:
            return self.iv_nonlinearity.label
        else:
            return "ideal"

    def is_nonideal(self) -> bool:
        if self.iv_nonlinearity is not None:
            return True
        else:
            return False


class Iterable:
    repeat_idx = 0


class Dataset:
    def __init__(self, dataset: str):
        x_train, y_train, x_test, y_test, use_generator = utils.get_examples(dataset)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.use_generator = use_generator


class Training(Nonideal, Iterable):
    def __init__(self, batch_size: int = 1, num_epochs: int = 1, is_regularized: bool = False, iv_nonlinearity: IVNonlinearity = None, num_repeats: int = 0) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_repeats = num_repeats
        self.is_regularized = is_regularized
        Nonideal.__init__(self, iv_nonlinearity)
        Iterable.__init__(self)

    def regularized_label(self) -> str:
        if self.is_regularized:
            return "regularized"
        else:
            return "non-regularized"

    def network_label(self):
        return "network-{}".format(self.repeat_idx)


class Inference(Nonideal):
    def __init__(self, iv_nonlinearity: IVNonlinearity = None, num_repeats: int = 0) -> None:
        self.num_repeats = num_repeats
        Nonideal.__init__(self, iv_nonlinearity)
        Iterable.__init__(self)

    def repeat_label(self):
        return "repeat-{}".format(self.repeat_idx)


class Iterator(Dataset):
    is_training = False

    def __init__(self, dataset: str, G_min: float, G_max: float, training: Training, inference: Inference = None) -> None:
        self.dataset = dataset
        self.G_min = G_min
        self.G_max = G_max
        self.training = training
        self.inference = inference
        Dataset.__init__(self, dataset)

    def training_nonideality_dir(self):
        return os.path.join(
                os.getcwd(), "models", self.dataset, self.training.regularized_label(), self.training.nonideality_label()
                )

    def network_dir(self):
        return os.path.join(
                self.training_nonideality_dir(), self.training.network_label()
                )

    def weights_path(self):
        return os.path.join(
                self.network_dir(), "model.h5"
                )

    def history_path(self):
        return os.path.join(
                self.network_dir(), "history.pkl"
                )

    def inference_nonideality_dir(self):
        return os.path.join(
                network_dir, self.inference.nonideality_label()
                )

    def power_path(self):
        return os.path.join(
                self.network_dir(), "power.csv"
                )

    def current_stage(self) -> bool:
        if self.is_training:
            return self.training
        else:
            return self.inference

    def Train(self):
        self.is_training = True
        for _ in range(self.training.num_repeats):
            os.makedirs(self.network_dir(), exist_ok=True)
            network.train(self)
            self.training.repeat_idx += 1

        self.training.repeat_idx = 0

