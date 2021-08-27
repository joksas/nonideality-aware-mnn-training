from typing import NamedTuple, List
import os
from . import network, utils


class NonlinearityParams:
    def __init__(self, label: str, G_min: float, G_max: float, n_avg: float, n_std: float):
        self.label = label
        self.G_min = G_min
        self.G_max = G_max
        self.n_avg = n_avg
        self.n_std = n_std

class Nonideal:
    nonlinearity_params_idx = 0

    def __init__(self, nonlinearity_params_lst: List[NonlinearityParams]) -> None:
        self.nonlinearity_params_lst = nonlinearity_params_lst

    def is_memristive(self) -> bool:
        return self.num_nonideality_params() > 0

    def nonideality_params(self):
        return self.nonlinearity_params_lst[self.nonlinearity_params_idx]

    def num_nonideality_params(self):
        return len(self.nonlinearity_params_lst)

    def nonideality_label(self) -> str:
        if len(self.nonlinearity_params_lst) > 0:
            return self.nonlinearity_params_lst[self.nonlinearity_params_idx].label
        else:
            return "ideal"


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
    def __init__(self, batch_size: int = 1, num_epochs: int = 1, nonlinearity_params_lst: List[NonlinearityParams] = [], num_repeats: int = 0) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_repeats = num_repeats
        self.is_regularized = False
        Nonideal.__init__(self, nonlinearity_params_lst)
        Iterable.__init__(self)

    def regularized_label(self) -> str:
        if self.is_regularized:
            return "regularized"
        else:
            return "non-regularized"

    def network_label(self):
        return "network-{}".format(self.repeat_idx)


class Inference(Nonideal):
    def __init__(self, nonlinearity_params_lst: List[NonlinearityParams] = [], num_repeats: int = 0) -> None:
        self.nonlinearity_params_lst = nonlinearity_params_lst
        self.num_repeats = num_repeats
        Nonideal.__init__(self, nonlinearity_params_lst)
        Iterable.__init__(self)

    def repeat_label(self):
        return "repeat-{}".format(self.repeat_idx)


class Iterator(Dataset):
    def __init__(self, dataset: str, training: Training = Training(), inference: Inference = Inference()) -> None:
        self.dataset = dataset
        self.training = training
        self.inference = inference
        Dataset.__init__(self, dataset)

    def training_nonideality_dir(self):
        return os.path.join(
                os.getcwd(), "new-models", self.dataset, self.training.regularized_label(), self.training.nonideality_label()
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

    def Train(self):
        if self.training.is_memristive():
            for _ in range(len(self.training.nonlinearity_params_lst)):
                for _ in range(self.training.num_repeats):
                    network.train(self)
                    self.training.repeat_idx += 1

                self.training.repeat_idx = 0
                self.training.nonlinearity_params_idx += 1

            self.training.nonlinearity_params_idx = 0
        else:
            for _ in range(self.training.num_repeats):
                print(self.network_dir())
                os.makedirs(self.network_dir(), exist_ok=True)
                self.training.repeat_idx += 1

            self.training.repeat_idx = 0

