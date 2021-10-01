import os
import pickle
from . import network, utils


class IVNonlinearity:
    def __init__(self, n_avg: float, n_std: float):
        self.n_avg = n_avg
        self.n_std = n_std

    def label(self):
        return f"IVNL:{self.n_avg:.3g}_{self.n_std:.3g}"


class Stuck:
    def __init__(self, p: float):
        self.p = p


class StuckAtGMin(Stuck):
    def __init__(self, p: float):
        Stuck.__init__(self, p)

    def label(self) -> str:
        return f"StuckMin:{self.p:.3g}"


class StuckAtGMax(Stuck):
    def __init__(self, p: float):
        Stuck.__init__(self, p)

    def label(self) -> str:
        return f"StuckMax:{self.p:.3g}"


class Nonideal:
    def __init__(self,
            iv_nonlinearity: IVNonlinearity = None,
            stuck_at_G_min: StuckAtGMin = None,
            stuck_at_G_max: StuckAtGMax  = None,
            ) -> None:
        self.iv_nonlinearity = iv_nonlinearity
        self.stuck_at_G_min = stuck_at_G_min
        self.stuck_at_G_max = stuck_at_G_max

    def nonideality_list(self):
        nonidealities = []
        for nonideality in [self.iv_nonlinearity, self.stuck_at_G_min, self.stuck_at_G_max]:
            if nonideality is not None:
                nonidealities.append(nonideality)

        return nonidealities

    def nonideality_label(self) -> str:
        nonidealities = self.nonideality_list()
        if len(nonidealities) == 0:
            return "ideal"

        return "+".join(nonideality.label() for nonideality in nonidealities)

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
    def __init__(self, batch_size: int = 1, num_epochs: int = 1, is_regularized: bool = False,
            num_repeats: int = 0, nonidealities={}) -> None:
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_repeats = num_repeats
        self.is_regularized = is_regularized
        Nonideal.__init__(self, **nonidealities)
        Iterable.__init__(self)

    def regularized_label(self) -> str:
        if self.is_regularized:
            return "regularized"
        else:
            return "non-regularized"

    def is_aware(self) -> bool:
        return self.is_nonideal()

    def network_label(self):
        return "network-{}".format(self.repeat_idx)


class Inference(Nonideal, Iterable):
    def __init__(self, num_repeats: int = 0, nonidealities={}) -> None:
        self.num_repeats = num_repeats
        Nonideal.__init__(self, **nonidealities)
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

    def conductance_label(self):
        return f"{self.G_min:.3g}_{self.G_max:.3g}"

    def training_nonideality_dir(self):
        return os.path.join(
                os.getcwd(), "models", self.dataset, self.training.regularized_label(),
                self.conductance_label(), self.training.nonideality_label()
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
                self.network_dir(), self.inference.nonideality_label()
                )

    def inference_repeat_dir(self):
        return os.path.join(
                self.inference_nonideality_dir(), self.inference.repeat_label()
                )

    def power_path(self):
        return os.path.join(
                self.inference_repeat_dir(), "power.csv"
                )

    def loss_path(self):
        return os.path.join(
                self.inference_repeat_dir(), "loss.csv"
                )
    def accuracy_path(self):
        return os.path.join(
                self.inference_repeat_dir(), "accuracy.csv"
                )

    def history(self):
        with open(self.history_path(), "rb") as pickle_file:
            return pickle.load(pickle_file)

    def current_stage(self) -> bool:
        if self.is_training:
            return self.training
        else:
            return self.inference

    def train(self):
        self.is_training = True
        for _ in range(self.training.num_repeats):
            network.train(self)
            self.training.repeat_idx += 1

        self.training.repeat_idx = 0

    def infer(self):
        self.is_training = False
        for _ in range(self.training.num_repeats):
            for _ in range(self.inference.num_repeats):
                network.infer(self)
                self.inference.repeat_idx += 1

            self.inference.repeat_idx = 0
            self.training.repeat_idx += 1

        self.training.repeat_idx = 0
