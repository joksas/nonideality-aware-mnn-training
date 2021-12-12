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


def get_energy_efficiency(
    avg_power: float,
    num_neurons_lst: list[int] = [784, 25, 10],
    read_time: float = 50e-9,
):
    num_synapses = get_num_synapses(num_neurons_lst)
    energy_efficiency = (2 * num_synapses) / (read_time * avg_power)
    return energy_efficiency


def get_num_synapses(num_neurons_lst: list[int]):
    num_synapses = 0
    for idx, num_neurons in enumerate(num_neurons_lst[:-1]):
        num_synapses += (num_neurons + 1) * num_neurons_lst[idx + 1]

    return num_synapses
