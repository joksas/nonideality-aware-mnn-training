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
