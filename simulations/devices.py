from training.iterator import IVNonlinearity


def ideal():
    return {
            "G_min": None,
            "G_max": None,
            "nonidealities": {}
            }


def low_R():
    return {
            "G_min": 1/1003,
            "G_max": 1/284.6,
            "nonidealities": {"iv_nonlinearity": IVNonlinearity(2.132, 0.095)}
            }


def high_R():
    return {
            "G_min": 1/1295000,
            "G_max": 1/366200,
            "nonidealities": {"iv_nonlinearity": IVNonlinearity(2.989, 0.369)}
            }
