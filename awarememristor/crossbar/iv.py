import numpy as np
import numpy.typing as npt
import scipy.constants as const


def qpc_model(
    V: npt.NDArray[np.float64], N: int, alpha: float, phi_eff: float
) -> npt.NDArray[np.float64]:
    """Calculates currents using quantum point contact theory.

    See [http://dx.doi.org/10.1063/1.4836935](doi:10.1063/1.4836935) for
    details.

    Args:
        V: Voltages.
        N: Conductance multiplier.
        alpha: Constant related to longitudinal shape of the barrier.
        phi_eff: Effective barrier height.

    Returns
        Currents.
    """
    return const.physical_constants["conductance quantum"][0] * (
        N * V
        + 2
        / (const.elementary_charge * alpha)
        * np.exp(-alpha * phi_eff)
        * np.sinh(const.elementary_charge * alpha * V / 2)
    )


def pf_model(V: npt.NDArray[np.float64], c: float, d_times_perm: float) -> npt.NDArray[np.float64]:
    """Calculates currents according to Poole-Frenkel model.

    Args:
        V: Voltages.
        c: Scaling factor.
        d_times_perm: The product of thickness and permittivity.

    Returns:
        Currents.
    """
    return (
        c
        * V
        * np.exp(
            const.elementary_charge
            * np.sqrt(const.elementary_charge * V / (const.pi * d_times_perm))
            / (const.Boltzmann * (const.zero_Celsius + 20))
        )
    )


def thermionic_model(
    V: npt.NDArray[np.float64], c: float, d_times_perm: float
) -> npt.NDArray[np.float64]:
    """Calculates currents according to thermionic emission model.

    Args:
        V: Voltages.
        c: Scaling factor.
        d_times_perm: The product of thickness and permittivity.

    Returns:
        Currents.
    """
    return c * np.exp(
        const.elementary_charge
        * np.sqrt(const.elementary_charge * V / (4 * const.pi * d_times_perm))
        / (const.Boltzmann * (const.zero_Celsius + 20))
    )
