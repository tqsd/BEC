from typing import Callable, Union, Optional, Tuple, cast
import numpy as np
from scipy.constants import e as _e, hbar as _hbar

from bec.light.classical import ClassicalTwoPhotonDrive
from bec.params.energy_levels import EnergyLevels


DetuningFn = Callable[[float], float]
Detuning = Union[float, DetuningFn]


def tanh_chirp(t0: float, Delta_max: float, tau: float):
    """
    Returns a detuning function in rad/s.
    Crosses zero at t0; asymptotes +_ Delta_max.
    Tau controls how fast the sweep happens around ~t0
    """

    def det(t_phys: float) -> float:
        return float(Delta_max * np.tanh((t_phys - t0) / tau))

    return det


def two_photon_detuning_profile(
    EL: EnergyLevels, drive: ClassicalTwoPhotonDrive, time_unit_s: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Computes the two photon detuning profile for a given drive and
    Quantum dot EL parameters.

    This function evaluates the effective two-photon detuning
    between |XX> and |G>. The resonance frequency is derived
    from the energy difference `EL.XX`.

    Behavior:
    ---------
    - If `drive.laser_omega` is provided, the detuning is
      computed as `Delta = 2*laser_omega - wXXG`
    - otherwise it falls back to `drive.detuning` if it is a float
    - If the drive has cached time grid, the detuning is returned
      as array on the grid.
    - If `drive.stark_kappa` is set, it multiplies the squared
      envelope and shifts the detuning.

    Arguments:
    ----------
    EL: EnergyLevels
        Energy levels parameters
    drive: TwoPhotonClassicalDrive
        The drive definition
    time_unit_s: float
        The solver time unit

    Return:
    -------
    tuple[np.ndarray|None, np.ndarray|None]
        Solver time grid, and detuning values in solver units.

    Notes:
    ------
    Kappa is included here but not actually used in this model, so this term
    should be considered inactive.
    """
    t = drive._cached_tlist
    if t.size == 0:
        return None, None

    wXXG = float(EL.XX) * _e / _hbar  # rad/s

    base = 0.0
    if drive.laser_omega is not None:
        base = 2.0 * float(drive.laser_omega) - wXXG

    det: Detuning = drive.detuning
    if callable(det):
        det_fn = cast(DetuningFn, det)
        det_t = np.array(
            [float(det_fn(time_unit_s * ti)) for ti in t], dtype=float
        )
    else:
        det_t = np.full_like(t, float(det), dtype=float)

    kappa = float(getattr(drive, "stark_kappa", 0.0))

    if kappa != 0.0:
        Om_sq = np.array(
            [float(drive.envelope(time_unit_s * ti)) ** 2 for ti in t],
            dtype=float,
        )
        det_t = det_t + kappa * Om_sq

    Delta_eff_phys = base + det_t  # rad/s
    return t, Delta_eff_phys * time_unit_s
