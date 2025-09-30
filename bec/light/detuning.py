import numpy as np
from scipy.constants import e as _e, hbar as _hbar


def two_photon_detuning_profile(EL, drive, time_unit_s: float):
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
    t = getattr(drive, "_cached_tlist", None)
    wXXG = float(EL.XX) * _e / _hbar  # rad/s

    if getattr(drive, "laser_omega", None) is not None:
        Delta_2g = 2.0 * float(drive.laser_omega) - wXXG
    else:
        Delta_2g = (
            float(drive.detuning) if not callable(drive.detuning) else 0.0
        )

    kappa = getattr(drive, "stark_kappa", 0.0)

    if t is not None and kappa:
        Om1_sq = np.array(
            [drive.envelope(time_unit_s * ti) ** 2 for ti in t], float
        )
        Delta_eff_phys = Delta_2g + kappa * Om1_sq  # rad/s
        return t, Delta_eff_phys * time_unit_s  # solver units
    elif t is not None:
        return t, np.full_like(t, Delta_2g * time_unit_s, dtype=float)
    else:
        return None, None
