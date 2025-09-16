import numpy as np
from scipy.constants import e as _e, hbar as _hbar


def two_photon_detuning_profile(EL, drive, time_unit_s: float):
    # time grid (solver units)
    t = getattr(drive, "_cached_tlist", None)
    # two-photon resonance frequency from levels (G=0, so just XX):
    wXXG = float(EL.XX) * _e / _hbar  # rad/s

    if getattr(drive, "laser_omega", None) is not None:
        # compute Δ from user-given laser frequency
        Delta_2g = 2.0 * float(drive.laser_omega) - wXXG
    else:
        # fall back to drive.detuning if user set it explicitly
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
