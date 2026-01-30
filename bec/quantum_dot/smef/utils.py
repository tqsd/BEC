import numpy as np


def _exciton_theta_rad_from_qd(qd) -> float:
    mp = getattr(qd, "mixing", None)
    if mp is None:
        return 0.0

    # fss = E_X1 - E_X2
    fss_eV = float(qd.energy.fss.to("eV").magnitude)

    dp = getattr(mp, "delta_prime", 0.0)
    try:
        dp_eV = float(dp.to("eV").magnitude)
    except Exception:
        dp_eV = float(dp)

    # theta = 0.5 * atan2(2*delta_prime, fss)
    return 0.5 * float(np.arctan2(2.0 * dp_eV, fss_eV))
