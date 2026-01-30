from __future__ import annotations

import numpy as np


def bell_state(which: str) -> np.ndarray:
    """
    Return Bell ket in computational basis [HH, HV, VH, VV].
    """
    which = str(which).lower()
    z = np.zeros((4,), dtype=complex)
    if which in ("phi_plus", "phiplus", "phi+"):
        # (|HH> + |VV>)/sqrt(2)
        z[0] = 1.0
        z[3] = 1.0
        return z / np.sqrt(2.0)
    if which in ("phi_minus", "phiminus", "phi-"):
        z[0] = 1.0
        z[3] = -1.0
        return z / np.sqrt(2.0)
    if which in ("psi_plus", "psiplus", "psi+"):
        z[1] = 1.0
        z[2] = 1.0
        return z / np.sqrt(2.0)
    if which in ("psi_minus", "psiminus", "psi-"):
        z[1] = 1.0
        z[2] = -1.0
        return z / np.sqrt(2.0)
    raise ValueError("unknown bell state: %s" % which)


def fidelity_to_bell(rho_2q: np.ndarray, which: str = "phi_plus") -> float:
    rho_2q = np.asarray(rho_2q, dtype=complex)
    if rho_2q.shape != (4, 4):
        raise ValueError("rho_2q must be 4x4")
    ket = bell_state(which)
    proj = np.outer(ket, np.conjugate(ket))
    f = np.real_if_close(np.trace(rho_2q @ proj))
    return float(f)
