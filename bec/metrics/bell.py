from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def bell_state(which: str) -> np.ndarray:
    """
    Return Bell ket in computational basis [HH, HV, VH, VV].
    """
    w = str(which).lower()
    z = np.zeros((4,), dtype=complex)

    if w in ("phi_plus", "phiplus", "phi+"):
        z[0] = 1.0
        z[3] = 1.0
        return z / np.sqrt(2.0)

    if w in ("phi_minus", "phiminus", "phi-"):
        z[0] = 1.0
        z[3] = -1.0
        return z / np.sqrt(2.0)

    if w in ("psi_plus", "psiplus", "psi+"):
        z[1] = 1.0
        z[2] = 1.0
        return z / np.sqrt(2.0)

    if w in ("psi_minus", "psiminus", "psi-"):
        z[1] = 1.0
        z[2] = -1.0
        return z / np.sqrt(2.0)

    raise ValueError("unknown bell state: %s" % which)


def fidelity_to_bell(rho_2q: np.ndarray, which: str = "phi_plus") -> float:
    """
    Fidelity F = <Bell| rho |Bell> for a 2-qubit polarization state rho.

    Assumes computational basis ordering [HH, HV, VH, VV].
    """
    rho = np.asarray(rho_2q, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("rho_2q must be 4x4")

    ket = bell_state(which)
    proj = np.outer(ket, np.conjugate(ket))
    f = np.real_if_close(np.trace(rho @ proj))
    return float(np.real(f))


@dataclass(frozen=True)
class BellComponent:
    p_hh: float
    p_hv: float
    p_vh: float
    p_vv: float
    parallel: float
    cross: float

    coh_phi_abs: float
    phi_phase_rad: float
    phi_phase_deg: float

    coh_psi_abs: float
    psi_phase_rad: float
    psi_phase_deg: float


def bell_component_from_rho_pol(rho_pol: np.ndarray) -> BellComponent:
    """
    rho_pol is 4x4 in basis [HH, HV, VH, VV] (early, late).

    Returns:
      - populations and parallel/cross weights
      - phi-like coherence (HH <-> VV): rho[0,3]
      - psi-like coherence (HV <-> VH): rho[1,2]
    """
    rho = np.asarray(rho_pol, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError("rho_pol must be 4x4")

    p_hh = float(np.real(rho[0, 0]))
    p_hv = float(np.real(rho[1, 1]))
    p_vh = float(np.real(rho[2, 2]))
    p_vv = float(np.real(rho[3, 3]))

    parallel = float(p_hh + p_vv)
    cross = float(p_hv + p_vh)

    c_phi = complex(rho[0, 3])
    coh_phi_abs = float(abs(c_phi))
    phi_phase_rad = float(np.angle(c_phi))
    phi_phase_deg = float(np.degrees(phi_phase_rad))

    c_psi = complex(rho[1, 2])
    coh_psi_abs = float(abs(c_psi))
    psi_phase_rad = float(np.angle(c_psi))
    psi_phase_deg = float(np.degrees(psi_phase_rad))

    return BellComponent(
        p_hh=p_hh,
        p_hv=p_hv,
        p_vh=p_vh,
        p_vv=p_vv,
        parallel=parallel,
        cross=cross,
        coh_phi_abs=coh_phi_abs,
        phi_phase_rad=phi_phase_rad,
        phi_phase_deg=phi_phase_deg,
        coh_psi_abs=coh_psi_abs,
        psi_phase_rad=psi_phase_rad,
        psi_phase_deg=psi_phase_deg,
    )
