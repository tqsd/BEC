from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DriveSeries:
    label: str
    t_s: np.ndarray
    E_env_V_m: Optional[np.ndarray] = None
    omega_L_rad_s: Optional[np.ndarray] = None
    delta_omega_rad_s: Optional[np.ndarray] = None
    Omega_rad_s: Optional[np.ndarray] = None  # drive-implied mu*E/hbar

    # Optional, extracted from carrier.omega0 when available:
    wavelength_nm: Optional[float] = None


@dataclass(frozen=True)
class QDTraces:
    t_solver: np.ndarray
    t_s: np.ndarray
    time_unit_s: float

    pops: Mapping[str, np.ndarray] = field(default_factory=dict)
    outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    coherences: Mapping[str, np.ndarray] = field(default_factory=dict)

    drives: Sequence[DriveSeries] = field(default_factory=tuple)

    # First-class wavelength metadata (nm), computed from qd.energy transition energies.
    # Keys are stable strings:
    #   GX_X1, GX_X2, GX_center
    #   XX_X1, XX_X2, XX_center
    output_wavelengths_nm: Mapping[str, float] = field(default_factory=dict)

    # Optional debugging / reporting extras:
    #   GX_X1, GX_X2, GX_center, XX_X1, XX_X2, XX_center
    output_transition_energies_eV: Mapping[str, float] = field(
        default_factory=dict
    )

    # Optional: central wavelength per drive label (if you want it even after
    # embedding wavelength into the label string).
    drive_wavelengths_nm: Mapping[str, float] = field(default_factory=dict)

    extra: Mapping[str, np.ndarray] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
