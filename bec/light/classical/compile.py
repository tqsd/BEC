from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from bec.light.envelopes.compile import compile_envelope as _compile_envelope
from bec.units import Q, magnitude

from bec.light.envelopes.base import Envelope

# if you placed it there

# If you did not place it in base.py, move it into this file and import accordingly.

from .carrier import Carrier
from .drive import ClassicalFieldDrive


@dataclass(frozen=True)
class CompiledDrive:
    """
    Compiled classical drive for solver usage.

    - E_env_V_m(t_solver): float in V/m
    - omega_L_solver(t_solver): optional float dimensionless (omega * time_unit_s)
    - pol: optional length-2 complex array
    """

    E_env_V_m: Callable[[float], float]
    omega_L_solver: Optional[Callable[[float], float]]
    pol: Optional[np.ndarray]
    meta: dict


def compile_carrier_to_solver(
    carrier: Carrier, *, time_unit_s: float
) -> Callable[[float], float]:
    s = float(time_unit_s)

    if not callable(carrier.delta_omega):
        w_const = carrier._omega0_rad_s + float(
            magnitude(carrier.delta_omega, "rad/s")
        )
        w_solver_const = w_const * s

        def omega_solver(_t: float) -> float:
            return w_solver_const

        return omega_solver

    # v1: unitful call per evaluation (slower, but correct)
    def omega_solver(t_solver: float) -> float:
        t_phys_s = s * float(t_solver)
        w = carrier.omega_phys(Q(t_phys_s, "s"))
        return float(magnitude(w, "rad/s")) * s

    return omega_solver


def compile_drive(
    drive: ClassicalFieldDrive, *, time_unit_s: float
) -> CompiledDrive:
    s = float(time_unit_s)

    # returns float given t_solver
    env_solver = _compile_envelope(drive.envelope, time_unit_s=s)
    E0 = drive.amplitude.E0_V_m()

    def E_env_V_m(t_solver: float) -> float:
        return E0 * float(env_solver(float(t_solver)))

    omega_L_solver = None
    if drive.carrier is not None:
        omega_L_solver = compile_carrier_to_solver(drive.carrier, time_unit_s=s)

    pol = drive.effective_pol()

    return CompiledDrive(
        E_env_V_m=E_env_V_m,
        omega_L_solver=omega_L_solver,
        pol=pol,
        meta={"label": drive.label},
    )
