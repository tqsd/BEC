from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from bec.units import Q, as_quantity, magnitude
from .drives import ClassicalCoherentDrive
from .amplitude import RabiAmplitude, FieldAmplitude

SolverFn = Callable[[float, dict], float]  # QuTiP style (t, args) -> float


@dataclass(frozen=True)
class CompiledDrive:
    # dimensionless Ω(t_solver) (already multiplied by time_unit_s)
    omega_solver: Callable[[float], complex]
    # dimensionless ω_L(t_solver) if carrier present (also * time_unit_s)
    laser_omega_solver: Optional[Callable[[float], float]]
    # polarization vector (unitless)
    pol: Optional[np.ndarray]
    meta: dict


def compile_drive(
    drv: ClassicalCoherentDrive,
    *,
    time_unit_s: float,
    field_to_rabi: Optional[Callable[[complex], complex]] = None,
) -> CompiledDrive:
    s = float(time_unit_s)

    pol = drv.effective_polarization()

    def env_val(t_solver: float) -> float:
        t_phys = Q(s * float(t_solver), "s")
        return float(drv.envelope(t_phys))

    if isinstance(drv.amplitude, RabiAmplitude):
        Om0 = magnitude(drv.amplitude.omega0, "rad/s")

        def omega_solver(t_solver: float) -> complex:
            # Ω_phys * s -> dimensionless
            return complex(Om0 * env_val(t_solver) * s)

    else:
        if field_to_rabi is None:
            raise ValueError(
                "FieldAmplitude drive requires field_to_rabi conversion."
            )
        E0 = magnitude(drv.amplitude.E0, "V/m")

        def omega_solver(t_solver: float) -> complex:
            t_phys = Q(s * float(t_solver), "s")
            E = complex(E0 * float(drv.envelope(t_phys)))  # V/m
            Om = field_to_rabi(E)  # rad/s
            return complex(Om * s)  # dimensionless

    laser_omega_solver = None
    if drv.carrier is not None:

        def laser_omega_solver(t_solver: float) -> float:
            t_phys = Q(s * float(t_solver), "s")
            w = drv.carrier.omega(t_phys)  # rad/s
            return float(magnitude(w, "rad/s") * s)  # dimensionless

    return CompiledDrive(
        omega_solver=omega_solver,
        laser_omega_solver=laser_omega_solver,
        pol=pol,
        meta={"label": drv.label},
    )
