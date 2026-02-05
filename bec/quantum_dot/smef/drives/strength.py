from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from smef.core.drives.protocols import (
    DriveDecodeContextProto,
    DriveStrengthModelProto,
)
from smef.core.drives.types import DriveCoefficients, ResolvedDrive
from smef.core.units import hbar, magnitude

from bec.quantum_dot.enums import TransitionPair
from bec.quantum_dot.smef.drives.context import QDDriveDecodeContext


def _payload_from_ctx(drive_id: Any, ctx: QDDriveDecodeContext) -> Any:
    if drive_id in ctx.meta_drives:
        return ctx.meta_drives[drive_id]
    raise KeyError("Missing drive payload for drive_id=%s" % (drive_id,))


def _effective_pol(payload: Any) -> np.ndarray | None:
    fn = getattr(payload, "effective_pol", None)
    if not callable(fn):
        return None
    out = fn()
    if out is None:
        return None
    return np.asarray(out, dtype=complex).reshape(2)


def _sample_E_env_V_m(payload: Any, t_phys_s: np.ndarray) -> np.ndarray:
    fn = getattr(payload, "E_env_V_m", None)
    if not callable(fn):
        raise AttributeError("Drive payload must provide E_env_V_m(t_phys_s)")
    out = np.empty(t_phys_s.size, dtype=float)
    for i in range(t_phys_s.size):
        out[i] = float(fn(float(t_phys_s[i])))
    return out


@dataclass
class QDDriveStrengthModel(DriveStrengthModelProto):
    """
    Produces Omega_solver(t) for each (drive_id, TransitionPair).

    Omega_phys(t) = (mu_Cm * E_env(t) / hbar) * pol_overlap
    Omega_solver(t) = Omega_phys(t) * time_unit_s

    Optional polaron renormalization:
      Omega_solver(t) *= B(tr_fwd)
    where B(...) must return 1.0 if disabled.
    """

    def compute(
        self,
        resolved: Sequence[ResolvedDrive],
        tlist: np.ndarray,
        *,
        time_unit_s: float,
        decode_ctx: DriveDecodeContextProto | None = None,
    ) -> DriveCoefficients:
        if not isinstance(decode_ctx, QDDriveDecodeContext):
            raise TypeError("QDDriveStrengthModel expects QDDriveDecodeContext")

        derived = decode_ctx.derived

        tlist_f = np.asarray(tlist, dtype=float).reshape(-1)
        s = float(time_unit_s)
        t_phys_s = s * tlist_f

        hbar_Js = float(magnitude(hbar, "J*s"))

        coeffs: dict[tuple[Any, Any], np.ndarray] = {}

        for rd in resolved:
            pair = rd.transition_key
            if not isinstance(pair, TransitionPair):
                raise TypeError(
                    "Expected TransitionPair transition_key, got %s"
                    % type(pair)
                )

            payload = _payload_from_ctx(rd.drive_id, decode_ctx)

            # Field envelope in V/m on physical time grid
            E_t = _sample_E_env_V_m(payload, t_phys_s)

            # Use forward transition as absorption direction
            fwd, _ = derived.t_registry.directed(pair)

            mu_Cm = float(derived.mu_Cm(fwd))

            pol_vec = _effective_pol(payload)
            pol = 1.0 + 0.0j
            if pol_vec is not None:
                pol = complex(derived.drive_projection(fwd, pol_vec))

            # Omega in rad/s, then convert to solver units and include polarization overlap
            omega_rad_s = (mu_Cm * E_t) / hbar_Js
            omega_solver = (omega_rad_s * s).astype(complex) * pol

            # Polaron renormalization (view handles disabled case)
            B = float(derived.polaron_B(fwd))
            omega_solver = omega_solver * (B + 0.0j)

            coeffs[(rd.drive_id, pair)] = omega_solver

        return DriveCoefficients(tlist=tlist_f, coeffs=coeffs)
