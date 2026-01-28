from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from smef.core.drives.protocols import (
    DriveDecodeContextProto,
    DriveStrengthModelProto,
)
from smef.core.drives.types import DriveCoefficients, ResolvedDrive
from smef.core.units import hbar, magnitude

from bec.quantum_dot.enums import TransitionPair
from bec.quantum_dot.smef.drives.context import QDDriveDecodeContext


def _payload_from_ctx(
    drive_id: Any,
    *,
    meta_drives: Mapping[Any, Any],
) -> Any:
    try:
        return meta_drives[drive_id]
    except KeyError as e:
        raise KeyError(
            f"Missing drive payload for drive_id={drive_id}. "
            "Expected decode_ctx.meta_drives[drive_id] to exist."
        ) from e


def _effective_pol(payload: Any) -> Optional[np.ndarray]:
    fn = getattr(payload, "effective_pol", None)
    if callable(fn):
        out = fn()
        if out is None:
            return None
        return np.asarray(out, dtype=complex).reshape(2)
    return None


def _sample_E_env_V_m(payload: Any, t_phys_s: np.ndarray) -> np.ndarray:
    fn = getattr(payload, "E_env_V_m", None)
    if not callable(fn):
        raise AttributeError("Drive payload must provide E_env_V_m(t_phys_s)")

    # Vectorization friendly, but keep it simple and explicit
    out = np.empty(t_phys_s.size, dtype=float)
    for i in range(t_phys_s.size):
        out[i] = float(fn(float(t_phys_s[i])))
    return out


@dataclass
class QDDriveStrengthModel(DriveStrengthModelProto):
    """
    ResolvedDrive -> coefficient arrays on solver tlist.

    Computes Omega(t) from mu * E(t) / hbar, multiplied by polarization overlap.

    Returned coefficient is in solver units (unitless):
        Omega_solver(t) = Omega_rad_s(t) * time_unit_s
    """

    def compute(
        self,
        resolved: Sequence[ResolvedDrive],
        tlist: np.ndarray,
        *,
        time_unit_s: float,
        decode_ctx: Optional[DriveDecodeContextProto] = None,
    ) -> DriveCoefficients:
        if not isinstance(decode_ctx, QDDriveDecodeContext):
            raise TypeError("QDDriveStrengthModel expects QDDriveDecodeContext")

        derived = decode_ctx.derived

        # IMPORTANT: payload lookup comes from context, not from ResolvedDrive.meta
        meta_drives = getattr(decode_ctx, "meta_drives", None)
        if meta_drives is None:
            raise AttributeError(
                "QDDriveDecodeContext must provide meta_drives mapping: drive_id -> payload"
            )

        tlist = np.asarray(tlist, dtype=float)
        s = float(time_unit_s)
        t_phys_s = s * tlist

        coeffs: dict[tuple[Any, Any], np.ndarray] = {}

        for rd in resolved:
            pair = rd.transition_key
            if not isinstance(pair, TransitionPair):
                raise TypeError(
                    f"Expected TransitionPair transition_key, got {type(pair)}"
                )

            payload = _payload_from_ctx(rd.drive_id, meta_drives=meta_drives)

            # E(t) in V/m (float array)
            E_t = _sample_E_env_V_m(payload, t_phys_s)

            # choose absorption direction (low->high): forward transition in registry
            fwd, _ = derived.t_registry.directed(pair)

            # mu in C*m (unitful)
            mu_q = derived.mu(fwd)
            mu_Cm = float(magnitude(mu_q, "C*m"))

            # polarization overlap (dimensionless complex scalar)
            pol_vec = _effective_pol(payload)
            pol = 1.0 + 0.0j
            if pol_vec is not None:
                pol = complex(derived.drive_projection(fwd, pol_vec))

            # Omega(t) = mu * E / hbar (rad/s), then convert to solver units by * time_unit_s
            hbar_Js = float(magnitude(hbar, "J*s"))
            omega_rad_s = (mu_Cm * E_t) / hbar_Js
            omega_solver = (omega_rad_s * s).astype(complex) * pol

            coeffs[(rd.drive_id, pair)] = omega_solver

        return DriveCoefficients(tlist=tlist, coeffs=coeffs)
