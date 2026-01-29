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


def _payload_from_anywhere(
    rd: ResolvedDrive,
    *,
    ctx: QDDriveDecodeContext,
) -> Any:
    # Preferred: context mapping
    if hasattr(ctx, "meta_drives") and rd.drive_id in ctx.meta_drives:
        return ctx.meta_drives[rd.drive_id]

    # Backwards-compatible fallback: old code used rd.meta["payload"]
    payload = rd.meta.get("payload")
    if payload is not None:
        return payload

    raise KeyError(
        f"Missing drive payload for drive_id={rd.drive_id}. "
        "Provide decode_ctx.meta_drives[drive_id] = payload (recommended), "
        "or set ResolvedDrive.meta['payload'] (legacy fallback)."
    )


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

    out = np.empty(t_phys_s.size, dtype=float)
    for i in range(t_phys_s.size):
        out[i] = float(fn(float(t_phys_s[i])))
    return out


@dataclass
class QDDriveStrengthModel(DriveStrengthModelProto):
    """
    ResolvedDrive -> coefficient arrays on solver tlist.

    Omega(t) = (mu * E(t) / hbar) * pol_overlap
    Returned in solver units: Omega_solver = Omega_rad_s * time_unit_s
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
            raise TypeError(
                "QDDriveStrengthModel expects QDDriveDecodeContext")

        derived = decode_ctx.derived

        tlist = np.asarray(tlist, dtype=float)
        s = float(time_unit_s)
        t_phys_s = s * tlist

        coeffs: dict[tuple[Any, Any], np.ndarray] = {}

        hbar_Js = float(magnitude(hbar, "J*s"))

        for rd in resolved:
            pair = rd.transition_key
            if not isinstance(pair, TransitionPair):
                raise TypeError(
                    f"Expected TransitionPair transition_key, got {type(pair)}"
                )

            payload = _payload_from_anywhere(rd, ctx=decode_ctx)

            E_t = _sample_E_env_V_m(payload, t_phys_s)

            # absorption direction: forward transition in registry
            fwd, _ = derived.t_registry.directed(pair)

            mu_q = derived.mu(fwd)
            mu_Cm = float(magnitude(mu_q, "C*m"))

            pol_vec = _effective_pol(payload)
            pol = 1.0 + 0.0j
            if pol_vec is not None:
                pol = complex(derived.drive_projection(fwd, pol_vec))

            omega_rad_s = (mu_Cm * E_t) / hbar_Js
            omega_solver = (omega_rad_s * s).astype(complex) * pol

            # ---- phonon polaron renormalization (multiplicative) ----
            # Uses directed transition (absorption direction) to pick s^2 = (phi_i - phi_j)^2.
            # If phonons disabled / non-polaron, derived.polaron_B should return 1.0.
            B = 1.0
            if getattr(derived, "phonon_outputs", None) is not None:
                # if you implement DerivedQD.polaron_B(fwd)
                if hasattr(derived, "polaron_B"):
                    B = float(derived.polaron_B(fwd))
                else:
                    # fallback: access outputs dict directly
                    out = derived.phonon_outputs
                    B = float(out.B_polaron_per_transition.get(fwd, 1.0))
            omega_solver *= B + 0.0j

            coeffs[(rd.drive_id, pair)] = omega_solver

        return DriveCoefficients(tlist=tlist, coeffs=coeffs)
