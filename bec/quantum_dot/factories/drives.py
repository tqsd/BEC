from __future__ import annotations

import numpy as np
from smef.core.units import Q, as_quantity, hbar, magnitude

from bec.light.classical import carrier_profiles
from bec.light.classical.amplitude import FieldAmplitude
from bec.light.classical.carrier import Carrier
from bec.light.classical.field_drive import ClassicalFieldDriveU
from bec.light.core.polarization import (
    JonesMatrix,
    JonesState,
    effective_polarization,
)
from bec.light.envelopes.gaussian import GaussianEnvelopeU
from bec.quantum_dot.enums import TransitionPair


def _normalize_hv(v: np.ndarray) -> tuple[complex, complex]:
    a = np.asarray(v, dtype=complex).reshape(2)
    n = float(np.linalg.norm(a))
    if n == 0.0:
        return (1.0 + 0.0j, 0.0 + 0.0j)
    a = a / n
    return (complex(a[0]), complex(a[1]))


def auto_pol_for_transition(qd, *, pair: TransitionPair) -> JonesState:
    """
    Choose JonesState aligned with the transition dipole (directed fwd transition).

    Uses:
      - qd.derived_view.t_registry.directed(pair)
      - qd.derived_view.dipoles.e_pol_hv(tr)  -> length-2 complex
    """
    derived = qd.derived_view
    fwd, _ = derived.t_registry.directed(pair)

    e = np.asarray(derived.dipoles.e_pol_hv(fwd), dtype=complex).reshape(2)
    j0, j1 = _normalize_hv(e)
    return JonesState(jones=(j0, j1), normalize=True)


def _infer_kind_from_pair(pair: TransitionPair) -> str:
    # If your TransitionRegistry exposes metadata, prefer that.
    if pair is TransitionPair.G_XX:
        return "2ph"
    return "1ph"


def _infer_omega0_rad_s(qd, *, pair: TransitionPair, kind: str) -> float:
    derived = qd.derived_view
    omega_ref = float(derived.omega_ref_rad_s(pair))  # physical rad/s
    if kind == "2ph":
        # emitter uses mult=2 for detuning and assumes omega_L is per-photon
        return 0.5 * omega_ref
    return omega_ref



def make_gaussian_field_drive_pi(
    qd,
    *,
    pair: TransitionPair,
    t0,
    sigma,
    pol_state: JonesState | None = None,
    pol_transform: JonesMatrix | None = None,
    theta_rad: float = float(np.pi),
    compensate_polaron: bool = True,
    omega0_rad_s: float | None = None,
    # New: chirp selection (backwards compatible)
    chirp_kind: str | None = None,  # "none", "constant", "linear", "tanh"
    chirp_t0=None,
    # Backwards compatible linear chirp parameter
    chirp_rate_rad_s2: float | None = None,
    # New parameters
    chirp_delta_rad_s: float | None = None,   # for constant chirp
    tanh_delta_rad_s: float | None = None,    # for tanh chirp
    tanh_tau_s: float | None = None,          # for tanh chirp
    preferred_kind: str | None = None,  # "1ph" or "2ph"
    label: str | None = None,
) -> ClassicalFieldDriveU:
    """
    Build a Gaussian pi (or general theta) pulse for a given TransitionPair.

    - Auto polarization: if pol_state is None, align with the transition dipole.
    - Auto carrier: if omega0_rad_s is None, infer from derived_view.omega_ref_rad_s(pair)
      and preferred_kind (1ph -> omega_ref, 2ph -> 0.5*omega_ref).
    - Calibration: choose E0 so that (closed two-level resonant limit)
        integral Omega(t) dt = theta_rad
      where Omega(t) is the Rabi frequency used by SMEF for this drive.

    Chirp handling:
    - Default behavior is preserved:
        * if chirp_kind is None:
            - chirp_rate_rad_s2 is None -> no chirp
            - chirp_rate_rad_s2 provided -> linear chirp
    - Explicit chirp_kind overrides:
        "none"     -> delta_omega(t) = 0
        "constant" -> delta_omega(t) = chirp_delta_rad_s
        "linear"   -> delta_omega(t) = chirp_rate_rad_s2 * (t - chirp_t0)
        "tanh"     -> delta_omega(t) = tanh_delta_rad_s * tanh((t - chirp_t0)/tanh_tau_s)
    """
    derived = qd.derived_view
    fwd, _bwd = derived.t_registry.directed(pair)

    kind = preferred_kind if preferred_kind is not None else _infer_kind_from_pair(pair)
    if kind not in ("1ph", "2ph"):
        raise ValueError("preferred_kind must be '1ph' or '2ph'")

    if pol_state is None:
        pol_state = auto_pol_for_transition(qd, pair=pair)

    E_pol = effective_polarization(pol_state=pol_state, pol_transform=pol_transform)
    if E_pol is None:
        raise ValueError("Could not construct effective polarization")

    proj = complex(derived.drive_projection(fwd, E_pol))
    proj_abs = abs(proj)
    if proj_abs == 0.0:
        raise ValueError("Polarization is orthogonal to the dipole for this transition")

    mu_Cm = float(magnitude(derived.mu(fwd), "C*m"))

    B = float(derived.polaron_B(fwd)) if compensate_polaron else 1.0
    if not np.isfinite(B) or B <= 0.0:
        raise ValueError("Invalid polaron_B (must be finite and > 0), got %s" % (B,))

    t0_q = as_quantity(t0, "s")
    sigma_q = as_quantity(sigma, "s")
    env = GaussianEnvelopeU(t0=t0_q, sigma=sigma_q)

    area_env_s = float(env.area_seconds())
    if area_env_s <= 0.0:
        raise ValueError("Envelope area must be > 0")

    hbar_SI = float(hbar.to("J*s").magnitude)
    E0_V_m = float(theta_rad) * hbar_SI / (B * mu_Cm * proj_abs * area_env_s)

    # Carrier
    w0 = omega0_rad_s
    if w0 is None:
        w0 = _infer_omega0_rad_s(qd, pair=pair, kind=kind)

    carrier = None
    if w0 is not None:
        omega0_q = Q(float(w0), "rad/s")
        base_t0 = t0_q if chirp_t0 is None else as_quantity(chirp_t0, "s")

        # Backwards-compat default: if chirp_kind not given, infer from chirp_rate_rad_s2
        if chirp_kind is None:
            ck = "linear" if chirp_rate_rad_s2 is not None else "none"
        else:
            ck = str(chirp_kind).strip().lower()

        if ck == "none":
            dw = carrier_profiles.constant(Q(0.0, "rad/s"))

        elif ck == "constant":
            if chirp_delta_rad_s is None:
                raise ValueError("chirp_delta_rad_s is required for chirp_kind='constant'")
            dw = carrier_profiles.constant(Q(float(chirp_delta_rad_s), "rad/s"))

        elif ck == "linear":
            if chirp_rate_rad_s2 is None:
                raise ValueError("chirp_rate_rad_s2 is required for chirp_kind='linear'")
            dw = carrier_profiles.linear_chirp(
                rate=Q(float(chirp_rate_rad_s2), "rad/s^2"),
                t0=base_t0,
            )

        elif ck == "tanh":
            if tanh_delta_rad_s is None or tanh_tau_s is None:
                raise ValueError(
                    "tanh_delta_rad_s and tanh_tau_s are required for chirp_kind='tanh'"
                )
            # Requires carrier_profiles.tanh_chirp to exist (see note below).
            dw = carrier_profiles.tanh_chirp(
                delta_max=Q(float(tanh_delta_rad_s), "rad/s"),
                tau=Q(float(tanh_tau_s), "s"),
                t0=base_t0,
            )
        else:
            raise ValueError("Unknown chirp_kind %r" % (ck,))

        carrier = Carrier(omega0=omega0_q, delta_omega=dw)

    return ClassicalFieldDriveU(
        envelope=env,
        amplitude=FieldAmplitude(E0=Q(E0_V_m, "V/m")),
        carrier=carrier,
        pol_state=pol_state,
        pol_transform=pol_transform,
        preferred_kind=kind,
        label=label,
    )
