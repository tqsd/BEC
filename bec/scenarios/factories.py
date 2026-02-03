from __future__ import annotations

from typing import Any, List, Optional, Tuple


from smef.core.units import Q
from bec.light.classical.carrier import Carrier
from smef.core.drives.types import DriveSpec
from bec.quantum_dot.enums import TransitionPair
from bec.quantum_dot.factories.drives import make_gaussian_field_drive_pi
from .types import DriveFactory, SchemeKind


def _apply_common_error_knobs(
    payload: Any, *, amp_scale: float, detuning_offset_rad_s: float
) -> Any:

    p = payload

    # amplitude scaling
    if hasattr(p, "scaled") and callable(getattr(p, "scaled")):
        if float(amp_scale) != 1.0:
            p = p.scaled(float(amp_scale))
    else:
        # fallback to old setter convention if some payloads still use it
        if hasattr(p, "set_amp_scale") and callable(
            getattr(p, "set_amp_scale")
        ):
            p.set_amp_scale(float(amp_scale))

    # detuning offset: shift omega_L(t) by constant detuning_offset_rad_s
    dw = float(detuning_offset_rad_s)
    if abs(dw) > 0.0:
        if (
            hasattr(p, "carrier")
            and getattr(p, "carrier") is not None
            and Carrier is not None
        ):
            car = p.carrier
            d = car.delta_omega

            if callable(d):

                def d_new(t):
                    return d(t) + Q(dw, "rad/s")

                delta_omega_new = d_new
            else:
                delta_omega_new = d + Q(dw, "rad/s")

            car_new = Carrier(
                omega0=car.omega0,
                delta_omega=delta_omega_new,
                phi0=float(car.phi0),
                label=car.label,
            )

            # rebuild drive (ClassicalFieldDriveU is frozen)
            p = type(p)(
                envelope=p.envelope,
                amplitude=p.amplitude,
                carrier=car_new,
                pol_state=getattr(p, "pol_state", None),
                pol_transform=getattr(p, "pol_transform", None),
                preferred_kind=getattr(p, "preferred_kind", None),
                label=getattr(p, "label", None),
            )
        else:
            # fallback to old setter convention
            if hasattr(p, "set_detuning_offset_rad_s") and callable(
                getattr(p, "set_detuning_offset_rad_s")
            ):
                p.set_detuning_offset_rad_s(dw)

    return p


def make_tpe_drive_specs(
    qd: Any,
    *,
    cfg: Any,
    amp_scale: float = 1.0,
    detuning_offset_rad_s: float = 0.0,
    label: str = "tpe",
) -> Tuple[List[Any], List[Any]]:
    """
    TPE: single 2ph drive on G<->XX.

    Expects your factory to accept:
      compensate_polaron, omega0_rad_s, chirp_rate_rad_s2

    Error knobs:
    - amp_scale: should scale E_env(t)
    - detuning_offset_rad_s: should offset omega_L(t) (laser omega) in the payload
    """

    omega_ref = float(qd.derived_view.omega_ref_rad_s(TransitionPair.G_XX))
    omega0 = 0.5 * omega_ref

    payload = make_gaussian_field_drive_pi(
        qd,
        pair=TransitionPair.G_XX,
        t0=Q(float(cfg.t0_ns), "ns"),
        sigma=Q(float(cfg.sigma_ns), "ns"),
        preferred_kind="2ph",
        label=label,
        compensate_polaron=bool(cfg.compensate_polaron),
        omega0_rad_s=float(omega0),
        chirp_rate_rad_s2=None,
    )

    _apply_common_error_knobs(
        payload,
        amp_scale=float(amp_scale),
        detuning_offset_rad_s=float(detuning_offset_rad_s),
    )

    specs = [DriveSpec(payload=payload, drive_id=payload.label or label)]
    return specs, [payload]


def make_arp_drive_specs(
    qd: Any,
    *,
    cfg: Any,
    amp_scale: float = 1.0,
    detuning_offset_rad_s: float = 0.0,
    label: str = "arp",
    # Chirp selection
    chirp_kind: Optional[str] = None,  # None | "linear" | "tanh" | "constant"
    chirp_rate_rad_s2: Optional[float] = None,  # linear chirp
    chirp_delta_rad_s: Optional[float] = None,  # constant detuning
    tanh_delta_rad_s: Optional[float] = None,  # tanh amplitude
    tanh_tau_ns: Optional[float] = None,  # tanh timescale (ns)
    # Pulse overrides
    sigma_ns: Optional[float] = None,
    t0_ns: Optional[float] = None,
) -> Tuple[List[Any], List[Any]]:

    # Per-scheme overrides (fallback to cfg)
    sigma = float(cfg.sigma_ns) if sigma_ns is None else float(sigma_ns)
    t0 = float(cfg.t0_ns) if t0_ns is None else float(t0_ns)

    # Default chirp behavior (backward compatible):
    # - if chirp_kind is None:
    #     * chirp_rate_rad_s2 None -> no chirp
    #     * chirp_rate_rad_s2 set  -> linear chirp
    if chirp_kind is None:
        if chirp_rate_rad_s2 is None:
            chirp_kind_use = "none"
        else:
            chirp_kind_use = "linear"
    else:
        chirp_kind_use = chirp_kind.lower()

    omega_ref = float(qd.derived_view.omega_ref_rad_s(TransitionPair.G_XX))
    omega0 = 0.5 * omega_ref

    payload = make_gaussian_field_drive_pi(
        qd,
        pair=TransitionPair.G_XX,
        t0=Q(t0, "ns"),
        sigma=Q(sigma, "ns"),
        preferred_kind="2ph",
        label=label,
        compensate_polaron=bool(cfg.compensate_polaron),
        omega0_rad_s=float(omega0),
        # Chirp plumbing
        chirp_kind=chirp_kind_use,
        chirp_rate_rad_s2=chirp_rate_rad_s2,
        chirp_delta_rad_s=chirp_delta_rad_s,
        tanh_delta_rad_s=tanh_delta_rad_s,
        tanh_tau_s=(
            None
            if tanh_tau_ns is None
            else float(Q(tanh_tau_ns, "ns").to("s").magnitude)
        ),
        chirp_t0=Q(t0, "ns"),
    )

    payload = _apply_common_error_knobs(
        payload,
        amp_scale=float(amp_scale),
        detuning_offset_rad_s=float(detuning_offset_rad_s),
    )

    specs = [DriveSpec(payload=payload, drive_id=payload.label or label)]
    return specs, [payload]


def _set_carrier_omega(
    payload: Any, omega0_rad_s: Any, rel_phase_rad: float = 0.0
) -> Any:
    """
    Rebuild payload with a Carrier whose base frequency is set to omega0_rad_s.

    - omega0_rad_s may be a float/int (interpreted as rad/s) or a QuantityLike convertible to rad/s.
    - delta_omega is preserved from the existing carrier if present (constant QuantityLike or callable profile).
    - phi0 is preserved and incremented by rel_phase_rad (dimensionless radians).
    """
    from bec.light.classical.carrier import Carrier
    from smef.core.units import Q, as_quantity

    # Convert omega0 into a unitful quantity [rad/s]
    omega0_q = as_quantity(omega0_rad_s, "rad/s")

    c = getattr(payload, "carrier", None)

    if c is None:
        # No previous carrier: default delta_omega = 0 [rad/s], phi0 = rel_phase_rad
        car_new = Carrier(
            omega0=omega0_q,
            delta_omega=Q(0.0, "rad/s"),
            phi0=float(rel_phase_rad),
            label=None,
        )
    else:
        # Preserve delta_omega either as callable or as a unitful quantity [rad/s]
        d = getattr(c, "delta_omega", Q(0.0, "rad/s"))
        if callable(d):
            delta_new = d
        else:
            delta_new = as_quantity(d, "rad/s")

        phi0_new = float(getattr(c, "phi0", 0.0)) + float(rel_phase_rad)

        car_new = Carrier(
            omega0=omega0_q,
            delta_omega=delta_new,
            phi0=phi0_new,
            label=getattr(c, "label", None),
        )

    # Rebuild drive (ClassicalFieldDriveU is frozen in your code)
    return type(payload)(
        envelope=payload.envelope,
        amplitude=payload.amplitude,
        carrier=car_new,
        pol_state=getattr(payload, "pol_state", None),
        pol_transform=getattr(payload, "pol_transform", None),
        preferred_kind=getattr(payload, "preferred_kind", None),
        label=getattr(payload, "label", None),
    )


def make_bichromatic_drive_specs(
    qd,
    *,
    cfg,
    amp_scale=1.0,
    detuning_offset_rad_s=0.0,
    dpe_delta_rad_s=0.0,
    label="bichromatic",
    rel_phase_rad=0.0,
    dt_ns=0.10,
    sigma_gx_ns=None,
    sigma_xx_ns=None,
):
    from smef.core.drives.types import DriveSpec
    from smef.core.units import Q
    from bec.quantum_dot.enums import TransitionPair
    from bec.quantum_dot.factories.drives import make_gaussian_field_drive_pi

    if sigma_gx_ns is None:
        sigma_gx_ns = float(cfg.sigma_ns)
    if sigma_xx_ns is None:
        sigma_xx_ns = float(cfg.sigma_ns) * 0.6

    t0_gx = float(cfg.t0_ns) - 0.5 * float(dt_ns)
    t0_xx = float(cfg.t0_ns) + 0.5 * float(dt_ns)

    pair_gx = TransitionPair.G_X1
    pair_xx = TransitionPair.X1_XX

    payload1 = make_gaussian_field_drive_pi(
        qd,
        pair=pair_gx,
        t0=Q(t0_gx, "ns"),
        sigma=Q(float(sigma_gx_ns), "ns"),
        preferred_kind="1ph",
        label=f"{label}_gx",
        compensate_polaron=bool(cfg.compensate_polaron),
        omega0_rad_s=None,
        chirp_rate_rad_s2=None,
    )

    payload2 = make_gaussian_field_drive_pi(
        qd,
        pair=pair_xx,
        t0=Q(t0_xx, "ns"),
        sigma=Q(float(sigma_xx_ns), "ns"),
        preferred_kind="1ph",
        label=f"{label}_xx",
        compensate_polaron=bool(cfg.compensate_polaron),
        omega0_rad_s=None,
        chirp_rate_rad_s2=None,
    )

    # --- DPE-style symmetric detuning while keeping 2-ph resonance
    if float(dpe_delta_rad_s) != 0.0 or float(detuning_offset_rad_s) != 0.0:
        # You need the reference transition frequencies for each tone.
        # Replace these with the exact calls you already use elsewhere.
        omega_gx_ref = float(qd.derived_view.omega_ref_rad_s(pair_gx))
        omega_xx_ref = float(qd.derived_view.omega_ref_rad_s(pair_xx))

        omega1 = (
            omega_gx_ref + float(dpe_delta_rad_s) +
            float(detuning_offset_rad_s)
        )
        omega2 = (
            omega_xx_ref - float(dpe_delta_rad_s) +
            float(detuning_offset_rad_s)
        )

        payload1 = _set_carrier_omega(payload1, omega1, rel_phase_rad=0.0)
        payload2 = _set_carrier_omega(
            payload2, omega2, rel_phase_rad=float(rel_phase_rad)
        )
    else:
        # keep your existing rel phase behavior if no explicit omega override
        payload2 = _set_carrier_omega(
            payload2,
            getattr(payload2.carrier, "omega0", 0.0),
            rel_phase_rad=float(rel_phase_rad),
        )

    # Keep amplitude scaling etc.
    payload1 = _apply_common_error_knobs(
        payload1, amp_scale=amp_scale, detuning_offset_rad_s=0.0
    )
    payload2 = _apply_common_error_knobs(
        payload2, amp_scale=amp_scale, detuning_offset_rad_s=0.0
    )

    specs = [
        DriveSpec(payload=payload1, drive_id=payload1.label or f"{label}_gx"),
        DriveSpec(payload=payload2, drive_id=payload2.label or f"{label}_xx"),
    ]
    return specs, [payload1, payload2]


def get_scheme_factory(kind: SchemeKind) -> DriveFactory:
    if kind is SchemeKind.TPE:
        return make_tpe_drive_specs
    if kind is SchemeKind.ARP:
        return make_arp_drive_specs
    if kind is SchemeKind.BICHROMATIC:
        return make_bichromatic_drive_specs
    raise ValueError(f"Unknown scheme kind: {kind}")


def _raise_not_implemented(msg: str):
    raise NotImplementedError(msg)
