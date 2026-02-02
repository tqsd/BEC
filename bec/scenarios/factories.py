from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import DriveFactory, SchemeKind


def _apply_common_error_knobs(
    payload: Any, *, amp_scale: float, detuning_offset_rad_s: float
) -> Any:
    """
    Convention:
    - payload may expose these optional callables/fields:
        * payload.set_amp_scale(scale: float)
        * payload.set_detuning_offset_rad_s(dw: float)
    If absent, do nothing.

    This avoids hard-coding payload implementation details here.
    """
    if hasattr(payload, "set_amp_scale") and callable(
        getattr(payload, "set_amp_scale")
    ):
        payload.set_amp_scale(float(amp_scale))
    if hasattr(payload, "set_detuning_offset_rad_s") and callable(
        getattr(payload, "set_detuning_offset_rad_s")
    ):
        payload.set_detuning_offset_rad_s(float(detuning_offset_rad_s))
    return payload


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
    from smef.core.drives.types import DriveSpec
    from smef.core.units import Q
    from bec.quantum_dot.enums import TransitionPair
    from bec.quantum_dot.factories.drives import make_gaussian_field_drive_pi

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
    # scheme-specific knobs:
    chirp_rate_rad_s2: Optional[float] = None,
) -> Tuple[List[Any], List[Any]]:
    """
    ARP: usually one chirped pulse on G<->XX (2ph) or on exciton (1ph) depending on your design.

    This is a placeholder: wire it to your actual ARP payload factory.
    The only hard requirements for sweeps:
    - returns specs and payloads
    - respects amp_scale and detuning_offset_rad_s via setters or payload fields
    """
    from smef.core.drives.types import DriveSpec

    # TODO: replace with your actual ARP payload factory.
    # Example idea:
    # payload = make_tanh_chirp_drive(..., chirp_rate_rad_s2=chirp_rate_rad_s2, ...)
    payload = _raise_not_implemented(
        "make_arp_drive_specs: hook up your ARP payload factory"
    )

    _apply_common_error_knobs(
        payload,
        amp_scale=float(amp_scale),
        detuning_offset_rad_s=float(detuning_offset_rad_s),
    )

    specs = [
        DriveSpec(
            payload=payload, drive_id=getattr(payload, "label", None) or label
        )
    ]
    return specs, [payload]


def make_bichromatic_drive_specs(
    qd: Any,
    *,
    cfg: Any,
    amp_scale: float = 1.0,
    detuning_offset_rad_s: float = 0.0,
    label: str = "bichromatic",
    # scheme-specific knobs:
    rel_phase_rad: float = 0.0,
) -> Tuple[List[Any], List[Any]]:
    """
    Bichromatic: typically two 1ph tones (or two pulses) driving G<->X and X<->XX.

    Placeholder: return two DriveSpec entries if your implementation uses two payloads.
    Apply error knobs to both payloads consistently.
    """
    from smef.core.drives.types import DriveSpec

    # TODO: replace with your actual bichromatic payload factory(s).
    payloads = _raise_not_implemented(
        "make_bichromatic_drive_specs: hook up your bichromatic payload factory"
    )

    # If you have two payloads, apply knobs to both:
    for p in payloads:
        _apply_common_error_knobs(
            p,
            amp_scale=float(amp_scale),
            detuning_offset_rad_s=float(detuning_offset_rad_s),
        )

    specs = [
        DriveSpec(
            payload=p, drive_id=getattr(p, "label", None) or f"{label}_{i}"
        )
        for i, p in enumerate(payloads)
    ]
    return specs, list(payloads)


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
