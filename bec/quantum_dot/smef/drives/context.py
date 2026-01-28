from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

from smef.core.drives.protocols import DriveDecodeContextProto


@dataclass(frozen=True)
class DecodePolicy:
    allow_multi: bool = True
    k_bandwidth: float = 3.0
    sample_points: int = 5

    # polarization penalty/gating
    pol_gate_eps: float = 1e-6
    pol_penalty_power: float = 1.0
    pol_penalty_weight: float = 1.0


@dataclass(frozen=True)
class QDDriveDecodeContext(DriveDecodeContextProto):
    """
    Passive context for decoding and strength.

    derived: DerivedQD instance (unitful derived quantities)
    policy: decoding policy knobs
    bandwidth_sigma_omega_rad_s: optional fixed bandwidth estimate (physical rad/s)
    """

    derived: Any
    policy: DecodePolicy = DecodePolicy()
    bandwidth_sigma_omega_rad_s: Optional[float] = None

    meta: Mapping[str, Any] = field(default_factory=dict)
    meta_drives: MutableMapping[Any, Any] = field(default_factory=dict)
