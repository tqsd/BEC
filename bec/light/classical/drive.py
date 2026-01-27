from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from bec.units import QuantityLike, Q, as_quantity, magnitude

from bec.light.envelopes.base import Envelope, SerializableEnvelope, TimeLike
from bec.light.envelopes.registry import envelope_from_json, envelope_to_json

from bec.light.core.polarization import (
    JonesMatrix,
    JonesState,
    effective_polarization,
)

from .amplitude import FieldAmplitude
from .carrier import Carrier


@dataclass(frozen=True)
class ClassicalFieldDrive:
    """
    Classical optical field drive.

    The physical field envelope magnitude is:
        E_env(t) = E0 * envelope(t)
    where E0 has units V/m, envelope(t) is dimensionless.

    Carrier (optional) describes omega_L(t) in rad/s (unitful).
    Polarization (optional) is Jones-state-based and unitless.

    This object is QD-agnostic: it does not know transitions or detunings.
    """

    envelope: Envelope
    amplitude: FieldAmplitude

    carrier: Optional[Carrier] = None

    pol_state: Optional[JonesState] = None
    pol_transform: Optional[JonesMatrix] = None

    preferred_kind: Optional[str] = None  # "1ph" or "2ph"

    label: Optional[str] = None

    def E_env_phys(self, t_phys: TimeLike) -> QuantityLike:
        """
        Return E_env(t) in V/m (unitful).
        """
        val = float(self.envelope(t_phys))
        return self.amplitude.E0 * val

    def E_env_V_m(self, t_phys: TimeLike) -> float:
        """
        Return E_env(t) as float in V/m.
        This is fast if the envelope has a seconds fast-path.
        """
        val = float(self.envelope(t_phys))
        return self.amplitude.E0_V_m() * val

    def omega_L_phys(self, t_phys: TimeLike) -> Optional[QuantityLike]:
        """
        Return omega_L(t) as QuantityLike in rad/s (or None if no carrier).
        """
        if self.carrier is None:
            return None
        return self.carrier.omega_phys(t_phys)

    def effective_pol(self) -> Optional[np.ndarray]:
        """
        Return effective polarization vector (length-2 complex array), or None.
        """
        return effective_polarization(
            pol_state=self.pol_state, pol_transform=self.pol_transform
        )

    def to_dict(self) -> Dict[str, Any]:
        if not isinstance(self.envelope, SerializableEnvelope):
            raise TypeError(
                "Envelope is not serializable; expected SerializableEnvelope"
            )

        data: Dict[str, Any] = {
            "type": "classical_field_drive",
            "label": self.label,
            "envelope": envelope_to_json(self.envelope),
            "amplitude": self.amplitude.to_dict(),
            "carrier": None if self.carrier is None else self.carrier.to_dict(),
            "pol_state": (
                None if self.pol_state is None else self.pol_state.to_dict()
            ),
            "pol_transform": (
                None
                if self.pol_transform is None
                else self.pol_transform.to_dict()
            ),
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassicalFieldDrive":
        env = envelope_from_json(data["envelope"])
        amp = FieldAmplitude.from_dict(data["amplitude"])

        car = data.get("carrier")
        carrier = None if car is None else Carrier.from_dict(car)

        ps = data.get("pol_state")
        pol_state = None if ps is None else JonesState.from_dict(ps)

        pt = data.get("pol_transform")
        pol_transform = None if pt is None else JonesMatrix.from_dict(pt)

        return cls(
            envelope=env,
            amplitude=amp,
            carrier=carrier,
            pol_state=pol_state,
            pol_transform=pol_transform,
            label=data.get("label"),
        )
