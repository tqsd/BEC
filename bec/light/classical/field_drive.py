from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from smef.core.units import QuantityLike, as_quantity, magnitude

from bec.light.envelopes.base import EnvelopeU, SerializableEnvelopeU
from bec.light.envelopes.registry import envelope_from_json, envelope_to_json
from bec.light.core.polarization import (
    JonesMatrix,
    JonesState,
    effective_polarization,
)

from .amplitude import FieldAmplitude
from .carrier import Carrier


TimeLike = Union[QuantityLike, float, int]


def _time_quantity(t: TimeLike) -> QuantityLike:
    # Numbers interpreted as seconds.
    return as_quantity(t, "s")


@dataclass(frozen=True)
class ClassicalFieldDriveU:
    """
    Classical optical field drive (unitful).

    Physical field envelope magnitude:
        E_env(t) = E0 * envelope(t)
    where:
      - E0 has units V/m
      - envelope(t) is dimensionless float
      - t is physical time (QuantityLike or seconds as float/int)

    Carrier (optional) describes omega_L(t) in rad/s (unitful).
    Polarization (optional) is Jones-state-based and unitless.

    This object is QD-agnostic: it does not know transitions or detunings.
    """

    envelope: EnvelopeU
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
        t_q = _time_quantity(t_phys)
        val = float(self.envelope(t_q))
        return self.amplitude.E0 * val

    def E_env_V_m(self, t_phys: TimeLike) -> float:
        """
        Return E_env(t) as float in V/m.
        """
        t_q = _time_quantity(t_phys)
        val = float(self.envelope(t_q))
        return self.amplitude.E0_V_m() * val

    def omega_L_phys(self, t_phys: TimeLike) -> Optional[QuantityLike]:
        """
        Return omega_L(t) as QuantityLike in rad/s (or None if no carrier).
        """
        if self.carrier is None:
            return None
        return self.carrier.omega_phys(_time_quantity(t_phys))

    def omega_L_rad_s(self, t_phys: TimeLike) -> Optional[float]:
        """
        Return omega_L(t) as float in rad/s (or None if no carrier).
        """
        if self.carrier is None:
            return None
        return float(
            magnitude(self.carrier.omega_phys(_time_quantity(t_phys)), "rad/s")
        )

    def effective_pol(self) -> Optional[np.ndarray]:
        """
        Return effective polarization vector (length-2 complex array), or None.
        """
        return effective_polarization(
            pol_state=self.pol_state, pol_transform=self.pol_transform
        )

    def to_dict(self) -> Dict[str, Any]:
        if not isinstance(self.envelope, SerializableEnvelopeU):
            raise TypeError(
                "Envelope is not serializable; expected SerializableEnvelopeU"
            )

        return {
            "type": "classical_field_drive",
            "label": self.label,
            "preferred_kind": self.preferred_kind,
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassicalFieldDriveU":
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
            preferred_kind=data.get("preferred_kind"),
            label=data.get("label"),
        )

    def report_plain(
        self,
        *,
        time_unit_s: float,
        t_eval_solver: Optional[float] = None,
        sample_window: Tuple[float, float] = (0.0, 100.0),
        sample_points: int = 1001,
        show_ascii_plot: bool = True,
    ) -> str:
        from .report.common import compute_drive_report_data
        from .report.plain import render_plain

        rep = compute_drive_report_data(
            self,
            time_unit_s=time_unit_s,
            t_eval_solver=t_eval_solver,
            sample_window=sample_window,
            sample_points=sample_points,
            include_curve=show_ascii_plot,
        )
        return render_plain(rep, show_ascii_plot=show_ascii_plot)

    def report_rich(
        self,
        *,
        time_unit_s: float,
        t_eval_solver: Optional[float] = None,
        sample_window: Tuple[float, float] = (0.0, 100.0),
        sample_points: int = 1001,
        show_ascii_plot: bool = True,
    ) -> str:
        from .report.common import compute_drive_report_data
        from .report.rich import render_rich

        rep = compute_drive_report_data(
            self,
            time_unit_s=time_unit_s,
            t_eval_solver=t_eval_solver,
            sample_window=sample_window,
            sample_points=sample_points,
            include_curve=show_ascii_plot,
        )
        return render_rich(rep, show_ascii_plot=show_ascii_plot)
