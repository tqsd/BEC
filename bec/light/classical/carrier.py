from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from smef.core.units import QuantityLike, Q, as_quantity, magnitude


TimeLike = Union[QuantityLike, float, int]
OmegaLike = Union[QuantityLike, float, int]

# A frequency profile returns omega(t) in rad/s (unitful).
OmegaFn = Callable[[QuantityLike], QuantityLike]


def _time_s(t: TimeLike) -> float:
    if hasattr(t, "to"):
        return float(magnitude(t, "s"))
    return float(t)


def _omega_rad_s(w: OmegaLike) -> float:
    if hasattr(w, "to"):
        return float(magnitude(w, "rad/s"))
    return float(w)


@dataclass(frozen=True)
class Carrier:
    """
    Optical carrier definition.

    omega0: base angular frequency [rad/s]
    delta_omega: additional angular frequency offset [rad/s] or callable profile
    phi0: constant phase offset [rad]
    """

    omega0: QuantityLike
    delta_omega: Union[QuantityLike, OmegaFn] = 0.0
    phi0: float = 0.0
    label: Optional[str] = None

    _omega0_rad_s: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        w0 = as_quantity(self.omega0, "rad/s")
        object.__setattr__(self, "omega0", w0)
        object.__setattr__(self, "_omega0_rad_s", float(magnitude(w0, "rad/s")))

        d = self.delta_omega
        if not callable(d):
            d_q = as_quantity(d, "rad/s")
            object.__setattr__(self, "delta_omega", d_q)

    def omega_phys(self, t_phys: TimeLike) -> QuantityLike:
        """
        Return instantaneous omega at physical time t_phys.
        Output is a QuantityLike in rad/s.
        """
        t_s = _time_s(t_phys)
        d = self.delta_omega
        if callable(d):
            dw = d(Q(t_s, "s"))
            return as_quantity(dw, "rad/s") + self.omega0
        return self.omega0 + as_quantity(d, "rad/s")

    def omega_rad_s(self, t_phys: TimeLike) -> float:
        """
        Fast path: return instantaneous omega at physical time t_phys as float rad/s.

        If delta_omega is callable, this still calls it (unitful input), so for hot loops
        you should compile first.
        """
        d = self.delta_omega
        if callable(d):
            return float(magnitude(self.omega_phys(t_phys), "rad/s"))
        return self._omega0_rad_s + float(magnitude(d, "rad/s"))

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON serialization. Only works for constant delta_omega.
        """
        if callable(self.delta_omega):
            raise TypeError("Callable delta_omega is not JSON serializable")

        return {
            "type": "carrier",
            "label": self.label,
            "omega0": {
                "value": float(magnitude(self.omega0, "rad/s")),
                "unit": "rad/s",
            },
            "delta_omega": {
                "value": float(magnitude(self.delta_omega, "rad/s")),
                "unit": "rad/s",
            },
            "phi0": float(self.phi0),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Carrier":
        w0 = data["omega0"]
        dw = data.get("delta_omega", {"value": 0.0, "unit": "rad/s"})
        return cls(
            omega0=as_quantity(
                float(w0["value"]), str(w0.get("unit", "rad/s"))
            ),
            delta_omega=as_quantity(
                float(dw["value"]), str(dw.get("unit", "rad/s"))
            ),
            phi0=float(data.get("phi0", 0.0)),
            label=data.get("label"),
        )
