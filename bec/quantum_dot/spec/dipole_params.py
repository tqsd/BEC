from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from bec.core.units import as_Cm
from bec.light.core.polarization import JonesState
from bec.quantum_dot.enums import Transition, TransitionPair, transition_pair_of


HVVec = np.ndarray  # shape (2,), complex


def _as_hv_vec(x: Any) -> HVVec:
    """
    Coerce to normalized complex 2-vector in HV basis.
    Accepts list/tuple/np array of shape (2,).
    """
    v = np.asarray(x, dtype=complex).reshape(2)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("Polarization vector must be non-zero.")
    return v / n


def _require_hv_state(js: JonesState) -> None:
    if js.basis != "HV":
        raise ValueError(f"JonesState basis must be 'HV', got {js.basis!r}")


@dataclass(frozen=True)
class DipoleParams:
    """
    Transition dipole moment parameters (unit-aware).

    Interpretation for directed transition `tr`:

      d_vec(tr) = mu(pair(tr)) * e_pol(pair(tr))

    where:
      - mu(...) is scalar magnitude in C*m (QuantityLike)
      - e_pol(...) is normalized complex 2-vector in HV basis (dimensionless)

    We store by TransitionPair so forward/backward remain consistent.
    """

    mu_default: Any = field(default_factory=lambda: as_Cm(1e-29))
    mu_by_pair: Dict[TransitionPair, Any] = field(default_factory=dict)

    pol_hv_by_pair: Dict[TransitionPair, Tuple[complex, complex]] = field(
        default_factory=dict
    )

    default_pol: JonesState = field(default_factory=JonesState.H)

    @classmethod
    def from_values(
        cls,
        *,
        mu_default_Cm: Any = 1e-29,
        mu_by_pair_Cm: Mapping[TransitionPair, Any] | None = None,
        pol_hv_by_pair: (
            Mapping[TransitionPair, Tuple[complex, complex]] | None
        ) = None,
        default_pol: JonesState | None = None,
    ) -> "DipoleParams":
        dp = cls(
            mu_default=as_Cm(mu_default_Cm),
            mu_by_pair={p: as_Cm(v) for p, v in (mu_by_pair_Cm or {}).items()},
            pol_hv_by_pair=dict(pol_hv_by_pair or {}),
            default_pol=(
                default_pol if default_pol is not None else JonesState.H()
            ),
        )
        dp.validate()
        return dp

    @classmethod
    def biexciton_cascade_defaults(
        cls,
        *,
        mu_default_Cm: Any = 1e-29,
        mu_by_pair_Cm: Mapping[TransitionPair, Any] | None = None,
    ) -> "DipoleParams":
        pol = {
            TransitionPair.G_X1: (1.0 + 0j, 0.0 + 0j),  # H
            TransitionPair.G_X2: (0.0 + 0j, 1.0 + 0j),  # V
            TransitionPair.X1_XX: (0.0 + 0j, 1.0 + 0j),  # V
            TransitionPair.X2_XX: (1.0 + 0j, 0.0 + 0j),  # H
        }
        return cls.from_values(
            mu_default_Cm=mu_default_Cm,
            mu_by_pair_Cm=mu_by_pair_Cm,
            pol_hv_by_pair=pol,
            default_pol=JonesState.H(),
        )

    def mu_pair(self, pair: TransitionPair):
        return self.mu_by_pair.get(pair, self.mu_default)

    def mu_Cm_pair(self, pair: TransitionPair) -> float:
        return float(self.mu_pair(pair).to("C*m").magnitude)

    def e_pol_hv_pair(self, pair: TransitionPair) -> HVVec:
        if pair in self.pol_hv_by_pair:
            dH, dV = self.pol_hv_by_pair[pair]
            return _as_hv_vec([dH, dV])
        _require_hv_state(self.default_pol)
        return _as_hv_vec(self.default_pol.as_array())

    def mu(self, tr: Transition):
        return self.mu_pair(transition_pair_of(tr))

    def mu_Cm(self, tr: Transition) -> float:
        return self.mu_Cm_pair(transition_pair_of(tr))

    def e_pol_hv(self, tr: Transition) -> HVVec:
        return self.e_pol_hv_pair(transition_pair_of(tr))

    def d_vec_hv(self, tr: Transition) -> HVVec:
        return self.mu_Cm(tr) * self.e_pol_hv(tr)

    def validate(self) -> None:
        mu0 = float(self.mu_default.to("C*m").magnitude)
        if mu0 <= 0.0:
            raise ValueError(f"mu_default must be > 0, got {self.mu_default}")

        for p, mu in self.mu_by_pair.items():
            val = float(mu.to("C*m").magnitude)
            if val <= 0.0:
                raise ValueError(f"mu_by_pair[{p}] must be > 0, got {mu}")

        for p, (dH, dV) in self.pol_hv_by_pair.items():
            _ = _as_hv_vec([dH, dV])

        _require_hv_state(self.default_pol)
        _ = _as_hv_vec(self.default_pol.as_array())

    def as_floats(self) -> Dict[str, Any]:
        return {
            "mu_default_Cm": float(self.mu_default.to("C*m").magnitude),
            "mu_by_pair_Cm": {
                p: float(mu.to("C*m").magnitude)
                for p, mu in self.mu_by_pair.items()
            },
            "pol_hv_by_pair": dict(self.pol_hv_by_pair),
        }
