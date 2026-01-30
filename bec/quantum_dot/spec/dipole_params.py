from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple, Optional

import numpy as np

from bec.core.units import as_Cm, as_eV
from bec.light.core.polarization import JonesState
from bec.quantum_dot.enums import Transition, TransitionPair, transition_pair_of


HVVec = np.ndarray  # shape (2,), complex


def _as_hv_vec(x: Any) -> HVVec:
    v = np.asarray(x, dtype=complex).reshape(2)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("Polarization vector must be non-zero.")
    return v / n


def _require_hv_state(js: JonesState) -> None:
    if js.basis != "HV":
        raise ValueError(f"JonesState basis must be 'HV', got {js.basis!r}")


def _rotated_linear_basis(theta_rad: float) -> tuple[HVVec, HVVec]:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    Ht = np.array([c + 0j, s + 0j], dtype=complex)
    Vt = np.array([-s + 0j, c + 0j], dtype=complex)
    return _as_hv_vec(Ht), _as_hv_vec(Vt)


def _elliptic_pair(alpha_rad: float, theta_rad: float) -> tuple[HVVec, HVVec]:
    """
    Returns two orthonormal Jones vectors (p1, p2) in HV coordinates.

    alpha = 0       -> linear basis (Ht, Vt)
    alpha = pi/4    -> circular basis (in the rotated frame)
    """
    Ht, Vt = _rotated_linear_basis(theta_rad)

    ca = float(np.cos(alpha_rad))
    sa = float(np.sin(alpha_rad))

    p1 = ca * Ht + 1j * sa * Vt
    p2 = -1j * sa * Ht + ca * Vt

    return _as_hv_vec(p1), _as_hv_vec(p2)


def _alpha_from_delta_eff(delta_eff_eV: float, fss_scale_eV: float) -> float:
    """
    Heuristic: alpha(|Delta_eff|) goes from pi/4 at Delta_eff=0
    to ~0 for Delta_eff >> fss_scale.
    """
    if fss_scale_eV <= 0.0:
        return 0.0
    x = abs(float(delta_eff_eV)) / float(fss_scale_eV)
    return 0.25 * np.pi * float(np.exp(-x))


def _theta_from_fss_and_delta_prime(
    fss_eV: float, delta_prime_eV: float
) -> float:
    """
    For exciton 2x2 Hamiltonian in {X1, X2} basis with:
      diag: +fss/2, -fss/2
      offdiag: delta_prime (real)
    the eigenvectors rotate by:
      theta = 0.5 * arctan2(2*delta_prime, fss)
    """
    return 0.5 * float(np.arctan2(2.0 * float(delta_prime_eV), float(fss_eV)))


@dataclass(frozen=True)
class DipoleParams:
    """
    Transition dipole moment parameters (unit-aware).

    Interpretation for directed transition `tr`:
      d_vec(tr) = mu(pair(tr)) * e_pol(pair(tr))

    We store by TransitionPair so forward/backward remain consistent.

    fss_model / delta_prime_model:
      Stored "assumptions" used by convenience constructors so you can later
      verify the dipole construction is consistent with EnergyStructure / mixing.
    """

    mu_default: Any = field(default_factory=lambda: as_Cm(1e-29))
    mu_by_pair: Dict[TransitionPair, Any] = field(default_factory=dict)

    pol_hv_by_pair: Dict[TransitionPair, Tuple[complex, complex]] = field(
        default_factory=dict
    )

    default_pol: JonesState = field(default_factory=JonesState.H)

    # Stored modeling assumptions for consistency checks
    fss_model: Any = field(default_factory=lambda: as_eV(0.0))
    delta_prime_model: Any = field(default_factory=lambda: as_eV(0.0))

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
        fss_model: Any = 0.0,
        delta_prime_model: Any = 0.0,
    ) -> "DipoleParams":
        dp = cls(
            mu_default=as_Cm(mu_default_Cm),
            mu_by_pair={p: as_Cm(v) for p, v in (mu_by_pair_Cm or {}).items()},
            pol_hv_by_pair=dict(pol_hv_by_pair or {}),
            default_pol=(
                default_pol if default_pol is not None else JonesState.H()
            ),
            fss_model=as_eV(fss_model),
            delta_prime_model=as_eV(delta_prime_model),
        )
        dp.validate()
        return dp

    @classmethod
    def biexciton_cascade_from_fss(
        cls,
        *,
        fss: Any,
        delta_prime: Any = 0.0,
        fss_scale: Any = 1e-6,
        # Optional extra rotation for "crystal axis" etc (applied on top)
        theta_extra_rad: float = 0.0,
        mu_default_Cm: Any = 1e-29,
        mu_by_pair_Cm: Optional[Mapping[TransitionPair, Any]] = None,
        x1_is_p1: bool = True,
    ) -> "DipoleParams":
        fss_q = as_eV(fss).to("eV")
        dp_q = as_eV(delta_prime).to("eV")
        scale_q = as_eV(fss_scale).to("eV")

        fss_eV = float(fss_q.magnitude)
        dp_eV = float(dp_q.magnitude)
        scale_eV = float(scale_q.magnitude)

        # Effective splitting magnitude (for "how linear vs circular" heuristic)
        delta_eff_eV = float(
            np.sqrt(fss_eV * fss_eV + (2.0 * dp_eV) * (2.0 * dp_eV))
        )

        # Rotation from exciton mixing + optional extra rotation
        theta_mix = _theta_from_fss_and_delta_prime(fss_eV, dp_eV)
        theta = float(theta_mix + float(theta_extra_rad))

        # Ellipticity from effective splitting
        alpha = _alpha_from_delta_eff(delta_eff_eV, scale_eV)

        p1, p2 = _elliptic_pair(alpha_rad=alpha, theta_rad=theta)

        if not x1_is_p1:
            p1, p2 = p2, p1

        pol = {
            TransitionPair.G_X1: (complex(p1[0]), complex(p1[1])),
            TransitionPair.G_X2: (complex(p2[0]), complex(p2[1])),
            TransitionPair.X1_XX: (complex(p2[0]), complex(p2[1])),
            TransitionPair.X2_XX: (complex(p1[0]), complex(p1[1])),
        }

        return cls.from_values(
            mu_default_Cm=mu_default_Cm,
            mu_by_pair_Cm=mu_by_pair_Cm,
            pol_hv_by_pair=pol,
            default_pol=JonesState.H(),
            fss_model=fss_q,
            delta_prime_model=dp_q,
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

        # Stored assumptions sanity (finite)
        _ = float(self.fss_model.to("eV").magnitude)
        _ = float(self.delta_prime_model.to("eV").magnitude)

    def as_floats(self) -> Dict[str, Any]:
        return {
            "mu_default_Cm": float(self.mu_default.to("C*m").magnitude),
            "mu_by_pair_Cm": {
                p: float(mu.to("C*m").magnitude)
                for p, mu in self.mu_by_pair.items()
            },
            "pol_hv_by_pair": dict(self.pol_hv_by_pair),
            "fss_model_eV": float(self.fss_model.to("eV").magnitude),
            "delta_prime_model_eV": float(
                self.delta_prime_model.to("eV").magnitude
            ),
        }
