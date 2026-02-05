from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from smef.core.units import (
    Q,
    QuantityLike,
    as_quantity,
    magnitude,
)
from smef.core.units import (
    c as _c,
)
from smef.core.units import (
    epsilon_0 as _eps_0,
)
from smef.core.units import (
    hbar as _hbar,
)

from bec.quantum_dot.enums import QDState, Transition
from bec.quantum_dot.spec.cavity_params import CavityParams
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    RAD_RATE_TO_TRANSITION,
    RateKey,
    TransitionRegistry,
)


@dataclass(frozen=True)
class DecayOutputs:
    r"""
    Unitful outputs of the decay model.

    Attributes
    ----------
    rates:
        Mapping ``RateKey -> gamma`` where each gamma is unitful with units ``1/s``.

    Notes
    -----
    This is intentionally "dumb data": the physics lives in :class:`DecayModel`.

    Conversion boundary
    -------------------
    SMEF adapters typically want floats. Convert at the boundary via:

    .. code-block:: python

        floats = {k: float(v.to("1/s").magnitude) for k, v in outputs.rates.items()}

    """

    rates: dict[RateKey, QuantityLike] = field(default_factory=dict)

    def rates_1_s(self) -> Mapping[RateKey, float]:
        return {k: float(v.to("1/s").magnitude) for k, v in self.rates.items()}


@dataclass(frozen=True)
class DecayModel:
    r"""
    Radiative decay model for a four-level quantum dot with optional cavity enhancement.

    This model produces spontaneous emission rates for allowed optical transitions.

    Physics
    -------
    For an optical transition with angular frequency :math:`\omega` and dipole
    moment magnitude :math:`\mu`, the free-space spontaneous emission rate is:

    .. math::

        \gamma_0(\omega, \mu) =
            \frac{\omega^3 \mu^2}{3 \pi \epsilon_0 \hbar c^3}.

    If a cavity is present, we include a simple Purcell enhancement:

    .. math::

        \gamma = \gamma_0 (1 + F_P),

    with

    .. math::

        F_P = \frac{3}{4\pi^2} \left(\frac{\lambda}{n}\right)^3 \frac{Q}{V_\mathrm{eff}} ,

    and :math:`\lambda = 2\pi c / \omega`. This is a standard on-resonance,
    weak-coupling expression.

    Contract
    --------
    - Inputs are unitful (energies, dipoles).
    - Outputs are unitful rates in ``1/s``.
    - No float conversion is done except internally for numerical safety.
    - Registry rules are respected (``decay_allowed`` on a transition family).

    Parameters
    ----------
    energy_structure:
        Absolute level energies.
    dipole_params:
        Provides transition dipole moments (must implement ``mu(tr)``).
    cavity_params:
        Optional cavity parameters for Purcell enhancement.
    transitions:
        Transition registry providing endpoints and specs.

    Notes
    -----
    - The decay model uses *energy differences* to compute emitted photon energies.
      If the computed photon energy is non-positive, the decay rate is zero.
    - If the registry marks a family as ``decay_allowed=False``, decay for that
      directed transition is forced to zero.
    """

    energy_structure: EnergyStructure
    dipole_params: DipoleParams
    cavity_params: CavityParams | None = None
    transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY

    def level_energy(self, s: QDState) -> QuantityLike:
        """
        Return absolute energy of level ``s`` in eV.
        """
        return as_quantity(getattr(self.energy_structure, s.name), "eV").to(
            "eV"
        )

    def dE_photon(self, tr: Transition) -> QuantityLike:
        r"""
        Emitted photon energy for transition ``tr`` in eV.

        For a transition :math:`|i\rangle \to |f\rangle` the photon energy is:

        .. math::

            \Delta E = E_i - E_f.

        If :math:`\Delta E \le 0`, returns ``0 eV``.
        """
        i, f = self.transitions.endpoints(tr)
        dE = (self.level_energy(i) - self.level_energy(f)).to("eV")
        if magnitude(dE, "eV") <= 0.0:
            return Q(0.0, "eV")
        return dE

    def omega(self, tr: Transition) -> QuantityLike:
        r"""
        Angular frequency :math:`\omega` for emitted photon in ``rad/s``.

        .. math::

            \omega = \Delta E / \hbar.

        Returns 0 if :math:`\Delta E \le 0`.
        """
        dE = self.dE_photon(tr)
        if magnitude(dE, "eV") <= 0.0:
            return Q(0.0, "rad/s")
        return (dE.to("J") / _hbar).to("rad/s")

    def gamma0(self, tr: Transition) -> QuantityLike:
        r"""
        Free-space spontaneous emission rate :math:`\gamma_0` in ``1/s``.

        Uses:

        .. math::

            \gamma_0 =
                \frac{\omega^3 \mu^2}{3 \pi \epsilon_0 \hbar c^3}.

        Returns 0 if :math:`\omega \le 0` or :math:`\mu \le 0`.
        """
        omega = as_quantity(self.omega(tr), "rad/s")
        mu = as_quantity(self.dipole_params.mu(tr), "C*m")

        if magnitude(omega, "rad/s") <= 0.0:
            return Q(0.0, "1/s")
        if magnitude(mu, "C*m") <= 0.0:
            return Q(0.0, "1/s")

        num = omega**3 * mu**2
        den = 3.0 * np.pi * _eps_0 * _hbar * _c**3
        return (num / den).to("1/s")

    def purcell_factor(self, tr: Transition) -> float:
        r"""
        Purcell enhancement factor :math:`F_P` (dimensionless float).

        Uses:

        .. math::

            F_P = \frac{3}{4\pi^2}\left(\frac{\lambda}{n}\right)^3\frac{Q}{V_\mathrm{eff}}.

        Returns 0 if:
        - no cavity params
        - invalid cavity params
        - :math:`\omega \le 0`
        """
        cp = self.cavity_params
        if cp is None:
            return 0.0

        omega = self.omega(tr)
        if magnitude(omega, "rad/s") <= 0.0:
            return 0.0

        Qfac = float(getattr(cp, "Q", 0.0) or 0.0)
        n = float(getattr(cp, "n", 1.0) or 1.0)
        if Qfac <= 0.0 or n <= 0.0:
            return 0.0

        lam = (2.0 * np.pi * _c / omega).to("m")

        if hasattr(cp, "Veff_m3"):
            Vm = as_quantity(getattr(cp, "Veff_m3"), "m^3")
        else:
            Veff_um3 = float(getattr(cp, "Veff_um3", 0.0) or 0.0)
            if Veff_um3 <= 0.0:
                return 0.0
            Vm = Q(Veff_um3, "um^3").to("m^3")

        Fp_q = (3.0 / (4.0 * np.pi**2)) * (lam / n) ** 3 * (Qfac / Vm)
        Fp = float(Fp_q.to("").magnitude)

        if not np.isfinite(Fp) or Fp < 0.0:
            return 0.0
        return Fp

    def gamma(self, tr: Transition) -> QuantityLike:
        """
        Total radiative decay rate in ``1/s`` for directed transition ``tr``.

        Respects registry rules: if the transition family has ``decay_allowed=False``,
        the result is zero.
        """
        spec = self.transitions.spec(tr)
        if not spec.decay_allowed:
            return Q(0.0, "1/s")

        g0 = self.gamma0(tr)
        Fp = self.purcell_factor(tr)
        return (g0 * (1.0 + Fp)).to("1/s")

    def compute_rates(self) -> dict[RateKey, QuantityLike]:
        """
        Compute all registered radiative rates keyed by RateKey.
        """
        out: dict[RateKey, QuantityLike] = {}
        for key, tr in RAD_RATE_TO_TRANSITION.items():
            out[key] = self.gamma(tr)
        return out

    def compute(self) -> DecayOutputs:
        """
        Compute unitful decay outputs.
        """
        return DecayOutputs(rates=self.compute_rates())
