from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from smef.core.units import QuantityLike

from bec.core.units import as_dimless, as_K, as_rad_s, as_rate_1_s, as_s2


class PhononModelKind(str, Enum):
    """
    Selects which phonon model is active.

    Notes
    -----
    This is intentionally a small enum so that the QuantumDot can always own a
    phonon model instance, and you can swap implementations later without
    changing call sites.
    """

    NONE = "none"
    POLARON_LA = "polaron_la"


class SpectralDensityKind(str, Enum):
    """
    Spectral density form used for acoustic phonons.

    Current implementation focuses on a common deformation-potential LA form:

    .. math::

        J(omega) = alpha * omega^3 * exp(-(omega / omega_c)^2)

    where ``alpha`` has units of ``s^2`` and ``omega_c`` has units of ``rad/s``.
    """

    SUPER_OHMIC_GAUSSIAN = "superohmic_gaussian"


@dataclass(frozen=True)
class PhenomenologicalPhononParams:
    """
    Optional phenomenological knobs.

    These are not "physical polaron" results; they exist so you can:
    - debug pipelines
    - compare against literature parameterizations
    - add missing processes before you have full microscopic formulas

    Parameters
    ----------
    gamma_phi_x1, gamma_phi_x2, gamma_phi_xx:
        Pure dephasing rates for X1, X2, XX in ``1/s``.

    gamma_relax_x1_x2, gamma_relax_x2_x1:
        Constant exciton relaxation rates (if provided) in ``1/s``.

    gamma_phi_eid_scale:
        Dimensionless scaling applied to drive-induced dephasing in a
        downstream stage (if you use it).

    Notes
    -----
    These rates are only used if they are > 0, and they are independent of the
    polaron model.
    """

    gamma_phi_x1: QuantityLike = as_rate_1_s(0.0)
    gamma_phi_x2: QuantityLike = as_rate_1_s(0.0)
    gamma_phi_xx: QuantityLike = as_rate_1_s(0.0)

    gamma_relax_x1_x2: QuantityLike = as_rate_1_s(0.0)
    gamma_relax_x2_x1: QuantityLike = as_rate_1_s(0.0)

    gamma_phi_eid_scale: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma_phi_x1", as_rate_1_s(self.gamma_phi_x1))
        object.__setattr__(self, "gamma_phi_x2", as_rate_1_s(self.gamma_phi_x2))
        object.__setattr__(self, "gamma_phi_xx", as_rate_1_s(self.gamma_phi_xx))
        object.__setattr__(
            self, "gamma_relax_x1_x2", as_rate_1_s(self.gamma_relax_x1_x2)
        )
        object.__setattr__(
            self, "gamma_relax_x2_x1", as_rate_1_s(self.gamma_relax_x2_x1)
        )
        object.__setattr__(
            self, "gamma_phi_eid_scale", as_dimless(self.gamma_phi_eid_scale)
        )
        self.validate()

    def validate(self) -> None:
        checks = {
            "gamma_phi_x1": float(self.gamma_phi_x1.to("1/s").magnitude),
            "gamma_phi_x2": float(self.gamma_phi_x2.to("1/s").magnitude),
            "gamma_phi_xx": float(self.gamma_phi_xx.to("1/s").magnitude),
            "gamma_relax_x1_x2": float(
                self.gamma_relax_x1_x2.to("1/s").magnitude
            ),
            "gamma_relax_x2_x1": float(
                self.gamma_relax_x2_x1.to("1/s").magnitude
            ),
            "gamma_phi_eid_scale": float(self.gamma_phi_eid_scale),
        }
        for k, v in checks.items():
            if v < 0.0:
                raise ValueError(f"{k} must be non-negative")


@dataclass(frozen=True)
class PolaronLAParams:
    """
    Parameters for a deformation-potential LA polaron model.

    Model
    -----
    We use the common super-ohmic spectral density:

    .. math::

        J(omega) = alpha * omega^3 * exp(-(omega / omega_c)^2)

    Polaron dressing factor for a transition with displacement difference
    parameter ``s2`` is:

    .. math::

        <B>(T) = exp(-0.5 * s2 * Integral_0^infty [ J(omega)/omega^2 * coth(beta*hbar*omega/2) d(omega) ])

    This is implemented numerically using a dimensionless change of variables
    ``x = omega / omega_c`` for stability.

    Parameters
    ----------
    alpha:
        Strength parameter (units ``s^2``).
    omega_c:
        Cutoff frequency (units ``rad/s``).
    spectral_density:
        Currently only one implementation is supported, but this leaves room to
        extend later.
    enable_polaron_renorm:
        If False, returns <B> = 1.
    enable_exciton_relaxation:
        If True, the model may compute X1 <-> X2 relaxation rates if the
        exciton splitting is provided and the coupling distinguishes X1 and X2.
    enable_eid:
        If True, the model will expose an EID config for a downstream stage.
        The time-dependent scattering calculation is typically done where you
        already have drive envelopes.
    """

    enable_polaron_renorm: bool = True
    enable_exciton_relaxation: bool = False
    enable_eid: bool = False
    enable_polaron_scattering: bool = True

    spectral_density: SpectralDensityKind = (
        SpectralDensityKind.SUPER_OHMIC_GAUSSIAN
    )
    alpha: QuantityLike = as_s2(0.0)
    omega_c: QuantityLike = as_rad_s(1.0e12)

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", as_s2(self.alpha))
        object.__setattr__(self, "omega_c", as_rad_s(self.omega_c))
        self.validate()

    @property
    def alpha_s2(self) -> float:
        return float(self.alpha.to("s**2").magnitude)

    @property
    def omega_c_rad_s(self) -> float:
        return float(self.omega_c.to("rad/s").magnitude)

    def validate(self) -> None:
        if self.alpha_s2 < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.omega_c_rad_s < 0.0:
            raise ValueError("omega_c must be non-negative")
        if self.enable_polaron_renorm and self.omega_c_rad_s == 0.0:
            raise ValueError(
                "omega_c must be > 0 when polaron renorm is enabled"
            )


@dataclass(frozen=True)
class PhononCouplings:
    """
    Dimensionless phonon displacement parameters per QD state.

    The polaron model uses differences between states:

    .. math::

        s^2(i,j) = (phi_i - phi_j)^2

    To allow exciton relaxation X1 <-> X2, you must allow X1 and X2 to differ.

    Parameters
    ----------
    phi_g:
        Ground state displacement parameter.
    phi_x1, phi_x2:
        Exciton displacement parameters.
    phi_xx:
        Biexciton displacement parameter.

    Notes
    -----
    These are dimensionless here, by design. If later you want a more microscopic
    parameterization, you can add a separate coupling spec and map it into these
    effective displacements.
    """

    phi_g: float = 0.0
    phi_x1: float = 1.0
    phi_x2: float = 1.0
    phi_xx: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "phi_g", as_dimless(self.phi_g))
        object.__setattr__(self, "phi_x1", as_dimless(self.phi_x1))
        object.__setattr__(self, "phi_x2", as_dimless(self.phi_x2))
        object.__setattr__(self, "phi_xx", as_dimless(self.phi_xx))
        self.validate()

    def validate(self) -> None:
        for name, v in [
            ("phi_g", float(self.phi_g)),
            ("phi_x1", float(self.phi_x1)),
            ("phi_x2", float(self.phi_x2)),
            ("phi_xx", float(self.phi_xx)),
        ]:
            if v < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return {
            "phi_g": float(self.phi_g),
            "phi_x1": float(self.phi_x1),
            "phi_x2": float(self.phi_x2),
            "phi_xx": float(self.phi_xx),
        }


@dataclass(frozen=True)
class PhononParams:
    """
    Public phonon parameter block (unit-aware, validated).

    Contract
    --------
    - Everything here is unitful or dimensionless.
    - No float-only physics beyond trivial dimensionless parameters.
    - Models consume this and produce unitful outputs.

    Parameters
    ----------
    kind:
        Which model implementation to use.
    temperature:
        Lattice temperature in Kelvin.
    couplings:
        Effective displacement parameters per QD state.
    polaron_la:
        Parameters for the LA polaron model.
    phenomenological:
        Optional constant-rate knobs.

    Notes
    -----
    If ``kind == NONE``, the phonon model should behave as a no-op.
    """

    kind: PhononModelKind = PhononModelKind.POLARON_LA
    temperature: QuantityLike = as_K(4.0)

    couplings: PhononCouplings = field(default_factory=PhononCouplings)
    polaron_la: PolaronLAParams = field(default_factory=PolaronLAParams)
    phenomenological: PhenomenologicalPhononParams = field(
        default_factory=PhenomenologicalPhononParams
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature", as_K(self.temperature))
        self.validate()

    @property
    def temperature_K(self) -> float:
        return float(self.temperature.to("K").magnitude)

    def validate(self) -> None:
        if self.temperature_K < 0.0:
            raise ValueError("temperature must be non-negative")
        # sub-blocks validate themselves already
