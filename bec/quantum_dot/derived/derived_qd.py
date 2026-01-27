from __future__ import annotations

from dataclasses import dataclass

from bec.quantum_dot.derived.hamiltonians import HamiltoniansMixin
from bec.quantum_dot.derived.modes import ModesMixin

from .core import DerivedQDBase
from .energies import EnergiesMixin
from .dipoles import DipolesMixin
from .polarization import PolarizationMixin
from .phonons import PhononsMixin
from .cavity import CavityMixin
from .rates import RatesMixin
from .fast import FastMixin
from .report import ReportMixin


@dataclass(frozen=True)
class DerivedQD(
    DerivedQDBase,
    EnergiesMixin,
    DipolesMixin,
    PolarizationMixin,
    PhononsMixin,
    CavityMixin,
    RatesMixin,
    FastMixin,
    ModesMixin,
    HamiltoniansMixin,
    ReportMixin,  # MUST BE LAST
):
    """
    Unitful derived quantities for a QuantumDot instance.
    All logic lives in mixins; this is the stable public façade.
    """

    pass
