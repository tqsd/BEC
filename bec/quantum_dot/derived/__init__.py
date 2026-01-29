from dataclasses import dataclass

from bec.quantum_dot.derived.phonons import PhononsMixin
from bec.quantum_dot.derived.rates import RatesMixin

from .core import DerivedQDBase
from .dipoles import DipolesMixin
from .energies import EnergiesMixin
from .transitions import TransitionsMixin


@dataclass(frozen=True)
class DerivedQD(
    DerivedQDBase,
    EnergiesMixin,
    TransitionsMixin,
    DipolesMixin,
    RatesMixin,
    PhononsMixin,
):
    pass
