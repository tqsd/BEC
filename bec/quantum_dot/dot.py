from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

from smef.core.ir.terms import Term
from smef.core.model.protocols import (
    CompilableModelProto,
    CompileBundle,
    MaterializeBundle,
    TermCatalogProto,
)
from smef.engine import UnitSystem

from bec.quantum_dot.derived import DerivedQD
from bec.quantum_dot.models.decay_model import DecayModel
from bec.quantum_dot.smef.catalogs.observables import QDObservablesCatalog
from bec.quantum_dot.smef.materializer import default_qd_materializer
from bec.quantum_dot.smef.modes import QDModes
from bec.quantum_dot.spec import energy_structure
from bec.quantum_dot.spec.cavity_params import CavityParams
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.phonon_params import PhononParams
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    TransitionRegistry,
)

from smef.core.model.protocols import CompileBundle
from bec.quantum_dot.smef.drives import (
    QDDriveDecodeContext,
    QDDriveDecoder,
    QDDriveStrengthModel,
    QDDriveTermEmitter,
)

from bec.quantum_dot.smef.catalogs import (
    QDHamiltonianCatalog,
    QDCollapseCatalog,
)


@dataclass(frozen=True)
class FrozenCatalog(TermCatalogProto):
    _terms: tuple[Term, ...]

    @property
    def all_terms(self) -> Sequence[Term]:
        return self._terms


@dataclass(frozen=True)
class QuantumDot(CompilableModelProto):
    energy: EnergyStructure
    dipoles: DipoleParams
    cavity: Optional[CavityParams] = None
    phonons: Optional[PhononParams] = None

    transitions: TransitionRegistry = DEFAULT_TRANSITION_REGISTRY

    @cached_property
    def decay_model(self) -> DecayModel:
        return DecayModel(
            energy_structure=self.energy,
            dipole_params=self.dipoles,
            cavity_params=self.cavity,
            transitions=self.transitions,
        )

    @cached_property
    def derived(self) -> DerivedQD:
        return DerivedQD(qd=self)

    def materialize_bundle(self) -> MaterializeBundle:
        ctx = default_qd_materializer()
        return MaterializeBundle(ops=ctx)

    def compile_bundle(self, *, units: UnitSystem):
        modes = QDModes(fock_dim=2)

        derived = self.derived
        decode_ctx = QDDriveDecodeContext(derived=derived)

        return CompileBundle(
            modes=modes,
            hamiltonian=QDHamiltonianCatalog.from_qd(self, units=units),
            collapse=QDCollapseCatalog.from_qd(self, modes=modes, units=units),
            observables=QDObservablesCatalog(modes=modes),
            drive_decode=decode_ctx,
            drive_decoder=QDDriveDecoder(),
            drive_strength=QDDriveStrengthModel(),
            drive_emitter=QDDriveTermEmitter(),
        )
