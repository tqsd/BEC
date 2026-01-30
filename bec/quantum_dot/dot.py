from dataclasses import dataclass
from functools import cached_property
from typing import Mapping, Optional, Sequence

from smef.core.ir.terms import Term
from smef.core.model.protocols import (
    CompilableModelProto,
    CompileBundle,
    MaterializeBundle,
    TermCatalogProto,
)
from smef.core.units import QuantityLike
from smef.engine import UnitSystem

from bec.quantum_dot.enums import RateKey
from bec.quantum_dot.models.decay_model import DecayModel, DecayOutputs
from bec.quantum_dot.models.phonon_model import (
    NullPhononModel,
    PhononOutputs,
    PhononModelProto,
    PolaronLAPhononModel,
)
from bec.quantum_dot.smef.catalogs.observables import QDObservablesCatalog
from bec.quantum_dot.smef.derived_view import QDDerivedView
from bec.quantum_dot.smef.drives.emitter.emitter import QDDriveTermEmitter
from bec.quantum_dot.smef.materializer import default_qd_materializer
from bec.quantum_dot.smef.modes import QDModes
from bec.quantum_dot.spec.cavity_params import CavityParams
from bec.quantum_dot.spec.dipole_params import DipoleParams
from bec.quantum_dot.spec.energy_structure import EnergyStructure
from bec.quantum_dot.spec.exciton_mixing_params import ExcitonMixingParams
from bec.quantum_dot.spec.phonon_params import PhononModelKind, PhononParams
from bec.quantum_dot.transitions import (
    DEFAULT_TRANSITION_REGISTRY,
    TransitionRegistry,
)

from smef.core.model.protocols import CompileBundle
from bec.quantum_dot.smef.drives import (
    QDDriveDecodeContext,
    QDDriveDecoder,
    QDDriveStrengthModel,
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
    """
    Four-level quantum dot model (G, X1, X2, XX) compiled into SMEF IR.

    Architecture overview
    ---------------------
    This class is intentionally a thin, immutable container that wires together:

    1) Unitful specifications (inputs)
       - EnergyStructure: absolute level energies in eV
       - DipoleParams: dipole magnitudes and (optionally) polarization metadata
       - CavityParams: optional cavity enhancement parameters
       - PhononParams: optional phonon dressing/decoherence parameters
       - ExcitonMixingParams: optional exciton mixing (Hamiltonian-level)

    2) Unitful physical models (physics lives here)
       - DecayModel: computes radiative rates (unitful, 1/s)
       - PhononModel: computes polaron dressing and phonon-induced rates (unitful, 1/s)

       These models return *outputs* dataclasses (DecayOutputs, PhononOutputs).
       No SMEF IR is constructed inside the models.

    3) Derived views
       - DerivedQD: cached, query-friendly access to derived scalars and outputs
         used by the SMEF catalogs (e.g., rates, polaron factors).

    4) SMEF compilation (IR construction)
       - QDHamiltonianCatalog: produces Hamiltonian IR terms (operators + coeffs)
       - QDCollapseCatalog: produces collapse IR terms from qd.rates
       - QDDrive* pipeline: decodes user drives, applies renormalization/EID, emits drive terms
       - QDObservablesCatalog: provides standard observables

    Unit boundary
    -------------
    All inputs and model outputs are unitful. Conversion to float solver units
    happens only when building IR term coefficients / in backend adapters.

    Notes
    -----
    The TransitionRegistry defines allowed transition families and metadata
    (drive_allowed/decay_allowed). Catalogs must respect these constraints.
    """

    energy: EnergyStructure
    dipoles: DipoleParams
    cavity: Optional[CavityParams] = None
    phonons: Optional[PhononParams] = None
    mixing: Optional[ExcitonMixingParams] = None

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
    def decay_outputs(self) -> DecayOutputs:
        return self.decay_model.compute()

    @cached_property
    def phonon_model(self) -> PhononModelProto:
        if self.phonons is None or self.phonons.kind is PhononModelKind.NONE:
            return NullPhononModel()
        if self.phonons.kind is PhononModelKind.POLARON_LA:
            # If later you need exciton_split_rad_s, pass it here (or compute in DerivedQD)
            return PolaronLAPhononModel(
                params=self.phonons, transitions=self.transitions
            )

        # Safety fallback
        return NullPhononModel()

    @cached_property
    def phonon_outputs(self) -> PhononOutputs:
        if self.phonons is None:
            return PhononOutputs()
        return self.phonon_model.compute()

    # @cached_property
    # def derived(self) -> DerivedQD:
    #    return DerivedQD(qd=self)

    @cached_property
    def rates(self) -> Mapping[RateKey, QuantityLike]:
        out: dict[RateKey, QuantityLike] = {}
        out.update(self.decay_outputs.rates)
        out.update(self.phonon_outputs.rates)
        # validate
        for k, v in out.items():
            if float(v.to("1/s").magnitude) < 0.0:
                raise ValueError(f"Negative rate {k}: {v}")
        return out

    @cached_property
    def derived_view(self) -> QDDerivedView:
        return QDDerivedView(
            energy=self.energy,
            dipoles=self.dipoles,
            mixing=self.mixing,
            phonons=self.phonons,
            t_registry=self.transitions,
            decay_outputs=self.decay_outputs,
            phonon_outputs=self.phonon_outputs,
            rates=self.rates,
        )

    def materialize_bundle(self) -> MaterializeBundle:
        ctx = default_qd_materializer()
        from bec.quantum_dot.ops.symbols import qd_symbol_latex_map

        return MaterializeBundle(
            ops=ctx,
            meta={"render_latex": True, "symbol_latex": qd_symbol_latex_map()},
        )

    def compile_bundle(self, *, units: UnitSystem):
        modes = QDModes(fock_dim=2)

        decode_ctx = QDDriveDecodeContext(derived=self.derived_view)

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
