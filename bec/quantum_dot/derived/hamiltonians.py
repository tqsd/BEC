from __future__ import annotations

from functools import cached_property
from typing import List

from bec.quantum_dot.me.hamiltonian_builder import (
    HamiltonianBuilder,
    HamiltonianCatalog,
)
from bec.quantum_dot.me.types import Term


class HamiltoniansMixin:
    """
    Exposes Hamiltonian operator catalogs (IR terms) for reporting/debugging.

    Provides:
      - h_catalog(time_unit_s)
      - h_static_terms
      - h_detuning_basis
      - h_coherence_basis
      - h_all_terms
    """

    @cached_property
    def _h_builder(self) -> HamiltonianBuilder:
        qd = self.qd
        return self.qd.hamiltonian_builder

    def h_catalog(self) -> HamiltonianCatalog:
        # time_unit_s affects static coefficients (Δ, Δ')
        return self.qd.hamiltonian_catalog()

    def h_static_terms(self, *, time_unit_s: float) -> List[Term]:
        return self.h_catalog().static_terms

    def h_detuning_basis(self) -> List[Term]:
        # independent of time scale
        return self._h_builder.build_catalog().detuning_basis

    def h_coherence_basis(self) -> List[Term]:
        return self._h_builder.build_catalog().coherence_basis

    def h_all_terms(self, *, time_unit_s: float) -> List[Term]:
        return self.h_catalog().all_terms
