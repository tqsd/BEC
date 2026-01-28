from __future__ import annotations

from dataclasses import dataclass

from smef.core.model.protocols import TermCatalogProto

from bec.quantum_dot.smef.catalogs.base import FrozenCatalog


@dataclass(frozen=True)
class QDHamiltonianCatalog(FrozenCatalog):
    @classmethod
    def from_qd(cls, qd, *, units) -> TermCatalogProto:
        # For now: no static Hamiltonian terms.
        # Later you can add detunings / FSS / exciton mixing etc.
        return cls(_terms=())
