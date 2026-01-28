from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from smef.core.ir.terms import Term
from smef.core.model.protocols import TermCatalogProto


@dataclass(frozen=True)
class FrozenCatalog(TermCatalogProto):
    _terms: tuple[Term, ...]

    @property
    def all_terms(self) -> Sequence[Term]:
        return self._terms
