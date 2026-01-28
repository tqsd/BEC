from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np

from smef.core.ir.protocols import OpMaterializeContextProto

from bec.quantum_dot.smef.symbols import (
    SymbolLibrary,
    as_dims,
    build_default_symbol_library,
    canon_symbol,
    fock_ops,
)


@dataclass(frozen=True)
class QDOpMaterializeContext(OpMaterializeContextProto):
    """
    SMEF OpMaterializeContext for the QD repo.

    Conventions:
      - local symbols are resolved with dims = (local_dim,)
      - full-space symbols (SymbolOp) can be supported if you register them in the library
        with dims = full dims tuple (e.g. (4,2,2)). For now we focus on locals.
    """

    lib: SymbolLibrary
    allow_generic_fock: bool = True

    def resolve_symbol(self, symbol: Any, dims: Sequence[int]) -> np.ndarray:
        d = as_dims(dims)
        s = canon_symbol(symbol)

        # Optional: allow generic mode symbols for any (N,) without pre-registering N
        if (
            self.allow_generic_fock
            and len(d) == 1
            and d[0] >= 1
            and s in ("a", "adag", "n", "I")
        ):
            return np.asarray(fock_ops(int(d[0]))[s], dtype=complex)

        return np.asarray(self.lib.resolve(s, d), dtype=complex)

    def resolve_embedded(
        self, embedded: Any, dims: Sequence[int]
    ) -> np.ndarray:
        # Not used by smef.core.ir.materialize.materialize_op_expr right now.
        raise NotImplementedError(
            "EmbeddedKron is materialized in smef.core.ir.materialize"
        )


def default_qd_materializer(
    *, register_fock_dims: Sequence[int] = (2, 3, 4, 5)
) -> QDOpMaterializeContext:
    lib = build_default_symbol_library(
        register_fock_dims=tuple(int(x) for x in register_fock_dims)
    )
    return QDOpMaterializeContext(lib=lib, allow_generic_fock=True)
