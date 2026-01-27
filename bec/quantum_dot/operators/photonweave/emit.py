from __future__ import annotations

from typing import Any, Callable, Optional

from bec.quantum_dot.ir.ops import (
    OpExpr,
    OpExprKind,
    EmbeddedKron,
    QDOpRef,
    FockOpRef,
    FockOpKind,
)
from bec.quantum_dot.mode_registry import ModeRegistry


# ---- PhotonWeave operator tokens (adjust if your interpreter uses different names) ----
PW_ADD = "add"
PW_MUL = "mult"
PW_SMULT = "smult"
PW_KRON = "kron"


def emit_op(op: OpExpr, modes: ModeRegistry) -> Any:
    """
    Emit a PhotonWeave expression tuple from a typed operator expression.

    Returns a nested tuple structure, e.g.:
      ("add",
         ("kron", "s_X1_G", "a0H_dag", "if1", ...),
         ("kron", "s_G_X1", "a0H",     "if1", ...),
      )
    """
    kind = op.kind

    if kind == OpExprKind.PRIMITIVE:
        if op.primitive is None:
            raise ValueError("PRIMITIVE OpExpr requires .primitive")
        return _emit_kron(op.primitive, modes)

    if kind == OpExprKind.SUM:
        if not op.args:
            raise ValueError("SUM OpExpr requires non-empty args")
        return (PW_ADD, *[emit_op(a, modes) for a in op.args])

    if kind == OpExprKind.PROD:
        if not op.args:
            raise ValueError("PROD OpExpr requires non-empty args")
        return (PW_MUL, *[emit_op(a, modes) for a in op.args])

    if kind == OpExprKind.SCALE:
        if op.scalar is None or len(op.args) != 1:
            raise ValueError(
                "SCALE OpExpr requires .scalar and exactly one arg"
            )
        return (PW_SMULT, op.scalar, emit_op(op.args[0], modes))

    raise ValueError(f"Unknown OpExprKind: {kind}")


def _qd_atom(qd: QDOpRef) -> Any:
    """
    PhotonWeave interpreter should accept either a symbol string or a raw matrix.
    """
    if qd.key is not None:
        return qd.key
    if qd.mat is not None:
        return qd.mat
    raise ValueError("QDOpRef has neither key nor mat")


def _fock_symbol(i: int, fock: FockOpRef) -> str:
    if fock.kind == FockOpKind.A:
        return f"a{i}"
    if fock.kind == FockOpKind.ADAG:
        return f"a{i}_dag"
    if fock.kind == FockOpKind.N:
        return f"n{i}"
    if fock.kind == FockOpKind.VAC:
        return f"vac{i}"
    raise ValueError(...)


def _emit_kron(prim: EmbeddedKron, modes: ModeRegistry) -> Any:
    qd_atom = _qd_atom(prim.qd)

    atoms: list[Any] = [PW_KRON, qd_atom]
    for i in range(len(modes.channels)):
        atoms.append(f"if{i}")

    if prim.fock is None:
        return tuple(atoms)

    idx = modes.index_of(prim.fock.key)
    atoms[2 + idx] = _fock_symbol(idx, prim.fock)
    return tuple(atoms)
