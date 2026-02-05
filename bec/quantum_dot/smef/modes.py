from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from smef.core.model.protocols import ModeRegistryProto

from bec.quantum_dot.enums import TransitionPair


@dataclass(frozen=True)
class QDModeKey:
    """
    Typed subsystem key for ModeRegistryProto.index_of().

    This is "wiring", not physics:
    - kind="qd" refers to the dot subsystem
    - kind="mode" refers to one emitted field mode, identified by (band, pol)

    band:
      - "GX" for the G<->X manifold photon (X->G emission)
      - "XX" for the X<->XX manifold photon (XX->X emission)

    pol:
      - "H" or "V"
    """

    kind: str
    band: str | None = None
    pol: str | None = None

    @staticmethod
    def qd() -> QDModeKey:
        return QDModeKey(kind="qd")

    @staticmethod
    def gx(pol: str) -> QDModeKey:
        return QDModeKey(kind="mode", band="GX", pol=pol)

    @staticmethod
    def xx(pol: str) -> QDModeKey:
        return QDModeKey(kind="mode", band="XX", pol=pol)


@dataclass(frozen=True)
class QDModes(ModeRegistryProto):
    """
    Compiler-facing subsystem ordering and dimensions.

    Fixed ordering:
      0: qd
      1: GX_H
      2: GX_V
      3: XX_H
      4: XX_V
    """

    fock_dim: int = 2

    def __post_init__(self) -> None:
        if int(self.fock_dim) < 1:
            raise ValueError("fock_dim must be >= 1")

    def dims(self) -> Sequence[int]:
        d = int(self.fock_dim)
        return (4, d, d, d, d)

    def index_of(self, key: Any) -> int:
        # Typed keys
        if isinstance(key, QDModeKey):
            if key.kind == "qd":
                return 0
            if key.kind == "mode":
                if key.band == "GX" and key.pol == "H":
                    return 1
                if key.band == "GX" and key.pol == "V":
                    return 2
                if key.band == "XX" and key.pol == "H":
                    return 3
                if key.band == "XX" and key.pol == "V":
                    return 4
            raise KeyError(key)

        # String shortcuts (handy in debugging and quick prototypes)
        if key == "qd":
            return 0
        if key == "GX_H":
            return 1
        if key == "GX_V":
            return 2
        if key == "XX_H":
            return 3
        if key == "XX_V":
            return 4

        # Optional: allow mapping from TransitionPair to a band (GX vs XX)
        # so callers can ask index_of((pair, pol)) or similar later.
        if isinstance(key, tuple) and len(key) == 2:
            pair, pol = key
            if isinstance(pair, TransitionPair):
                if pair in (TransitionPair.G_X1, TransitionPair.G_X2):
                    return (
                        1
                        if pol == "H"
                        else 2 if pol == "V" else _raise_key(key)
                    )
                if pair in (TransitionPair.X1_XX, TransitionPair.X2_XX):
                    return (
                        3
                        if pol == "H"
                        else 4 if pol == "V" else _raise_key(key)
                    )

        raise KeyError(key)

    @property
    def channels(self) -> Sequence[Any] | None:
        return ("qd", "GX_H", "GX_V", "XX_H", "XX_V")


def _raise_key(key: Any) -> int:
    raise KeyError(key)
