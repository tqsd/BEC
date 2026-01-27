from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence


class HamiltonianTermProto(Protocol):
    @property
    def kind(self) -> Any: ...

    @property
    def label(self) -> str: ...

    @property
    def op(self) -> Any: ...

    @property
    def coeff(self) -> Any: ...

    @property
    def meta(self) -> Mapping[str, Any]: ...


class HamiltonianCatalogProto(Protocol):
    @property
    def all_terms(self) -> Sequence[HamiltonianTermProto]: ...


class ModeRegistryProto(Protocol):
    def num_modes(self) -> int: ...


class OpMaterializeContextProto(Protocol):
    def resolve_symbol(self, key: str, dims: Sequence[int]) -> Any: ...


class DriveStrengthModelProto(Protocol):
    def compute_drive_coeffs(
        self,
        *,
        resolved: Sequence[Any],
        tlist: Any,
        time_unit_s: float,
    ) -> Mapping[str, Any]: ...


class OperatorCatalogProto(Protocol):
    # The compiler only needs to iterate; concrete type can be your Term or Term-like.
    @property
    def all_terms(self) -> Sequence[Any]: ...


class DriveDecodeContextProto(Protocol):
    # keep this aligned with bec.simulation.drive_decode.protocols.DriveDecodeContext
    transitions: object
    pol: object | None
    phonons: object | None
    bandwidth: object | None
