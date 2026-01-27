# bec/core/model/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from bec.core.model.protocols import (
    DriveDecodeContextProto,
    DriveStrengthModelProto,
    ModeRegistryProto,
    OpMaterializeContextProto,
    OperatorCatalogProto,
)


class CompilableModel(ABC):
    """
    Standard vocabulary / capability surface for simulation compilation.

    Terminology:
      - Catalog: static IR definitions (operators + meta), drive-agnostic.
      - Context: runtime glue that exposes adapters for a stage.
      - Adapter: interface object consumed by simulation stages.
    """

    # --- structure ---
    @property
    @abstractmethod
    def mode_registry(self) -> ModeRegistryProto:
        raise NotImplementedError

    # --- operators / IR ---
    @abstractmethod
    def hamiltonian_catalog(self) -> OperatorCatalogProto:
        raise NotImplementedError

    # Later you can add:
    # def collapse_catalog(self) -> OperatorCatalogProto: ...
    # def observables_catalog(self) -> Any: ...

    # --- materialization ---
    @property
    def materialize_context(self) -> Optional[OpMaterializeContextProto]:
        return None

    # --- drive decode ---
    @property
    def drive_decode_context(self) -> Optional[DriveDecodeContextProto]:
        return None

    # --- drive strength ---
    @property
    def drive_strength_model(self) -> Optional[DriveStrengthModelProto]:
        return None

    # --- reporting hook ---
    @property
    def derived_for_report(self) -> Optional[Any]:
        return None
