from dataclasses import dataclass
from typing import List
from qutip import Qobj, qeye
import numpy as np

try:
    # QuTiP 5+
    from qutip import kraus_to_super  # type: ignore[attr-defined]
except Exception:
    try:
        # Some 5.x builds expose it here
        from qutip.core.superoperator import kraus_to_super  # type: ignore
    except Exception:
        # QuTiP 4.x fallback
        from qutip.superop_reps import kraus_to_super  # type: ignore


@dataclass(frozen=True)
class GeneralKrausChannel:
    kraus: List[Qobj]
    dims_in: List[int]
    dims_out: List[int]

    def apply(self, rho_in: Qobj) -> Qobj:
        """Apply the channel: rho_out = sum_K K rho_in K^†."""
        Nout = int(np.prod(self.dims_out))
        acc = Qobj(np.zeros((Nout, Nout), dtype=complex))
        for K in self.kraus:
            # Pyright doesn't know Qobj overloads; the operation is valid in QuTiP.
            acc = acc + (K * rho_in * K.dag())  # type: ignore[operator]
        return acc

    def check_cptp(self, tol: float = 1e-9) -> None:
        """Check trace-preserving: sum_K K^† K == I (within tol)."""
        Nin = int(np.prod(self.dims_in))
        acc = Qobj(np.zeros((Nin, Nin), dtype=complex))
        for K in self.kraus:
            acc = acc + (K.dag() * K)  # type: ignore[operator]
        if (acc - qeye(Nin)).norm() > tol:
            raise ValueError("Channel is not TP to tolerance")

    def as_super(self) -> Qobj:
        """Convert to a superoperator in column-stacking convention."""
        if self.dims_in != self.dims_out:
            raise ValueError(
                "Only square channels can be converted to superoperators"
            )
        return kraus_to_super(self.kraus)
