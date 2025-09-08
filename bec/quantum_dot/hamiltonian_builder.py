from typing import Any, Dict, List
from photon_weave.extra import interpreter
from qutip import Qobj
import numpy as np

from bec.quantum_dot.kron_pad_utility import KronPad
from bec.quantum_dot.protocols import HamiltonianProvider


class HamiltonianBuilder(HamiltonianProvider):
    def __init__(
        self, context: Dict[str, Any], kron: KronPad, energy_levels, pm_map
    ):
        self._ctx = context
        self._kron = kron
        self._EL = energy_levels  # EnergyLevels instance
        self._pm = pm_map  # function idx->'+'|'-' or None

    def _qobj(self, expr: tuple, dims: List[int]) -> Qobj:
        arr = interpreter(expr, self._ctx, dims)
        return Qobj(np.array(arr), dims=[dims, dims]).to("csr")

    def fss(self, dims: List[int]) -> Qobj:
        Delta = self._EL.fss
        Delta_p = self._EL.delta_prime
        proj_X1 = self._ctx["s_X1_X1"]([])
        proj_X2 = self._ctx["s_X2_X2"]([])
        X1X2 = self._ctx["s_X1_X2"]([])
        X2X1 = self._ctx["s_X2_X1"]([])
        Hloc = (Delta / 2) * (proj_X1 - proj_X2) + (Delta_p / 2) * (X1X2 + X2X1)
        return self._qobj(self._kron.pad(Hloc, "i", -1), dims)

    def lmi(self, label: str, dims: List[int]) -> Qobj:
        idx = self._kron.by_label(label)
        ab = ["s_G_X1", "s_G_X2", "s_X1_XX", "s_X2_XX"]
        em = ["s_X1_G", "s_X2_G", "s_XX_X1", "s_XX_X2"]

        # determine which transitions this mode drives
        mode = [
            m
            for m in self._kron._modes.modes
            if getattr(m, "label", None) == label
        ][0]
        H_ints = []
        for i in mode.transitions:
            pm = self._pm(i)
            h = (
                "s_mult",
                1,
                (
                    "add",
                    self._kron.pad(em[i], f"a{pm}_dag", idx),
                    self._kron.pad(ab[i], f"a{pm}", idx),
                ),
            )
            H_ints.append(h)
        return self._qobj(("add", *H_ints), dims)

    def tpe(self, label: str, dims: List[int]) -> Qobj:
        i = self._kron.by_label(label)
        H_abs = self._kron.pad("s_G_XX", "aa", i)
        H_em = self._kron.pad("s_XX_G", "aa_dag", i)
        return self._qobj(("add", H_abs, H_em), dims)

    def classical_2g_flip(self, dims: List[int]) -> Qobj:
        Hloc = self._ctx["s_G_XX"]([]) + self._ctx["s_XX_G"]([])
        return self._qobj(self._kron.pad(Hloc, "i", -1), dims)

    def classical_2g_detuning(self, dims: List[int]) -> Qobj:
        Hloc = self._ctx["s_XX_XX"]([])
        return self._qobj(self._kron.pad(Hloc, "i", -1), dims)
