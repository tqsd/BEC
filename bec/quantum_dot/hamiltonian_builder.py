from typing import Any, Dict, List
from photon_weave.extra import interpreter
from qutip import Qobj
import numpy as np

from scipy.constants import e as _e, hbar as _hbar

from bec.quantum_dot.kron_pad_utility import KronPad
from bec.quantum_dot.protocols import HamiltonianProvider


class HamiltonianBuilder(HamiltonianProvider):
    """
    Class handles building the hamiltoinan terms for quantum-dot
    with its light mode system.

    The builder uses:
    - a context dict of callables that produce local operations
    - a KronPad utility helper to embed local operators into the full
      Hilbert space,
    - an energy-level object for physical parameters,
    - an optional polarization mapping function.

    Provided terms include:
    - fss: exciton fine-structure splitting Hamiltonian (and delta_prime mix)
    - lmi: pre-mode light matter interaction (not used in the current model)
    - classical_2g_flip: two-photon flip term between G and XX
    - classical_2g_detuning: two-photon detuning term on XX

    Parameters:
    -----------
    context: dict[str, Callable]
        Mapping from symbolic labels to callables used by the
        `photon_weave.extra.interpreter`
    kron: KronPad
        Utility to pad local operators into the full Hilbert space
    energy_levels: EnergyLevels
        Object with `fss` attributes
    pm_map: Callable, optional
        A callable mapping a transition index to a ploarization label
        + or -.
    """

    def __init__(
        self, context: Dict[str, Any], kron: KronPad, energy_levels, pm_map
    ):
        self._ctx = context
        self._kron = kron
        self._EL = energy_levels
        self._pm = pm_map

    def _qobj(self, expr: tuple, dims: List[int]) -> Qobj:
        """
        Evaluates a symbolic expression and wraps it into a QuTiP Qobj.

        Parameters:
        -----------
        expr: tuple
            Symbolic expression for `photon_weave.extra.interpreter`
        dims: list[int]
            Local dimensions of the composite Hilbert space.

        Returns:
        --------
        qutip.Qobj
            CSR operator with dims

        Raises:
        -------
        KeyError
            If required context keys are missing.
        ValueError
            If the interpreter result cannot be shaped to (N,N)

        """
        arr = interpreter(expr, self._ctx, dims)
        return Qobj(np.array(arr), dims=[dims, dims]).to("csr")

    def fss(self, dims: List[int], time_unit_s: float) -> Qobj:
        r"""
        Builds the fine structure splitting Hamiltonian on the
        exciton manifold. It adds the oscilations between the
        non-degenerate energy levels.

        .. math::
            H_fss=(Delta/2)(|X_1\rangle\langle X_1| - |X_2\rangle\langle X_2|)

        scaled by the provided time units for solver consistency.

        Parameters:
        -----------
        dims: list[int]
            Dimensions of the full Hilbert Space
        time_unit_s: float
            Seconds per folver time unit

        Returns:
        --------
        qutip.Qobj
            Padded Hamiltoian in wrapped in Qobj (CSR)

        """
        Delta = self._EL.fss * _e / _hbar * time_unit_s
        Delta_p = self._EL.delta_prime * _e / _hbar * time_unit_s
        proj_X1 = self._ctx["s_X1_X1"]([])
        proj_X2 = self._ctx["s_X2_X2"]([])
        X1X2 = self._ctx["s_X1_X2"]([])
        X2X1 = self._ctx["s_X2_X1"]([])
        Hloc = (Delta / 2) * (proj_X1 - proj_X2) + (Delta_p / 2) * (X1X2 + X2X1)
        return self._qobj(self._kron.pad(Hloc, "i", -1), dims)

    def lmi(self, label: str, dims: List[int]) -> Qobj:
        """
        Light matter interacion Hamiltonian, not used by the model.
        """
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

    def classical_2g_flip(self, dims: List[int]) -> Qobj:
        r"""
        Two-photon flip term between G and XX.

        .. math::
            0.5 * (|G\rangle\langle XX| + |XX\rangle\langle G|)

        Parameters
        ----------
        dims: list[int]
            Local dimensions of the composite Hilbert space.

        Returns:
        --------
        qutip.Qobj
            Padded hamiltonian wrapped into Qobj (CSR)
        """
        Hloc = 0.5 * (self._ctx["s_G_XX"]([]) + self._ctx["s_XX_G"]([]))
        return self._qobj(self._kron.pad(Hloc, "i", -1), dims)

    def classical_2g_detuning(self, dims: List[int]) -> Qobj:
        r"""
        Two-photon detuning flip term between G and XX.

        .. math::
            0.5 * (|XX\rangle\langle XX|)

        Parameters
        ----------
        dims: list[int]
            Local dimensions of the composite Hilbert space.

        Returns:
        --------
        qutip.Qobj
            Padded hamiltonian wrapped into Qobj (CSR)
        """
        Hloc = 0.5 * self._ctx["s_XX_XX"]([])
        return self._qobj(self._kron.pad(Hloc, "i", -1), dims)
