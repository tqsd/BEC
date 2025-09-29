import numpy as np

from bec.quantum_dot.protocols import ModeProvider


class KronPad:
    """
    Builder for operators (embedded)

    Helper that helps convert local quantum-dot operator and
    single mode photonic operator at position `i` into a
    symbolic tuple that can be used by the expression interpreter
    `photon_weave.extras.interpreter`.

    Notes
    -----
    - This class only constructs the expressions
    - No algebra is performed here
    """

    def __init__(self, mode_provider: ModeProvider):
        self._modes = mode_provider

    def pad(
        self, qd_op: str | np.ndarray, fock_op: str, position: int
    ) -> tuple:
        """
        Builds a Kronecker product expression tuple with photonic
        operation applied at ghe given `position`.

        The `fock_op` selects which operator should be inserted
        at `position`. Implemented labels are:
        - "a+"      ->  f"a{position}+"
        - "a+_dag"  ->  f"a{position}+_dag"
        - "a-"      ->  f"a{position}-"
        - "a-_dag"  ->  f"a{position}-_dag"
        - "aa"      ->  f"aa{position}"
        - "aa_dag"  ->  f"aa{position}_dag"
        - "n+"      ->  f"n{position}+"
        - "n-"      ->  f"n{position}-"
        - "i"       ->  f"if{position}"
        - "vac"     ->  f"vac{position}"

        For all mode indices which are not the position, the identity
        placeholder is omitted.

        Parameters
        ----------
        qd_op: str or numpy.ndarray
            Quantum-dot operator. If a string, it should be resolved by the
            context, passed to the interpreter.
        fock_op: str
            selector for the photonic operator
        position: int
            Index of the selected photonic operator

        Returns
        -------
        tuple
            Symbolic expression

        Raises
        ------
        ValueError
            If `fock_op` is not recognized
        IndexError
            If the `position` is outside of scope
        """
        match fock_op:
            case "a+":
                op = f"a{position}+"
            case "a+_dag":
                op = f"a{position}+_dag"
            case "a-":
                op = f"a{position}-"
            case "a-_dag":
                op = f"a{position}-_dag"
            case "aa":
                op = f"aa{position}"
            case "aa_dag":
                op = f"aa{position}_dag"
            case "n+":
                op = f"n{position}+"
            case "n-":
                op = f"n{position}-"
            case "i":
                op = f"if{position}"
            case "vac":
                op = f"vac{position}"
            case _:
                raise ValueError(f"Unknown operator {fock_op}")
        op_order = [
            (
                op
                if i == position
                else f"if{
                    i}"
            )
            for i, _ in enumerate(self._modes.modes)
        ]
        return ("kron", qd_op, *op_order)
