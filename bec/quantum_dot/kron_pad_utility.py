import numpy as np

from bec.quantum_dot.protocols import ModeProvider


class KronPad:
    def __init__(self, mode_provider: ModeProvider):
        self._modes = mode_provider

    def pad(
        self, qd_op: str | np.ndarray, fock_op: str, position: int
    ) -> tuple:
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
