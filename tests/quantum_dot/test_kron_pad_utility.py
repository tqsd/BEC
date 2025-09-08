import unittest
import numpy as np
from types import SimpleNamespace

from bec.quantum_dot.kron_pad_utility import KronPad


class FakeModeProvider:
    def __init__(self, n):
        # Only .modes length is used by KronPad
        self.modes = [SimpleNamespace(label=f"m{i}") for i in range(n)]


class KronPadTests(unittest.TestCase):
    def setUp(self):
        self.nmodes = 3
        self.provider = FakeModeProvider(self.nmodes)
        self.kron = KronPad(self.provider)

    def _assert_pad(self, qd_op, fock_op, pos, expected_symbol):
        expr = self.kron.pad(qd_op, fock_op, pos)
        # tuple shape
        self.assertIsInstance(expr, tuple)
        self.assertEqual(expr[0], "kron")
        # qd_op is carried through as-is
        if isinstance(qd_op, np.ndarray):
            self.assertIs(expr[1], qd_op)
        else:
            self.assertEqual(expr[1], qd_op)
        # op list length equals number of modes
        op_list = expr[2:]
        self.assertEqual(len(op_list), self.nmodes)
        # check placement
        for i, sym in enumerate(op_list):
            if i == pos:
                self.assertEqual(sym, expected_symbol)
            else:
                self.assertEqual(sym, f"if{i}")

    def test_all_supported_fock_ops_map_correctly(self):
        qd_op_str = "s_X1_G"

        # Position 0
        self._assert_pad(qd_op_str, "a+", 0, "a0+")
        self._assert_pad(qd_op_str, "a+_dag", 0, "a0+_dag")
        self._assert_pad(qd_op_str, "a-", 0, "a0-")
        self._assert_pad(qd_op_str, "a-_dag", 0, "a0-_dag")
        self._assert_pad(qd_op_str, "aa", 0, "aa0")
        self._assert_pad(qd_op_str, "aa_dag", 0, "aa0_dag")
        self._assert_pad(qd_op_str, "n+", 0, "n0+")
        self._assert_pad(qd_op_str, "n-", 0, "n0-")
        self._assert_pad(qd_op_str, "i", 0, "if0")
        self._assert_pad(qd_op_str, "vac", 0, "vac0")

        # Position 2 (last)
        self._assert_pad(qd_op_str, "a+", 2, "a2+")
        self._assert_pad(qd_op_str, "a+_dag", 2, "a2+_dag")
        self._assert_pad(qd_op_str, "a-", 2, "a2-")
        self._assert_pad(qd_op_str, "a-_dag", 2, "a2-_dag")
        self._assert_pad(qd_op_str, "aa", 2, "aa2")
        self._assert_pad(qd_op_str, "aa_dag", 2, "aa2_dag")
        self._assert_pad(qd_op_str, "n+", 2, "n2+")
        self._assert_pad(qd_op_str, "n-", 2, "n2-")
        self._assert_pad(qd_op_str, "i", 2, "if2")
        self._assert_pad(qd_op_str, "vac", 2, "vac2")

    def test_qd_op_can_be_numpy_array_and_is_preserved(self):
        qd = np.eye(4)
        expr = self.kron.pad(qd, "a+", 1)
        self.assertEqual(expr[0], "kron")
        self.assertIs(expr[1], qd)  # same object
        # operator list: op in slot 1, "if{i}" elsewhere
        self.assertEqual(expr[3], "a1+")
        self.assertEqual(expr[2], "if0")
        self.assertEqual(expr[4], "if2")

    def test_unknown_fock_op_raises(self):
        with self.assertRaises(ValueError):
            _ = self.kron.pad("s_X2_G", "not_an_op", 1)

    def test_position_selects_correct_slot(self):
        # Sanity check across all positions
        for pos in range(self.nmodes):
            expr = self.kron.pad("s_G_X1", "n+", pos)
            op_list = expr[2:]
            for i, sym in enumerate(op_list):
                if i == pos:
                    self.assertEqual(sym, f"n{pos}+")
                else:
                    self.assertEqual(sym, f"if{i}")


if __name__ == "__main__":
    unittest.main()
