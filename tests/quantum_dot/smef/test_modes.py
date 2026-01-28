import unittest

from bec.quantum_dot.enums import TransitionPair
from bec.quantum_dot.smef.modes import QDModeKey, QDModes


class TestQDModes(unittest.TestCase):
    def test_dims(self) -> None:
        m = QDModes(fock_dim=2)
        self.assertEqual(tuple(m.dims()), (4, 2, 2, 2, 2))

        m3 = QDModes(fock_dim=3)
        self.assertEqual(tuple(m3.dims()), (4, 3, 3, 3, 3))

    def test_channels(self) -> None:
        m = QDModes(fock_dim=2)
        self.assertEqual(
            tuple(m.channels or ()), ("qd", "GX_H", "GX_V", "XX_H", "XX_V")
        )

    def test_index_of_typed_qd(self) -> None:
        m = QDModes(fock_dim=2)
        self.assertEqual(m.index_of(QDModeKey.qd()), 0)

    def test_index_of_typed_modes(self) -> None:
        m = QDModes(fock_dim=2)
        self.assertEqual(m.index_of(QDModeKey.gx("H")), 1)
        self.assertEqual(m.index_of(QDModeKey.gx("V")), 2)
        self.assertEqual(m.index_of(QDModeKey.xx("H")), 3)
        self.assertEqual(m.index_of(QDModeKey.xx("V")), 4)

    def test_index_of_string_shortcuts(self) -> None:
        m = QDModes(fock_dim=2)
        self.assertEqual(m.index_of("qd"), 0)
        self.assertEqual(m.index_of("GX_H"), 1)
        self.assertEqual(m.index_of("GX_V"), 2)
        self.assertEqual(m.index_of("XX_H"), 3)
        self.assertEqual(m.index_of("XX_V"), 4)

    def test_index_of_pair_mapping_gx(self) -> None:
        m = QDModes(fock_dim=2)
        self.assertEqual(m.index_of((TransitionPair.G_X1, "H")), 1)
        self.assertEqual(m.index_of((TransitionPair.G_X1, "V")), 2)
        self.assertEqual(m.index_of((TransitionPair.G_X2, "H")), 1)
        self.assertEqual(m.index_of((TransitionPair.G_X2, "V")), 2)

    def test_index_of_pair_mapping_xx(self) -> None:
        m = QDModes(fock_dim=2)
        self.assertEqual(m.index_of((TransitionPair.X1_XX, "H")), 3)
        self.assertEqual(m.index_of((TransitionPair.X1_XX, "V")), 4)
        self.assertEqual(m.index_of((TransitionPair.X2_XX, "H")), 3)
        self.assertEqual(m.index_of((TransitionPair.X2_XX, "V")), 4)

    def test_invalid_fock_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            QDModes(fock_dim=0)

        with self.assertRaises(ValueError):
            QDModes(fock_dim=-2)

    def test_invalid_key_raises_keyerror(self) -> None:
        m = QDModes(fock_dim=2)

        with self.assertRaises(KeyError):
            m.index_of("NOPE")

        with self.assertRaises(KeyError):
            m.index_of(("not_a_pair", "H"))

        with self.assertRaises(KeyError):
            m.index_of((TransitionPair.G_X1, "D"))  # invalid pol

        with self.assertRaises(KeyError):
            # wrong structure: tuple len != 2 should not match the special case
            m.index_of((TransitionPair.G_X1, "H", "extra"))

    def test_typed_key_invalid_raises(self) -> None:
        m = QDModes(fock_dim=2)

        with self.assertRaises(KeyError):
            m.index_of(QDModeKey(kind="mode", band="GX", pol="D"))

        with self.assertRaises(KeyError):
            m.index_of(QDModeKey(kind="mode", band="YY", pol="H"))

        with self.assertRaises(KeyError):
            m.index_of(QDModeKey(kind="something_else"))


if __name__ == "__main__":
    unittest.main()
