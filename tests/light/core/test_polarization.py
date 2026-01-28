import unittest

import numpy as np

from bec.light.core.polarization import (
    HV_BASIS,
    JonesMatrix,
    JonesState,
    effective_polarization,
)


class TestPolarization(unittest.TestCase):
    def assertComplexAllClose(
        self, a: np.ndarray, b: np.ndarray, tol: float = 1e-12
    ) -> None:
        self.assertEqual(a.shape, b.shape)
        diff = np.max(np.abs(a - b))
        self.assertLessEqual(float(diff), float(tol))

    def test_state_normalization(self) -> None:
        s = JonesState(jones=(2.0 + 0j, 0.0 + 0j), normalize=True)
        v = s.as_array()
        self.assertComplexAllClose(v, np.array([1.0 + 0j, 0.0 + 0j]))

        s2 = JonesState(jones=(2.0 + 0j, 0.0 + 0j), normalize=False)
        v2 = s2.as_array()
        self.assertComplexAllClose(v2, np.array([2.0 + 0j, 0.0 + 0j]))

    def test_effective_polarization_identity(self) -> None:
        s = JonesState.D()  # (1,1) normalized
        E = effective_polarization(pol_state=s, pol_transform=None)
        self.assertIsNotNone(E)
        self.assertAlmostEqual(float(np.linalg.norm(E)), 1.0, places=12)

    def test_rotation_90_deg_maps_H_to_V(self) -> None:
        s = JonesState.H()
        R = JonesMatrix.rotation(np.pi / 2.0)
        out = R.apply(s)
        self.assertComplexAllClose(
            out, np.array([0.0 + 0j, 1.0 + 0j]), tol=1e-12
        )

    def test_linear_polarizer_blocks_V(self) -> None:
        s = JonesState.V()
        P = JonesMatrix.linear_polarizer(0.0)  # pass H
        out = P.apply(s)
        self.assertComplexAllClose(out, np.array([0.0 + 0j, 0.0 + 0j]))

    def test_serialization_roundtrip_state(self) -> None:
        s = JonesState.R()
        d = s.to_dict()
        self.assertEqual(d["basis"], HV_BASIS)
        s2 = JonesState.from_dict(d)
        self.assertEqual(s2.basis, HV_BASIS)
        self.assertComplexAllClose(s.as_array(), s2.as_array())

    def test_serialization_roundtrip_matrix(self) -> None:
        M = JonesMatrix.qwp(theta_rad=0.123)
        d = M.to_dict()
        self.assertEqual(d["basis"], HV_BASIS)
        M2 = JonesMatrix.from_dict(d)
        self.assertEqual(M2.basis, HV_BASIS)
        self.assertComplexAllClose(M.J, M2.J)

    def test_basis_rejected(self) -> None:
        with self.assertRaises(ValueError):
            # type: ignore[arg-type]
            _ = JonesState(jones=(1.0 + 0j, 0.0 + 0j), basis="RL")
        with self.assertRaises(ValueError):
            _ = JonesMatrix(
                J=np.eye(2, dtype=np.complex128), basis="RL"
            )  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            _ = JonesState.from_dict(
                {
                    "type": "jones_state",
                    "basis": "RL",
                    "normalize": True,
                    "jones": [[1.0, 0.0], [0.0, 0.0]],
                }
            )
        with self.assertRaises(ValueError):
            _ = JonesMatrix.from_dict(
                {
                    "type": "jones_matrix",
                    "basis": "RL",
                    "J": [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
                }
            )


if __name__ == "__main__":
    unittest.main()
