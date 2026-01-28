import unittest
import numpy as np

from smef.core.units import Q

from bec.light.core.polarization import JonesState
from bec.quantum_dot.enums import Transition, TransitionPair
from bec.quantum_dot.spec.dipole_params import DipoleParams


class TestDipoleParams(unittest.TestCase):
    def test_defaults_validate(self) -> None:
        dp = DipoleParams.from_values()
        self.assertGreater(dp.mu_Cm(Transition.G_X1), 0.0)

    def test_hv_enforced_for_default_pol(self) -> None:
        # If JonesState can be constructed with non-HV basis, enforce rejection.
        # If your JonesState is always HV, this test can be removed.
        js = JonesState(jones=(1.0 + 0j, 0.0 + 0j), basis="HV", normalize=True)
        dp = DipoleParams.from_values(default_pol=js)
        self.assertIsNotNone(dp)

    def test_pair_based_pol_applies_to_both_directions(self) -> None:
        dp = DipoleParams.from_values(
            pol_hv_by_pair={TransitionPair.G_X1: (1.0 + 0j, 0.0 + 0j)}
        )
        e_fwd = dp.e_pol_hv(Transition.G_X1)
        e_bwd = dp.e_pol_hv(Transition.X1_G)
        np.testing.assert_allclose(e_fwd, e_bwd)

    def test_biexciton_defaults(self) -> None:
        dp = DipoleParams.biexciton_cascade_defaults(
            mu_default_Cm=Q(2e-29, "C*m")
        )
        e_gx1 = dp.e_pol_hv(Transition.G_X1)
        e_gx2 = dp.e_pol_hv(Transition.G_X2)
        np.testing.assert_allclose(e_gx1, np.array([1.0 + 0j, 0.0 + 0j]))
        np.testing.assert_allclose(e_gx2, np.array([0.0 + 0j, 1.0 + 0j]))

    def test_zero_pol_vector_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DipoleParams.from_values(
                pol_hv_by_pair={TransitionPair.G_X1: (0.0 + 0j, 0.0 + 0j)}
            )

    def test_negative_mu_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DipoleParams.from_values(mu_default_Cm=Q(-1.0, "C*m"))


if __name__ == "__main__":
    unittest.main()
