from __future__ import annotations

import math
import unittest

from smef.core.units import Q

from bec.quantum_dot.enums import RateKey
from bec.quantum_dot.transitions import DEFAULT_TRANSITION_REGISTRY
from bec.quantum_dot.spec.phonon_params import (
    PhononParams,
    PhononCouplings,
    PhononModelKind,
    PolaronLAParams,
)
from bec.quantum_dot.models.phonon_model import PolaronLAPhononModel


class TestPhononParams(unittest.TestCase):
    def test_temperature_nonnegative(self) -> None:
        with self.assertRaises(ValueError):
            PhononParams(temperature=Q(-1.0, "K"))

    def test_couplings_nonnegative(self) -> None:
        with self.assertRaises(ValueError):
            PhononCouplings(phi_x1=-1.0)


class TestPolaronLAPhononModel(unittest.TestCase):
    def test_polaron_b_is_1_when_disabled(self) -> None:
        P = PhononParams(
            kind=PhononModelKind.POLARON_LA,
            polaron_la=PolaronLAParams(enable_polaron_renorm=False),
        )
        m = PolaronLAPhononModel(
            params=P, transitions=DEFAULT_TRANSITION_REGISTRY
        )
        self.assertEqual(m.polaron_b(s2=1.0), 1.0)

    def test_polaron_b_is_1_when_alpha_zero(self) -> None:
        P = PhononParams(
            kind=PhononModelKind.POLARON_LA,
            polaron_la=PolaronLAParams(
                enable_polaron_renorm=True,
                alpha=Q(0.0, "s**2"),
            ),
        )
        m = PolaronLAPhononModel(
            params=P, transitions=DEFAULT_TRANSITION_REGISTRY
        )
        self.assertEqual(m.polaron_b(s2=1.0), 1.0)

    def test_polaron_b_in_range(self) -> None:
        P = PhononParams(
            kind=PhononModelKind.POLARON_LA,
            temperature=Q(4.0, "K"),
            polaron_la=PolaronLAParams(
                enable_polaron_renorm=True,
                alpha=Q(1e-26, "s**2"),
                omega_c=Q(1e12, "rad/s"),
            ),
        )
        m = PolaronLAPhononModel(
            params=P, transitions=DEFAULT_TRANSITION_REGISTRY
        )
        B = m.polaron_b(s2=1.0)
        self.assertTrue(0.0 <= B <= 1.0)
        self.assertTrue(math.isfinite(B))

    def test_exciton_relaxation_requires_distinct_x1_x2(self) -> None:
        P = PhononParams(
            kind=PhononModelKind.POLARON_LA,
            couplings=PhononCouplings(phi_x1=1.0, phi_x2=1.0),
            polaron_la=PolaronLAParams(
                enable_exciton_relaxation=True,
                alpha=Q(1e-26, "s**2"),
                omega_c=Q(1e12, "rad/s"),
            ),
            temperature=Q(4.0, "K"),
        )
        m = PolaronLAPhononModel(
            params=P,
            transitions=DEFAULT_TRANSITION_REGISTRY,
            exciton_split_rad_s=Q(1e11, "rad/s"),
        )
        out = m.compute()
        self.assertNotIn(RateKey.PH_RELAX_X1_X2, out.rates)
        self.assertNotIn(RateKey.PH_RELAX_X2_X1, out.rates)

    def test_exciton_relaxation_appears_when_distinct_and_split_provided(
        self,
    ) -> None:
        # Choose larger alpha and higher T so rate is less likely to underflow to 0.
        P = PhononParams(
            kind=PhononModelKind.POLARON_LA,
            couplings=PhononCouplings(phi_x1=1.0, phi_x2=1.2),
            polaron_la=PolaronLAParams(
                enable_exciton_relaxation=True,
                alpha=Q(1e-22, "s**2"),
                omega_c=Q(1e12, "rad/s"),
            ),
            temperature=Q(10.0, "K"),
        )
        m = PolaronLAPhononModel(
            params=P,
            transitions=DEFAULT_TRANSITION_REGISTRY,
            exciton_split_rad_s=Q(1e11, "rad/s"),
        )
        out = m.compute()
        present = (RateKey.PH_RELAX_X1_X2 in out.rates) or (
            RateKey.PH_RELAX_X2_X1 in out.rates
        )
        self.assertTrue(present)

    def test_compute_includes_b_map_when_enabled(self) -> None:
        P = PhononParams(
            kind=PhononModelKind.POLARON_LA,
            polaron_la=PolaronLAParams(
                enable_polaron_renorm=True,
                alpha=Q(1e-26, "s**2"),
                omega_c=Q(1e12, "rad/s"),
            ),
        )
        m = PolaronLAPhononModel(
            params=P, transitions=DEFAULT_TRANSITION_REGISTRY
        )
        out = m.compute()
        self.assertIsInstance(out.b_polaron, dict)
        self.assertGreater(len(out.b_polaron), 0)


if __name__ == "__main__":
    unittest.main()
