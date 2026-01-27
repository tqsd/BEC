import unittest
from types import SimpleNamespace

from bec.params.energy_levels import EnergyLevels
from bec.params.phonon_params import PhononParams, PhononModelType
from bec.params.transitions import TransitionType
from bec.quantum_dot.dot import QuantumDot


class TestQuantumDotWiring(unittest.TestCase):
    # ---------- helpers ----------

    def _make_energy_levels(self) -> EnergyLevels:
        # Disable 2-gamma guard for tests (avoid unrelated validation failures)
        return EnergyLevels(
            biexciton=2.0 * 1.300 - 3e-3,
            exciton=1.300,
            fss=5e-6,
            delta_prime=0.0,
            enforce_2g_guard=False,
        )

    def _make_phonon_params(self) -> PhononParams:
        # Safe default: POLARON enabled but alpha=0 => B=1.0, no changes
        return PhononParams(
            model=PhononModelType.POLARON,
            temperature_K=4.0,
            enable_polaron_renorm=True,
            alpha_s2=0.0,
            omega_c_rad_s=1.0e12,
        )

    def _make_dipole_params(self):
        # DecayModel expects dipole_params.dipole_moment_Cm
        return SimpleNamespace(dipole_moment_Cm=1e-29)

    def _make_cavity_params(self):
        # DecayModel expects cavity.n, cavity.Veff_um3, cavity.Q
        return SimpleNamespace(n=3.5, Veff_um3=1.0, Q=10_000)

    def _make_external_mode(self, label="ext", energy_ev=1.3):
        # ModeRegistry/register_external typically needs at least label/source/energy.
        # Add label_tex/transition for safety across your codebase.
        return SimpleNamespace(
            label=str(label),
            label_tex=str(label),
            source=TransitionType.EXTERNAL,
            energy_ev=float(energy_ev),
            transition=None,
        )

    # ---------- tests ----------

    def test_init_wires_core_subsystems(self):
        qd = QuantumDot(
            energy_levels=self._make_energy_levels(),
            phonon_params=self._make_phonon_params(),
            cavity_params=self._make_cavity_params(),
            dipole_params=self._make_dipole_params(),
        )

        # Core wiring
        self.assertIs(qd.energy_levels, qd.decay_model.el)
        self.assertIsNotNone(qd.modes)
        self.assertIsNotNone(qd.context_builder)
        self.assertIsNotNone(qd.kron)

        # Context exists and is a dict (non-empty is a good sanity check)
        self.assertIsInstance(qd.context, dict)
        self.assertGreater(len(qd.context), 0)

        # Builders exist
        self.assertIsNotNone(qd.h_builder)
        self.assertIsNotNone(qd.c_builder)
        self.assertIsNotNone(qd.o_builder)

        # Models exist
        self.assertIsNotNone(qd.decay_model)
        self.assertIsNotNone(qd.phonon_model)

    def test_init_requires_dipole_params_if_decay_model_requires_it(self):
        # Your DecayModel raises ValueError if dipole_params is missing.
        # This test protects that wiring expectation (and documents it).
        with self.assertRaises(ValueError):
            QuantumDot(
                energy_levels=self._make_energy_levels(),
                phonon_params=self._make_phonon_params(),
                cavity_params=self._make_cavity_params(),
                dipole_params=None,
            )

    def test_models_are_callable_and_return_expected_shapes(self):
        qd = QuantumDot(
            energy_levels=self._make_energy_levels(),
            phonon_params=self._make_phonon_params(),
            cavity_params=self._make_cavity_params(),
            dipole_params=self._make_dipole_params(),
        )

        gammas = qd.decay_model.compute()
        self.assertIsInstance(gammas, dict)
        self.assertEqual(
            set(gammas.keys()), {"L_XX_X1", "L_XX_X2", "L_X1_G", "L_X2_G"}
        )
        for k, v in gammas.items():
            self.assertIsInstance(v, float, msg=f"{k} should be float")
            self.assertGreaterEqual(v, 0.0, msg=f"{k} should be >= 0")

        phon = qd.phonon_model.compute()
        # We don't import PhononOutputs here to keep tests decoupled; just check attributes.
        self.assertTrue(hasattr(phon, "B_polaron"))
        self.assertTrue(hasattr(phon, "rates_1_s"))
        self.assertIsInstance(phon.B_polaron, float)
        self.assertIsInstance(phon.rates_1_s, dict)

    def test_custom_pm_map_is_stored_and_passed_to_builders_when_exposed(self):
        def custom_pm_map(idx: int) -> str:
            return "X"

        qd = QuantumDot(
            energy_levels=self._make_energy_levels(),
            phonon_params=self._make_phonon_params(),
            cavity_params=self._make_cavity_params(),
            dipole_params=self._make_dipole_params(),
            pm_map=custom_pm_map,
        )
        self.assertIs(qd.pm_map, custom_pm_map)

        # Builders *should* receive pm_map; if they store it, we can sanity check.
        # (If they don't expose it, we don't fail the test.)
        hb_pm = getattr(qd.h_builder, "pm_map", None)
        cb_pm = getattr(qd.c_builder, "pm_map", None)
        if hb_pm is not None:
            self.assertIs(hb_pm, custom_pm_map)
        if cb_pm is not None:
            self.assertIs(cb_pm, custom_pm_map)

    def test_register_external_mode_rebuilds_context_and_builders(self):
        qd = QuantumDot(
            energy_levels=self._make_energy_levels(),
            phonon_params=self._make_phonon_params(),
            cavity_params=self._make_cavity_params(),
            dipole_params=self._make_dipole_params(),
        )

        old_context_id = id(qd.context)
        old_kron_id = id(qd.kron)
        old_hb_id = id(qd.h_builder)
        old_cb_id = id(qd.c_builder)
        old_ob_id = id(qd.o_builder)

        # Models should NOT be rebuilt on topology changes
        old_decay_id = id(qd.decay_model)
        old_phonon_id = id(qd.phonon_model)

        old_mode_count = len(getattr(qd.modes, "modes", []))

        qd.register_external_mode(
            self._make_external_mode(label="fly0", energy_ev=1.31)
        )

        # Mode count increases (best-effort: ModeRegistry exposes .modes)
        new_mode_count = len(getattr(qd.modes, "modes", []))
        self.assertGreater(new_mode_count, old_mode_count)

        # Context + kron + builders rebuilt
        self.assertNotEqual(id(qd.context), old_context_id)
        self.assertNotEqual(id(qd.kron), old_kron_id)
        self.assertNotEqual(id(qd.h_builder), old_hb_id)
        self.assertNotEqual(id(qd.c_builder), old_cb_id)
        self.assertNotEqual(id(qd.o_builder), old_ob_id)

        # Models unchanged
        self.assertEqual(id(qd.decay_model), old_decay_id)
        self.assertEqual(id(qd.phonon_model), old_phonon_id)

        # Context is still valid and non-empty after rebuild
        self.assertIsInstance(qd.context, dict)
        self.assertGreater(len(qd.context), 0)

    def test_registering_multiple_external_modes_is_stable(self):
        qd = QuantumDot(
            energy_levels=self._make_energy_levels(),
            phonon_params=self._make_phonon_params(),
            cavity_params=self._make_cavity_params(),
            dipole_params=self._make_dipole_params(),
        )

        qd.register_external_mode(
            self._make_external_mode(label="fly0", energy_ev=1.31)
        )
        ctx1 = qd.context
        qd.register_external_mode(
            self._make_external_mode(label="fly1", energy_ev=1.29)
        )
        ctx2 = qd.context

        self.assertIsInstance(ctx1, dict)
        self.assertIsInstance(ctx2, dict)
        self.assertNotEqual(id(ctx1), id(ctx2))
        self.assertGreater(len(ctx2), 0)

    def test_context_property_is_live_reference(self):
        qd = QuantumDot(
            energy_levels=self._make_energy_levels(),
            phonon_params=self._make_phonon_params(),
            cavity_params=self._make_cavity_params(),
            dipole_params=self._make_dipole_params(),
        )

        c0 = qd.context
        qd.register_external_mode(
            self._make_external_mode(label="fly0", energy_ev=1.31)
        )
        c1 = qd.context

        self.assertNotEqual(id(c0), id(c1))
        self.assertIs(c1, qd._context)  # property returns the backing dict


if __name__ == "__main__":
    unittest.main()
