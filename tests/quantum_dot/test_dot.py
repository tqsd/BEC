import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bec.operators.qd_operators import QDState
from bec.quantum_dot import QuantumDot


class TestQuantumDot(unittest.TestCase):
    def setUp(self):
        # Minimal EnergyLevels stub with the attributes/methods the façade uses
        self.EL = SimpleNamespace()
        self.EL.compute_modes = MagicMock(
            return_value=[]
        )  # intrinsic modes list
        self.EL.exciton_rotation_params = MagicMock(
            return_value=(0.3, 1.2, [0.0, 0.0])
        )
        # X1, X2, XX attributes used to form el_dict
        self.EL.X1 = 1.31
        self.EL.X2 = 1.29
        self.EL.XX = 2.57

    @patch("bec.quantum_dot.dot.ObservablesBuilder")
    @patch("bec.quantum_dot.dot.CollapseBuilder")
    @patch("bec.quantum_dot.dot.HamiltonianBuilder")
    @patch("bec.quantum_dot.dot.DecayModel")
    @patch("bec.quantum_dot.dot.QDContextBuilder")
    @patch("bec.quantum_dot.dot.ModeRegistry")
    def test_init_wires_components_and_builds_context(
        self,
        ModeRegistryMock,
        QDContextBuilderMock,
        DecayModelMock,
        HamiltonianBuilderMock,
        CollapseBuilderMock,
        ObservablesBuilderMock,
    ):
        # Arrange: return values / instances
        mode_registry = ModeRegistryMock.return_value
        mode_registry.modes = []  # iterable in iteration later

        ctx_builder = QDContextBuilderMock.return_value
        ctx_builder.build.return_value = {"idq": lambda _: None}

        decay = DecayModelMock.return_value
        decay.compute.return_value = {"L_X1_G": 1.0}

        hams = HamiltonianBuilderMock.return_value
        collapses = CollapseBuilderMock.return_value
        observables = ObservablesBuilderMock.return_value

        # Act
        qd = QuantumDot(
            self.EL,
            cavity_params=None,
            dipole_params=None,
            time_unit_s=1e-9,
            initial_state=QDState.G,
        )

        # Assert: ModeRegistry constructed with intrinsic modes and rotation params
        self.EL.compute_modes.assert_called_once()
        self.EL.exciton_rotation_params.assert_called_once()
        args, _ = ModeRegistryMock.call_args
        self.assertEqual(args[0], [])  # intrinsic modes
        self.assertEqual(args[1], (0.3, 1.2))  # rotation params

        # Context builder used and context built
        QDContextBuilderMock.assert_called_once_with(mode_registry, 0.3, 1.2)
        ctx_builder.build.assert_called_once()
        self.assertIn("idq", qd.context)

        # DecayModel used and gammas computed
        DecayModelMock.assert_called_once()
        decay.compute.assert_called_once()
        self.assertEqual(qd.gammas, {"L_X1_G": 1.0})

        # Hamiltonian/collapse/observables builders are wired
        HamiltonianBuilderMock.assert_called_once()
        CollapseBuilderMock.assert_called_once_with(
            qd.gammas, qd.context, qd.kron, mode_registry
        )
        ObservablesBuilderMock.assert_called_once_with(
            qd.context, qd.kron, mode_registry
        )

        # Internal references
        self.assertIs(qd.hams, hams)
        self.assertIs(qd.collapses, collapses)
        self.assertIs(qd.obs, observables)

    @patch("bec.quantum_dot.dot.DecayModel")
    @patch("bec.quantum_dot.dot.QDContextBuilder")
    @patch("bec.quantum_dot.dot.ModeRegistry")
    def test_register_flying_mode_rebuilds_context(
        self, ModeRegistryMock, QDContextBuilderMock, DecayModelMock
    ):
        # Arrange façade
        ModeRegistryMock.return_value.modes = []
        QDContextBuilderMock.return_value.build.return_value = {
            "idq": lambda _: None
        }
        DecayModelMock.return_value.compute.return_value = {"L_X1_G": 1.0}

        qd = QuantumDot(
            self.EL,
            cavity_params=None,
            dipole_params=None,
            initial_state=QDState.G,
        )

        # Prepare new context to verify refresh
        QDContextBuilderMock.return_value.build.reset_mock()
        QDContextBuilderMock.return_value.build.side_effect = [
            {"idq": lambda _: "new"},  # after registering
        ]

        # Act
        lm = SimpleNamespace()  # pretend LightMode instance
        qd.register_flying_mode(light_mode=lm)

        # Assert registry and context refresh
        ModeRegistryMock.return_value.register_external.assert_called_once_with(
            lm
        )
        self.assertEqual(qd.context["idq"](None), "new")

    @patch("bec.quantum_dot.dot.ObservablesBuilder")
    @patch("bec.quantum_dot.dot.CollapseBuilder")
    @patch("bec.quantum_dot.dot.HamiltonianBuilder")
    @patch("bec.quantum_dot.dot.DecayModel")
    @patch("bec.quantum_dot.dot.QDContextBuilder")
    @patch("bec.quantum_dot.dot.ModeRegistry")
    def test_build_hamiltonians_single_and_tpe_modes(
        self,
        ModeRegistryMock,
        QDContextBuilderMock,
        DecayModelMock,
        HamiltonianBuilderMock,
        CollapseBuilderMock,
        ObservablesBuilderMock,
    ):
        # Arrange façade with two external modes
        ModeRegistryMock.return_value.modes = [
            # external 'single' mode
            SimpleNamespace(
                source="external",
                role="single",
                label="m_single",
                gaussian=lambda t: 2.0 * t,
                tpe_eliminated=set(),
                tpe_alpha_X1=0.0,
                tpe_alpha_X2=0.0,
            ),
            # external 'tpe' mode
            SimpleNamespace(
                source="external",
                role="tpe",
                label="m_tpe",
                gaussian=lambda t: t,
                tpe_eliminated={"X1"},
                tpe_alpha_X1=0.3,
                tpe_alpha_X2=0.0,
            ),
        ]
        QDContextBuilderMock.return_value.build.return_value = {
            "idq": lambda _: None
        }
        DecayModelMock.return_value.compute.return_value = {"L_X1_G": 1.0}

        hams = HamiltonianBuilderMock.return_value
        hams.fss.return_value = "H0"
        hams.lmi.return_value = "H_single"
        hams.tpe.return_value = "H_tpe"
        dims = [4, 2, 2]

        qd = QuantumDot(
            self.EL,
            cavity_params=None,
            dipole_params=None,
            initial_state=QDState.G,
        )

        # Act
        H = qd.build_hamiltonians(dims)

        # Assert structure: first static, then two time-dependent terms
        self.assertEqual(H[0], "H0")
        self.assertEqual(H[1][0], "H_single")
        self.assertTrue(callable(H[1][1]))
        self.assertEqual(H[2][0], "H_tpe")
        coeff2 = H[2][1]
        self.assertAlmostEqual(coeff2(2.0), 0.3 * (2.0**2))

        # Underlying builder calls
        hams.fss.assert_called_once_with(dims, 1.0)
        hams.lmi.assert_called_once_with("m_single", dims)
        hams.tpe.assert_called_once_with("m_tpe", dims)

    @patch("bec.quantum_dot.dot.ObservablesBuilder")
    @patch("bec.quantum_dot.dot.CollapseBuilder")
    @patch("bec.quantum_dot.dot.HamiltonianBuilder")
    @patch("bec.quantum_dot.dot.DecayModel")
    @patch("bec.quantum_dot.dot.QDContextBuilder")
    @patch("bec.quantum_dot.dot.ModeRegistry")
    def test_delegate_methods_for_collapse_and_observables(
        self,
        ModeRegistryMock,
        QDContextBuilderMock,
        DecayModelMock,
        HamiltonianBuilderMock,
        CollapseBuilderMock,
        ObservablesBuilderMock,
    ):
        ModeRegistryMock.return_value.modes = []
        QDContextBuilderMock.return_value.build.return_value = {
            "idq": lambda _: None
        }
        DecayModelMock.return_value.compute.return_value = {"L_X1_G": 1.0}

        CollapseBuilderMock.return_value.qutip_collapse_ops.return_value = [
            "C1",
            "C2",
        ]
        ObservablesBuilderMock.return_value.qd_projectors.return_value = {
            "P_G": "G"
        }
        ObservablesBuilderMock.return_value.light_mode_projectors.return_value = {
            "N[mode0]": "N0"
        }

        qd = QuantumDot(
            self.EL,
            cavity_params=None,
            dipole_params=None,
            initial_state=QDState.G,
        )

        dims = [4, 2, 2]
        C = qd.qutip_collapse_operators(dims)
        Pqd = qd.qutip_projectors(dims)
        Pm = qd.qutip_light_mode_projectors(dims)

        self.assertEqual(C, ["C1", "C2"])
        self.assertEqual(Pqd, {"P_G": "G"})
        self.assertEqual(Pm, {"N[mode0]": "N0"})

        CollapseBuilderMock.return_value.qutip_collapse_ops.assert_called_once_with(
            dims, 1.0
        )
        ObservablesBuilderMock.return_value.qd_projectors.assert_called_once_with(
            dims
        )
        ObservablesBuilderMock.return_value.light_mode_projectors.assert_called_once_with(
            dims
        )

    @patch("bec.quantum_dot.dot.DecayModel")
    @patch("bec.quantum_dot.dot.QDContextBuilder")
    @patch("bec.quantum_dot.dot.ModeRegistry")
    def test_context_property_returns_current_context(
        self, ModeRegistryMock, QDContextBuilderMock, DecayModelMock
    ):
        ModeRegistryMock.return_value.modes = []
        QDContextBuilderMock.return_value.build.return_value = {
            "idq": lambda _: "ctx1"
        }
        DecayModelMock.return_value.compute.return_value = {"L_X1_G": 1.0}

        qd = QuantumDot(
            self.EL,
            cavity_params=None,
            dipole_params=None,
            initial_state=QDState.G,
        )
        self.assertEqual(qd.context["idq"](None), "ctx1")

        # After registering a new mode, context should refresh
        QDContextBuilderMock.return_value.build.side_effect = [
            {"idq": lambda _: "ctx2"},
        ]
        qd.register_flying_mode(light_mode=SimpleNamespace())
        self.assertEqual(qd.context["idq"](None), "ctx2")


if __name__ == "__main__":
    unittest.main()
