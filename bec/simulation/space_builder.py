from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.fock import Fock
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import PolarizationLabel
from qutip import Qobj
import numpy as np
from bec.quantum_dot.dot import QuantumDot
from bec.simulation.protocols import SpaceBuilder


class DefaultSpaceBuilder(SpaceBuilder):
    def build_space(
        self, qd: QuantumDot, trunc_per_pol: int
    ) -> tuple[list[Envelope], list[Fock], CompositeEnvelope]:
        envs: list[Envelope] = []
        for m in qd.modes.modes:
            env_h = Envelope()
            env_h.fock.dimensions = trunc_per_pol

            env_v = Envelope()
            env_v.polarization.state = PolarizationLabel.V
            env_v.fock.dimensions = trunc_per_pol

            # stash back on the mode so downstream utilities can find them
            m.containerHV = [env_h, env_v]
            envs.extend([env_h, env_v])

        focks = [e.fock for e in envs]
        cstate = CompositeEnvelope(qd.dot, *envs)
        cstate.combine(qd.dot, *focks)
        cstate.reorder(qd.dot, *focks)

        return envs, focks, cstate

    def build_qutip_space(
        self, cstate: CompositeEnvelope, qd_dot, focks: list[Fock]
    ) -> tuple[list[int], list[list[int]], Qobj]:
        cstate.reorder(qd_dot, *focks)
        qd_dot.expand()
        dimensions = [s.dimensions for s in [qd_dot, *focks]]
        dims = [dimensions, dimensions]
        rho0 = Qobj(np.array(cstate.product_states[0].state), dims=dims).to(
            "csr"
        )
        return dimensions, dims, rho0
