from __future__ import annotations

from collections.abc import Sequence

from bec.quantum_dot.enums import QDState, TransitionKind, TransitionPair


class TransitionsMixin:
    def drive_targets(self) -> Sequence[TransitionPair]:
        """
        Transition families considered for coherent drives.
        Filters by registry spec.drive_allowed.
        """
        out = []
        for tp in self.t_registry.pairs():
            if bool(self.t_registry.spec(tp).drive_allowed):
                out.append(tp)
        return tuple(out)

    def drive_kind(self, pair: TransitionPair) -> str:
        kind = self.t_registry.spec(pair).kind
        if kind is TransitionKind.DIPOLE_1PH:
            return "1ph"
        if kind is TransitionKind.EFFECTIVE_2PH:
            return "2ph"
        raise KeyError(pair)

    def omega_ref_rad_s(self, pair: TransitionPair) -> float:
        """
        Reference angular frequency for this TransitionPair in rad/s (float).
        Delegates to EnergiesMixin.omega_ref_rad_s, which already accepts TransitionPair.
        """
        return float(self.omega_ref_rad_s_energy(pair))  # type: ignore[misc]

    def _pair_endpoint_energies(
        self, pair: TransitionPair
    ) -> tuple[object, object]:
        if pair is TransitionPair.G_X1:
            return (self.energy(QDState.G), self.energy(QDState.X1))
        if pair is TransitionPair.G_X2:
            return (self.energy(QDState.G), self.energy(QDState.X2))
        if pair is TransitionPair.X1_XX:
            return (self.energy(QDState.X1), self.energy(QDState.XX))
        if pair is TransitionPair.X2_XX:
            return (self.energy(QDState.X2), self.energy(QDState.XX))
        if pair is TransitionPair.G_XX:
            return (self.energy(QDState.G), self.energy(QDState.XX))
        raise KeyError(pair)
