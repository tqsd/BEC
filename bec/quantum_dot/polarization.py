from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from bec.quantum_dot.units import as_eV
from bec.units import QuantityLike, Q


@dataclass(frozen=True)
class PolarizationCoupling:
    theta: float
    phi: float
    Omega: QuantityLike  # eV

    def U_hv_to_pm(self) -> np.ndarray:
        th = float(self.theta)
        ph = float(self.phi)
        return np.array(
            [
                [np.cos(th), np.exp(1j * ph) * np.sin(th)],
                [-np.exp(-1j * ph) * np.sin(th), np.cos(th)],
            ],
            dtype=complex,
        )

    def U_pm_to_hv(self) -> np.ndarray:
        return self.U_hv_to_pm().conj().T

    def e_plus_hv(self) -> np.ndarray:
        v = self.U_pm_to_hv()[:, 0]
        return v / np.linalg.norm(v)

    def e_minus_hv(self) -> np.ndarray:
        v = self.U_pm_to_hv()[:, 1]
        return v / np.linalg.norm(v)


def exciton_hamiltonian_2x2(
    *, fss: QuantityLike, delta_prime: QuantityLike
) -> np.ndarray:
    # Delta is the FSS energy splitting in eV
    Delta = float(as_eV(fss).to("eV").magnitude)

    # allow complex anisotropic coupling (keeps phase info)
    delta = complex(as_eV(delta_prime).to("eV").magnitude)

    # H = [[ +Delta/2, delta ],
    #      [ conj(delta), -Delta/2 ]]
    return np.array(
        [[0.5 * Delta, delta], [np.conjugate(delta), -0.5 * Delta]],
        dtype=complex,
    )


def exciton_rotation_params(
    *, fss: QuantityLike, delta_prime: QuantityLike
) -> tuple[float, float, QuantityLike]:
    Delta = float(as_eV(fss).to("eV").magnitude)
    delta = complex(as_eV(delta_prime).to("eV").magnitude)

    # Omega = sqrt(Delta^2 + 4|delta|^2)
    Omega_eV = float(np.sqrt(Delta * Delta + 4.0 * (abs(delta) ** 2)))
    Omega = Q(Omega_eV, "eV")

    # theta = 0.5 * arctan2(2|delta|, Delta)
    theta = 0.5 * float(np.arctan2(2.0 * abs(delta), Delta))

    # phi = arg(delta)
    phi = float(np.angle(delta))
    return theta, phi, Omega
