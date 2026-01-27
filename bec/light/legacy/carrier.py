from dataclasses import dataclass
from typing import Callable, Union
from bec.units import QuantityLike, as_quantity

ChirpFn = Callable[[QuantityLike], QuantityLike]  # t -> rad/s


@dataclass(frozen=True)
class Carrier:
    omega0: QuantityLike  # rad/s
    delta_omega: Union[QuantityLike, ChirpFn] = 0.0  # rad/s

    def omega(self, t: QuantityLike) -> QuantityLike:
        w0 = as_quantity(self.omega0, "rad/s")
        d = self.delta_omega
        if callable(d):
            return w0 + as_quantity(d(t), "rad/s")
        return w0 + as_quantity(d, "rad/s")
