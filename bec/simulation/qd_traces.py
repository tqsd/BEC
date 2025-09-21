import csv
from typing import List, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class QDTraces:
    t: np.ndarray
    time_unit_s: float
    classical: bool
    flying_labels: List[str]
    intrinsic_labels: List[str]
    intrinsic_labels_tex: List[str]
    qd: List[np.ndarray]  # [|G>, |X1>, |X2>, |XX>]
    fly_H: List[np.ndarray]
    fly_V: List[np.ndarray]
    out_H: List[np.ndarray]
    out_V: List[np.ndarray]
    omega: Optional[np.ndarray] = None  # if classical drive used
    # cumulative area if classical drive used
    area: Optional[np.ndarray] = None

    def to_csv(self, path: str) -> None:
        """
        Save all traces into a single CSV file with all columns:
        time, Omega(t), Area(t), QD populations, flying/intrinsic modes.
        """
        t_ns = self.t * self.time_unit_s * 1e9  # convert to ns

        # --- Build header ---
        header = ["t [ns]"]

        # Classical drive
        if self.classical and self.omega is not None and self.area is not None:
            header += ["Omega(t) [rad/s]", "Area(t) [rad]"]

        # QD populations
        header += ["|G>|^2", "|X1>|^2", "|X2>|^2", "|XX>|^2"]

        # Flying modes
        for lbl in self.flying_labels:
            header.append(f"{lbl}_H")
            header.append(f"{lbl}_V")

        # Intrinsic modes
        for lbl in self.intrinsic_labels:
            header.append(f"{lbl}_out_H")
            header.append(f"{lbl}_out_V")

        # --- Collect data columns ---
        cols = [t_ns]

        if self.classical and self.omega is not None and self.area is not None:
            cols.append(self.omega)
            cols.append(self.area)

        cols.extend(self.qd)

        for arrs in (self.fly_H, self.fly_V, self.out_H, self.out_V):
            cols.extend(arrs)

        # --- Write file ---
        with open(f"{path}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in zip(*cols):
                writer.writerow(row)
