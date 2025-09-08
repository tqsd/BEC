from typing import List, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class QDTraces:
    t: np.ndarray
    classical: bool
    flying_labels: List[str]
    intrinsic_labels: List[str]
    qd: List[np.ndarray]  # [|G>, |X1>, |X2>, |XX>]
    fly_H: List[np.ndarray]
    fly_V: List[np.ndarray]
    out_H: List[np.ndarray]
    out_V: List[np.ndarray]
    omega: Optional[np.ndarray] = None  # if classical drive used
    # cumulative area if classical drive used
    area: Optional[np.ndarray] = None
