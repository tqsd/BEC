from dataclasses import dataclass
from typing import List
import numpy as np
from qutip import Qobj
from .general import GeneralKrausChannel


@dataclass
class ChannelIO:
    @staticmethod
    def save_npz(path: str, ch: GeneralKrausChannel) -> None:
        data = {
            "dims_in": np.array(ch.dims_in, dtype=np.int64),
            "dims_out": np.array(ch.dims_out, dtype=np.int64),
            "n_kraus": np.array([len(ch.kraus)], dtype=np.int64),
        }
        for i, K in enumerate(ch.kraus):
            data[f"K_{i}_data"] = np.asarray(K.full())
            data[f"K_{i}_dims_in"] = np.array(ch.dims_in, dtype=np.int64)
            data[f"K_{i}_dims_out"] = np.array(ch.dims_out, dtype=np.int64)
        np.savez(path, **data)

    @staticmethod
    def load_npz(path: str) -> GeneralKrausChannel:
        z = np.load(path, allow_pickle=True)
        dims_in = z["dims_in"].tolist()
        dims_out = z["dims_out"].tolist()
        n = int(z["n_kraus"][0])
        kraus: List[Qobj] = []
        for i in range(n):
            arr = z[f"K_{i}_data"]
            K = Qobj(arr)
            K.dims = [[dims_out], [dims_in]]
            kraus.append(K)
        return GeneralKrausChannel(kraus, dims_in=dims_in, dims_out=dims_out)
