import numpy as np


def extract_xx_peak_from_expect(
    *,
    tlist_solver: np.ndarray,
    expect: dict[str, np.ndarray],
    time_unit_s: float,
    t0_ns: float,
    sigma_ns: float,
    window: tuple[float, float] = (-2.0, 6.0),
) -> float:
    """
    Extract peak XX population in a window around the pulse.

    window is in units of sigma: (a, b) means [t0 + a*sigma, t0 + b*sigma].
    """
    pop_xx = expect.get("pop_XX", None)
    if pop_xx is None:
        return float("nan")

    t_s = np.asarray(tlist_solver, dtype=float) * float(time_unit_s)
    t_ns = t_s / 1.0e-9

    t0 = float(t0_ns)
    sig = float(sigma_ns)
    a, b = float(window[0]), float(window[1])

    lo = t0 + a * sig
    hi = t0 + b * sig
    m = (t_ns >= lo) & (t_ns <= hi)
    if not np.any(m):
        return float("nan")

    arr = np.asarray(pop_xx, dtype=float)
    return float(np.max(arr[m]))
