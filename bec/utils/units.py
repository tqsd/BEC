import re


def parse_time_unit_s(u):
    if isinstance(u, (int, float)):
        return float(u)
    s = str(u).strip().lower().replace("µ", "u")
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-z]*)\s*", s)
    if not m:
        raise ValueError(f"Bad time unit: {u!r}")
    val, suf = float(m.group(1)), m.group(2)
    scale = {
        "": 1.0,
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
        "fs": 1e-15,
    }.get(suf)
    if scale is None:
        raise ValueError(f"Unsupported suffix {suf!r}")
    return val * scale
