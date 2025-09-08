# --- put these where you define your envelopes & drive ---
from typing import Dict, Any, Optional


def build_drive_from_signal(signal: Dict[str, Any]) -> ClassicalTwoPhotonDrive:
    """
    Build a ClassicalTwoPhotonDrive from a signal spec.

    Accepted forms:
      (A) Flat: { "type": "gaussian"|"tpe_gaussian"|"tabulated", ...,
                  "omega0": <float>, "detuning": <float>, "label": <str> }
      (B) Nested: { "omega0": <float>, "detuning": <float>, "label": <str>,
                    "envelope": { "type": ..., ... } }
    """
    # Common scalars (optional)
    omega0: float = float(signal.get("omega0", 1.0))
    detuning: float = float(signal.get("detuning", 0.0))
    label: Optional[str] = signal.get("label")

    # CASE (B): nested envelope
    if "envelope" in signal:
        env_json = signal["envelope"]
        env = envelope_from_json(env_json)  # uses your registry
        return ClassicalTwoPhotonDrive.from_envelope(
            envelope=env, omega0=omega0, detuning=detuning, label=label
        )

    # CASE (A): flat spec (treat the whole dict as an envelope JSON)
    if "type" in signal:
        # Copy only envelope-related fields into a clean dict
        env_keys = {"type", "t0", "sigma",
                    "area", "strength", "t", "y", "fwhm"}
        env_json = {k: signal[k] for k in signal.keys() & env_keys}
        env_json["type"] = signal["type"]  # ensure present

        # Optional: allow constructing Gaussian from FWHM
        if (
            env_json["type"] in {"gaussian", "tpe_gaussian"}
            and "fwhm" in env_json
            and "sigma" not in env_json
        ):
            # convert fwhm->sigma so envelope_from_json can stay simple
            fwhm = float(env_json.pop("fwhm"))
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            env_json["sigma"] = sigma

        env = envelope_from_json(env_json)
        return ClassicalTwoPhotonDrive.from_envelope(
            envelope=env, omega0=omega0, detuning=detuning, label=label
        )

    # Otherwise, not a recognized envelope spec
    raise ValueError(
        "Signal must contain either an 'envelope' block or a flat envelope with a 'type' field."
    )
