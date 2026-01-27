from __future__ import annotations

from bec.units import magnitude


def report_plain_impl(d) -> str:
    """
    d is a DerivedQD-like object that provides the public API:
      energies, freq(), omega(), wavelength_vacuum(), mu(), polaron_B,
      cavity_omega, cavity_kappa, t_registry
    """
    from bec.quantum_dot.enums import TransitionPair

    lines: list[str] = []
    add = lines.append

    add("========= === DerivedQD === =========---")

    E = d.energies
    add("Energies (eV):")
    add(f"  X1 = {magnitude(E['X1'], 'eV'):.6f} eV")
    add(f"  X2 = {magnitude(E['X2'], 'eV'):.6f} eV")
    add(f"  XX = {magnitude(E['XX'], 'eV'):.6f} eV")

    es = getattr(d.qd, "energy_structure", None)
    if es is not None:
        add("Derived (eV):")
        add(f"  exciton_center = {magnitude(es.exciton_center, 'eV'):.6f} eV")
        add(f"  fss           = {magnitude(es.fss, 'eV'):.6e} eV")
        add(f"  binding       = {magnitude(es.binding, 'eV'):.6e} eV")

    pol = getattr(d.qd, "polarization", None)
    if pol is not None:
        add("Polarization (exciton eigenbasis):")
        add(f"  theta = {float(pol.theta):.4f} rad")
        add(f"  phi   = {float(pol.phi):.4f} rad")
        add(f"  Omega = {magnitude(pol.Omega, 'eV'):.6e} eV")
        ep = pol.e_plus_hv()
        em = pol.e_minus_hv()
        add(f"  e_plus(H,V)  = ({ep[0]:.3g}, {ep[1]:.3g})")
        add(f"  e_minus(H,V) = ({em[0]:.3g}, {em[1]:.3g})")

    add("Transitions (forward direction):")
    for tp in (
        TransitionPair.G_X1,
        TransitionPair.G_X2,
        TransitionPair.X1_XX,
        TransitionPair.X2_XX,
        TransitionPair.G_XX,
    ):
        fwd, _ = d.t_registry.directed(tp)

        try:
            if getattr(d.qd, "polarization", None) is not None:
                p_plus, p_minus = d.overlap_to_pm(fwd)
                add(
                    f"            couples-to: {d.coupled_pm_label(fwd)
                                               } (p+={p_plus:.2f}, p-={p_minus:.2f})"
                )
        except Exception:
            pass

        nu = magnitude(d.freq(tp), "Hz")
        w = magnitude(d.omega(tp), "rad/s")
        lam_nm = magnitude(d.wavelength_vacuum(tp), "m") * 1e9
        add(
            f"  {tp.value:7s}  nu={nu:.3e} Hz   omega={
                w:.3e} rad/s   lambda={lam_nm:.2f} nm"
        )

        try:
            mu = magnitude(d.mu(fwd), "C*m")
            add(f"            mu({fwd.value}) = {mu:.3e} C*m")
        except Exception:
            pass

    add("Phonons:")
    add(f"  polaron_B <B>(t) = {float(d.polaron_B):.6f}")

    add("Cavity:")
    add(f"  wc = {magnitude(d.cavity_omega, 'rad/s'):.3e} rad/s")
    add(f"  k  = {magnitude(d.cavity_kappa, 'rad/s'):.3e} rad/s")

    # Modes / channels
    mr = getattr(d.qd, "mode_registry", None)
    res = getattr(mr, "resolution", None)
    entries = getattr(d, "mode_entries", []) or []
    res_s = (
        str(res.value)
        if hasattr(res, "value")
        else str(res) if res is not None else "unknown"
    )
    add(f"Modes (resolution={res_s}, N={len(entries)}):")
    entries = getattr(d, "mode_entries", []) or []
    if not entries:
        add("  status = none")
    else:
        for r in entries:
            idx = r.get("idx", 0)
            kind = r.get("kind", "?")
            pol = r.get("pol", "?")
            e = r.get("energy_eV", None)
            lam = r.get("wavelength_nm", None)

            if e is not None and lam is not None:
                add(
                    f"  [{idx:02d}] {kind}:{pol}   E={
                        e:.6f} eV   lambda={lam:.2f} nm"
                )
            elif e is not None:
                add(f"  [{idx:02d}] {kind}:{pol}   E={e:.6f} eV")
            else:
                add(f"  [{idx:02d}] {kind}:{pol}")

    return "\n".join(lines)
