from smef.core.units import Q

from bec.light.classical import gaussian_field_drive
from bec.light.core.polarization import JonesState


def main() -> None:
    # One solver unit = 1 ps
    time_unit_s = float(Q(1.0, "ps").to("s").magnitude)

    drive = gaussian_field_drive(
        t0=Q(50.0, "ps"),
        sigma=Q(8.0, "ps"),
        E0=Q(2.0e5, "V/m"),
        wavelength=Q(930.0, "nm"),
        delta_omega=Q(0.0, "rad/s"),
        phi0=0.0,
        pol_state=JonesState.D(),
        preferred_kind="1ph",
        label="gaussian_demo",
    )

    window = (0.0, 120.0)  # solver units (ps here)

    print("\n" + "=" * 90)
    print("PLAIN REPORT")
    print("=" * 90)
    print(
        drive.report_plain(
            time_unit_s=time_unit_s,
            sample_window=window,
            sample_points=1501,
            show_ascii_plot=True,
        )
    )

    print("\n" + "=" * 90)
    print("RICH REPORT")
    print("=" * 90)
    print(
        drive.report_rich(
            time_unit_s=time_unit_s,
            sample_window=window,
            sample_points=1501,
            show_ascii_plot=True,
        )
    )


if __name__ == "__main__":
    main()
