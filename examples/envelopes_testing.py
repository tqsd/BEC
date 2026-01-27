from bec.units import Q
from bec.light.classical import gaussian_field_drive
from bec.light.classical.report import drive_report

drv = gaussian_field_drive(
    t0=Q(50, "ps"),
    fwhm=Q(20, "ps"),
    E0=Q(5e4, "V/m"),
    wavelength=Q(930, "nm"),
    delta_omega=Q(0, "rad/s"),
    label="pump",
)

txt = drive_report(
    drv,
    time_unit_s=float(Q(1, "ps").to("s").magnitude),
    sample_window=(0.0, 100.0),
    show_ascii_plot=True,
)

print(txt)
