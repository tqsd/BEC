import unittest

from smef.core.units import Q

from bec.light.classical.factories import gaussian_field_drive


class TestDriveReportingSplit(unittest.TestCase):
    def test_report_plain_method(self) -> None:
        d = gaussian_field_drive(
            t0=Q(50.0, "ps"),
            sigma=Q(10.0, "ps"),
            E0=Q(2.0, "V/m"),
            omega0=Q(5.0e12, "rad/s"),
            label="test",
        )
        txt = d.report_plain(time_unit_s=1e-12, show_ascii_plot=False)
        self.assertIn("ClassicalFieldDrive report", txt)
        self.assertIn("Envelope", txt)
        self.assertIn("Amplitude", txt)

    def test_report_rich_method(self) -> None:
        d = gaussian_field_drive(
            t0=Q(0.0, "ps"),
            sigma=Q(5.0, "ps"),
            E0=Q(1.0, "V/m"),
            label="test",
        )
        txt = d.report_rich(time_unit_s=1e-12, show_ascii_plot=False)
        self.assertIn("ClassicalFieldDrive report", txt)
        self.assertIn("Envelope", txt)
        self.assertIn("Amplitude", txt)


if __name__ == "__main__":
    unittest.main()
