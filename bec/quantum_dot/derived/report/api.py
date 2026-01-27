from __future__ import annotations

from .rich import (
    ReportStyle,
    has_rich,
    render_to_text,
    print_report as _print_rich,
)
from .plain import report_plain_impl


class ReportMixin:
    def _report_plain_impl(self) -> str:
        return report_plain_impl(self)

    def report_plain(self) -> str:
        return self._report_plain_impl()

    def render_report(self, *, style: dict | None = None) -> str:
        st = style or {}
        rs = ReportStyle(
            use_color=bool(st.get("use_color", True)),
            use_unicode=bool(st.get("use_unicode", True)),
            width=st.get("width", None),  # None => auto terminal width
            columns=bool(st.get("columns", True)),
            columns_min_width=int(st.get("columns_min_width", 140)),
        )
        if has_rich():
            return render_to_text(self, style=rs)
        return self.report_plain()

    def print_report(self, *, style: dict | None = None) -> None:
        st = style or {}
        rs = ReportStyle(
            use_color=bool(st.get("use_color", True)),
            use_unicode=bool(st.get("use_unicode", True)),
            width=st.get("width", None),
            columns=bool(st.get("columns", True)),
            columns_min_width=int(st.get("columns_min_width", 140)),
        )
        if has_rich():
            _print_rich(self, style=rs)
        else:
            print(self.report_plain())

    def report(self, *, rich: bool = True, style: dict | None = None) -> str:
        if not rich:
            return self.report_plain()

        st = style or {}
        rs = ReportStyle(
            use_color=bool(st.get("use_color", True)),
            use_unicode=bool(st.get("use_unicode", True)),
            width=st.get("width", None),  # auto when None
            columns=bool(st.get("columns", True)),
            columns_min_width=int(st.get("columns_min_width", 140)),
        )
        if has_rich():
            return render_to_text(self, style=rs)
        return self.report_plain()
