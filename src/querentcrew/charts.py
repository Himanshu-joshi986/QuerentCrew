"""Chart rendering with Matplotlib (PNG, base64-encoded so it can travel through the API)."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import clean_frame, detect_time_column  # noqa: E402
from .schemas import ChartSpec  # noqa: E402

PALETTE = ["#818cf8", "#22d3ee", "#f97316", "#10b981", "#c084fc", "#f43f5e", "#eab308", "#94a3b8"]

# Dark style matching the web UI, so the PNG fallback does not appear as a white box.
STYLE = {
    "figure.facecolor": "#0b1020", "axes.facecolor": "#0b1020", "savefig.facecolor": "#0b1020",
    "axes.edgecolor": "#2a3147", "axes.labelcolor": "#9aa3b8", "axes.titlecolor": "#e5e7f0",
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#1c2235", "grid.linewidth": 0.8,
    "xtick.color": "#9aa3b8", "ytick.color": "#9aa3b8", "xtick.major.size": 0, "ytick.major.size": 0,
    "text.color": "#e5e7f0", "legend.labelcolor": "#c9cfdd", "font.size": 9,
    "lines.linewidth": 2.2, "lines.markersize": 5,
}


class ChartBuilder:
    def auto_spec(self, df: pd.DataFrame, title: str = "") -> ChartSpec:
        """Heuristic chart choice, used when the LLM's spec is invalid."""
        df = clean_frame(df)
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not str(c).endswith("_id")]
        if df.empty or not numeric or len(df) < 2:
            return ChartSpec(chart_type="none", title=title)
        time_col = detect_time_column(df)
        if time_col and time_col != numeric[0]:
            others = [c for c in df.columns if c not in numeric and c != time_col]
            return ChartSpec(chart_type="line", x=time_col, y=numeric[0], series=others[0] if others else None,
                             title=title)
        labels = [c for c in df.columns if c not in numeric]
        if not labels:
            return ChartSpec(chart_type="none", title=title)
        return ChartSpec(chart_type="bar", x=labels[0], y=numeric[0], title=title)

    def validate(self, spec: ChartSpec, df: pd.DataFrame) -> ChartSpec:
        if spec.chart_type == "none":
            return spec
        cols = set(df.columns)
        if spec.x not in cols or spec.y not in cols or (spec.series and spec.series not in cols):
            return self.auto_spec(df, spec.title)
        if not pd.api.types.is_numeric_dtype(clean_frame(df)[spec.y]):
            return self.auto_spec(df, spec.title)
        if spec.chart_type == "pie" and df[spec.x].nunique() > 8:
            return spec.model_copy(update={"chart_type": "bar"})
        return spec

    def render(self, spec: ChartSpec, df: pd.DataFrame) -> str | None:
        """Return a base64 PNG, or None when no chart is appropriate."""
        df = clean_frame(df)
        spec = self.validate(spec, df)
        if spec.chart_type == "none" or df.empty:
            return None

        with plt.rc_context(STYLE):
            return self._draw(spec, df)

    def _draw(self, spec: ChartSpec, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
        ax.set_axisbelow(True)
        try:
            if spec.chart_type == "line":
                if spec.series:
                    pivot = df.pivot_table(index=spec.x, columns=spec.series, values=spec.y, aggfunc="sum").sort_index()
                    for i, col in enumerate(pivot.columns):
                        ax.plot(pivot.index.astype(str), pivot[col], marker="o", label=str(col),
                                color=PALETTE[i % len(PALETTE)])
                    ax.legend(frameon=False, fontsize=8)
                else:
                    data = df.sort_values(spec.x)
                    ax.plot(data[spec.x].astype(str), data[spec.y], marker="o", color=PALETTE[0])
                ax.set_ylabel(spec.y)
            elif spec.chart_type == "bar":
                if spec.series:
                    pivot = df.pivot_table(index=spec.x, columns=spec.series, values=spec.y, aggfunc="sum")
                    pivot.plot(kind="bar", ax=ax, color=PALETTE[: len(pivot.columns)], width=0.8)
                    ax.legend(frameon=False, fontsize=8)
                else:
                    data = df.groupby(spec.x)[spec.y].sum().sort_values(ascending=False).head(20)
                    ax.bar(data.index.astype(str), data.values, color=PALETTE[0])
                ax.set_ylabel(spec.y)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            elif spec.chart_type == "pie":
                data = df.groupby(spec.x)[spec.y].sum()
                ax.pie(data.values, labels=data.index.astype(str), autopct="%1.1f%%",
                       colors=PALETTE[: len(data)], startangle=90, pctdistance=0.8,
                       wedgeprops={"width": 0.42, "edgecolor": "#0b1020", "linewidth": 2})
                ax.axis("equal")
            if spec.chart_type != "pie":
                ax.grid(False, axis="x")  # pandas re-enables vertical grid lines
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:g}"))
            ax.set_title(spec.title or f"{spec.y} by {spec.x}", fontsize=11)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            return base64.b64encode(buffer.getvalue()).decode("ascii")
        finally:
            plt.close(fig)
