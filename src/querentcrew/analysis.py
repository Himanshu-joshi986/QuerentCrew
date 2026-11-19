"""Deterministic Pandas/NumPy analysis of query results.

The LLM never does arithmetic on its own: this module computes totals, shares
and period-over-period changes, and the analyst agent cites those numbers.
That keeps reported figures accurate and reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_TIME_NAME = re.compile(r"(month|quarter|date|period|year|week|day|^q$|^qtr)", re.I)
_TIME_VALUE = re.compile(r"^(\d{4}(-\d{2}){0,2}|\d{4}-?Q[1-4]|Q[1-4](\s?\d{4})?)$", re.I)


@dataclass
class Profile:
    rows: int
    columns: list[str]
    numeric_columns: list[str]
    category_columns: list[str]
    time_column: str | None
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a query result: numeric strings -> numbers, strip text, drop empty columns."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() == out[col].notna().sum() and out[col].notna().any():
                out[col] = converted
            else:
                out[col] = out[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    return out.dropna(axis=1, how="all")


def detect_time_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if _TIME_NAME.search(str(col)) and not pd.api.types.is_float_dtype(df[col]):
            return col
    for col in df.columns:
        values = df[col].dropna().astype(str).head(20)
        if len(values) and values.map(lambda v: bool(_TIME_VALUE.match(v))).all():
            return col
    return None


def _pct(a: float, b: float) -> float | None:
    if a is None or b is None or a == 0 or np.isnan(a) or np.isnan(b):
        return None
    return round((b - a) / abs(a) * 100, 2)


class DataProfiler:
    """Computes a compact, LLM-friendly statistical profile of a result table."""

    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    def profile(self, df: pd.DataFrame) -> Profile:
        df = clean_frame(df)
        time_col = detect_time_column(df)
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != time_col
                   and not str(c).lower().endswith("_id")]
        categories = [c for c in df.columns if c not in numeric and c != time_col
                      and not str(c).lower().endswith("_id")]
        prof = Profile(len(df), list(map(str, df.columns)), numeric, categories, time_col)

        if df.empty:
            prof.notes.append("query returned no rows")
            return prof

        # Summary statistics for numeric columns
        prof.stats["numeric_summary"] = {
            c: {
                "sum": round(float(df[c].sum()), 2),
                "mean": round(float(df[c].mean()), 2),
                "min": round(float(df[c].min()), 2),
                "max": round(float(df[c].max()), 2),
            }
            for c in numeric
        }
        metric = numeric[0] if numeric else None

        # Shares of total by the first category column
        if metric and categories and time_col is None and len(df) > 1:
            cat = categories[0]
            grouped = df.groupby(cat, dropna=False)[metric].sum().sort_values(ascending=False)
            total = grouped.sum()
            if total:
                prof.stats["share_of_total"] = {
                    "metric": metric,
                    "by": cat,
                    "shares_pct": {str(k): round(float(v / total * 100), 2) for k, v in grouped.head(self.top_n).items()},
                }

        # Period-over-period change for time series
        if metric and time_col:
            if categories:
                cat = categories[0]
                pivot = df.pivot_table(index=time_col, columns=cat, values=metric, aggfunc="sum").sort_index()
                if len(pivot) >= 2:
                    first, last = pivot.index[0], pivot.index[-1]
                    changes = {
                        str(c): _pct(float(pivot.loc[first, c]), float(pivot.loc[last, c])) for c in pivot.columns
                    }
                    changes = {k: v for k, v in changes.items() if v is not None}
                    prof.stats["change_first_to_last_period"] = {
                        "metric": metric, "by": cat, "from": str(first), "to": str(last),
                        "pct_change": dict(sorted(changes.items(), key=lambda kv: kv[1])[: self.top_n]),
                    }
            else:
                series = df.groupby(time_col)[metric].sum().sort_index()
                if len(series) >= 2:
                    pct = series.pct_change().mul(100).round(2)
                    prof.stats["period_over_period_pct"] = {
                        str(k): float(v) for k, v in pct.dropna().tail(self.top_n).items()
                    }
                    prof.stats["overall_change_pct"] = _pct(float(series.iloc[0]), float(series.iloc[-1]))
                    prof.stats["peak_period"] = str(series.idxmax())
                    prof.stats["lowest_period"] = str(series.idxmin())

        # Wide comparisons, e.g. columns q2_revenue and q3_revenue
        if len(numeric) == 2 and time_col is None:
            a, b = numeric
            per_row = {}
            label_col = categories[0] if categories else None
            for _, row in df.head(self.top_n * 2).iterrows():
                change = _pct(float(row[a]), float(row[b]))
                if change is not None:
                    label = str(row[label_col]) if label_col else str(len(per_row))
                    per_row[label] = change
            if per_row:
                prof.stats["change_between_columns"] = {
                    "from": a, "to": b,
                    "total_pct_change": _pct(float(df[a].sum()), float(df[b].sum())),
                    "pct_change_by_row": dict(sorted(per_row.items(), key=lambda kv: kv[1])),
                }

        if len(df) == 1 and numeric:
            prof.notes.append("single-row result")
        return prof

    @staticmethod
    def preview(df: pd.DataFrame, max_rows: int = 15) -> str:
        if df.empty:
            return "(no rows)"
        with pd.option_context("display.max_columns", 20, "display.width", 200):
            text = df.head(max_rows).to_string(index=False)
        if len(df) > max_rows:
            text += f"\n... ({len(df) - max_rows} more rows)"
        return text
