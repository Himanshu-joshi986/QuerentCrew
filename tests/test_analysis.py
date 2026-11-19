import base64

import pandas as pd

from querentcrew.analysis import DataProfiler, detect_time_column
from querentcrew.charts import ChartBuilder
from querentcrew.schemas import ChartSpec


def test_time_series_profile():
    df = pd.DataFrame({"month": ["2025-01", "2025-02", "2025-03"], "revenue": [100.0, 150.0, 120.0]})
    prof = DataProfiler().profile(df)
    assert prof.time_column == "month"
    assert prof.stats["period_over_period_pct"]["2025-02"] == 50.0
    assert prof.stats["overall_change_pct"] == 20.0
    assert prof.stats["peak_period"] == "2025-02"


def test_share_of_total():
    df = pd.DataFrame({"region": ["North", "South"], "revenue": [300, 100]})
    prof = DataProfiler().profile(df)
    assert prof.stats["share_of_total"]["shares_pct"] == {"North": 75.0, "South": 25.0}


def test_change_between_columns():
    df = pd.DataFrame({"category": ["A", "B"], "q2_revenue": [100, 200], "q3_revenue": [50, 220]})
    prof = DataProfiler().profile(df)
    stats = prof.stats["change_between_columns"]
    assert stats["pct_change_by_row"] == {"A": -50.0, "B": 10.0}
    assert stats["total_pct_change"] == -10.0


def test_change_by_category_over_time():
    df = pd.DataFrame({"quarter": ["Q2", "Q2", "Q3", "Q3"], "region": ["N", "S", "N", "S"],
                       "revenue": [100, 200, 110, 100]})
    stats = DataProfiler().profile(df).stats["change_first_to_last_period"]
    assert stats["pct_change"] == {"S": -50.0, "N": 10.0}


def test_numeric_strings_are_cleaned():
    df = pd.DataFrame({"region": ["N", "S"], "revenue": ["10.5", "20"]})
    assert DataProfiler().profile(df).numeric_columns == ["revenue"]


def test_detect_time_by_values():
    assert detect_time_column(pd.DataFrame({"p": ["2025-Q1", "2025-Q2"], "v": [1, 2]})) == "p"


def test_chart_render_and_fallback():
    df = pd.DataFrame({"region": ["N", "S", "E"], "revenue": [3, 1, 2]})
    builder = ChartBuilder()
    png = builder.render(ChartSpec(chart_type="bar", x="region", y="revenue"), df)
    assert base64.b64decode(png)[:4] == b"\x89PNG"
    # invalid column names from the LLM fall back to a heuristic chart
    fixed = builder.validate(ChartSpec(chart_type="bar", x="nope", y="nada"), df)
    assert fixed.x == "region" and fixed.y == "revenue"
    assert builder.render(ChartSpec(chart_type="none"), df) is None


def test_line_chart_with_series():
    df = pd.DataFrame({"month": ["2025-01", "2025-01", "2025-02", "2025-02"], "region": ["N", "S", "N", "S"],
                       "revenue": [1, 2, 3, 4]})
    spec = ChartBuilder().auto_spec(df)
    assert spec.chart_type == "line" and spec.series == "region"
    assert ChartBuilder().render(spec, df)
