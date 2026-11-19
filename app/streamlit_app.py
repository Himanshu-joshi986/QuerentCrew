"""QuerentCrew web UI. Run with: streamlit run app/streamlit_app.py"""

from __future__ import annotations

import base64
import html
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from querentcrew.analysis import clean_frame  # noqa: E402
from querentcrew.service import QuerentCrew  # noqa: E402

st.set_page_config(page_title="QuerentCrew", page_icon="📊", layout="wide", initial_sidebar_state="auto")

# Icons are Material Symbols names: rendered as `:material/<name>:` in Streamlit widgets
# and as <span class="ip-mi"> in hand-written HTML, so the whole UI uses one icon set.
EXAMPLES = [
    ("trending_down", "Root cause", "Why did revenue drop in the South region in Q3 2025?"),
    ("show_chart", "Trend", "Show monthly revenue for 2025 by region"),
    ("emoji_events", "Ranking", "Which 5 products earned the most revenue?"),
    ("groups", "Segments", "Compare average order value across customer segments"),
    ("storefront", "Channels", "What share of revenue comes from each sales channel?"),
    ("lock", "Governed data", "List the email addresses of our top 5 Enterprise customers"),
]

AGENT_INFO = {
    "router": ("alt_route", "Router"),
    "responder": ("chat", "Responder"),
    "planner": ("map", "Planner"),
    "sql_writer": ("edit_note", "SQL writer"),
    "sql_guard": ("shield", "SQL guard"),
    "human_approval": ("front_hand", "Human approval"),
    "sql_executor": ("bolt", "SQL executor"),
    "advance": ("arrow_forward", "Next step"),
    "analyst": ("science", "Analyst"),
    "charter": ("bar_chart", "Chart builder"),
    "reporter": ("description", "Reporter"),
}

PALETTE = ["#818cf8", "#22d3ee", "#f97316", "#10b981", "#c084fc", "#f43f5e", "#eab308", "#94a3b8"]
AVATAR_USER = ":material/person:"
AVATAR_BOT = ":material/insights:"


def mi(name: str) -> str:
    """Inline Material Symbol for raw HTML."""
    return f'<span class="ip-mi" aria-hidden="true">{name}</span>'


def icon(name: str) -> str:
    """Material Symbol shortcode for Streamlit widgets and markdown."""
    return f":material/{name}:"

STATUS_BADGE = {
    "ok": ("ok", "Success"),
    "failed": ("err", "Failed"),
    "blocked": ("warn", "Blocked"),
    "rejected": ("warn", "Rejected"),
    "llm_error": ("err", "Model unavailable"),
}

CSS_PATH = Path(__file__).with_name("style.css")


def load_css() -> str:
    css = "\n".join(line for line in CSS_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
    return f"<style>{css}</style>"  # no blank lines, or markdown would end the HTML block early



# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Starting agents...")
def get_pilot() -> QuerentCrew:
    ensure_sample_database()
    return QuerentCrew()


def ensure_sample_database() -> None:
    """Build the bundled sample database on first start (a fresh clone or a cloud deploy has none)."""
    from querentcrew.config import get_settings

    path = Path(get_settings().database_path)
    if path.exists():
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from seed_database import seed

    path.parent.mkdir(parents=True, exist_ok=True)
    seed(path)


@st.cache_data(ttl=300, show_spinner=False)
def llm_status(_pilot: QuerentCrew, model: str) -> tuple[bool, str]:
    return _pilot.check_llm()


def stretch(element, *args, **kwargs):
    """Full-width element that works on both newer (width="stretch") and older Streamlit."""
    try:
        return element(*args, width="stretch", **kwargs)
    except TypeError:
        return element(*args, use_container_width=True, **kwargs)


def init_state() -> None:
    st.session_state.setdefault("thread_id", QuerentCrew.new_thread_id())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("totals", {"questions": 0, "calls": 0, "tokens": 0, "seconds": 0.0})


def reset_conversation() -> None:
    st.session_state.thread_id = QuerentCrew.new_thread_id()
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.totals = {"questions": 0, "calls": 0, "tokens": 0, "seconds": 0.0}


def user_bubble(text: str) -> None:
    st.markdown(f'<span class="ip-user"></span><div class="ip-user-text">{esc(text)}</div>',
                unsafe_allow_html=True)


def no_math(text: str) -> str:
    """Escape $ so money amounts are not rendered as LaTeX by st.markdown."""
    return text.replace(r"\$", "$").replace("$", r"\$")


def esc(text) -> str:
    return html.escape(str(text))


def fmt_int(n) -> str:
    n = float(n or 0)
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n:.0f}"


# ---------------------------------------------------------------- charts
def interactive_chart(chart: dict, results: list[dict]):
    """Build an interactive Altair chart from the charter agent's spec (None if it can't)."""
    spec = chart.get("spec") or {}
    idx = chart.get("result_index", 0)
    if idx >= len(results) or results[idx].get("status") != "ok":
        return None
    rec = results[idx]
    df = clean_frame(pd.DataFrame(rec["rows"], columns=rec["columns"]))
    kind, x, y, series = spec.get("chart_type"), spec.get("x"), spec.get("y"), spec.get("series")
    if kind not in ("bar", "line", "pie") or x not in df.columns or y not in df.columns:
        return None
    if series and series not in df.columns:
        series = None
    df[x] = df[x].astype(str)
    title = spec.get("title") or f"{y} by {x}".replace("_", " ").capitalize()
    y_title = str(y).replace("_", " ").capitalize()
    y_tip = alt.Tooltip(field=y, type="quantitative", format=",.2f", title=y_title)
    x_tip = alt.Tooltip(field=x, type="nominal")
    palette = alt.Scale(range=PALETTE)
    legend = alt.Legend(title=None)
    angle = -30 if df[x].nunique() > 8 or df[x].str.len().max() > 10 else 0
    x_axis = alt.Axis(labelAngle=angle, labelOverlap=True, labelPadding=8, ticks=False)
    y_axis = alt.Axis(format="~s", tickCount=5, domain=False, ticks=False, labelPadding=8)

    if kind == "pie":
        data = df.groupby(x, as_index=False)[y].sum()
        total = float(data[y].sum())
        hover = alt.selection_point(on="pointerover", fields=[x], empty=True)
        arcs = alt.Chart(data).mark_arc(innerRadius=88, outerRadius=150, stroke="#0b1020", strokeWidth=3,
                                        cornerRadius=6, padAngle=0.01).encode(
            theta=alt.Theta(field=y, type="quantitative", stack=True),
            color=alt.Color(field=x, type="nominal", scale=palette, legend=legend),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.4)),
            tooltip=[x_tip, y_tip],
        ).add_params(hover)
        center = alt.Chart(pd.DataFrame({"v": [fmt_compact(total)], "l": ["TOTAL"]}))
        out = alt.layer(
            arcs,
            center.mark_text(fontSize=26, fontWeight=800, color="#f1f3fa", dy=-6).encode(text="v:N"),
            center.mark_text(fontSize=10, fontWeight=600, color="#9aa3b8", dy=18)
            .encode(text="l:N"),
        )
        return _theme(out.properties(title=title, height=360))

    if kind == "line":
        data = df.sort_values(x)
        x_enc = alt.X(field=x, type="ordinal", sort=None, title=None, axis=x_axis)
        y_enc = alt.Y(field=y, type="quantitative", title=y_title, axis=y_axis)
        nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False, clear="pointerout")
        base = alt.Chart(data).encode(x=x_enc, y=y_enc)
        tips = [x_tip, y_tip]
        if series:
            color = alt.Color(field=series, type="nominal", scale=palette, legend=legend)
            base = base.encode(color=color)
            tips.insert(0, alt.Tooltip(field=series, type="nominal"))
            layers = [base.mark_line(strokeWidth=2.5, interpolate="monotone")]
        else:
            grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                                stops=[alt.GradientStop(color="rgba(129,140,248,0)", offset=0),
                                       alt.GradientStop(color="rgba(129,140,248,0.35)", offset=1)])
            layers = [base.mark_area(interpolate="monotone", color=grad),
                      base.mark_line(strokeWidth=2.5, interpolate="monotone", color=PALETTE[0])]
        point_kw = {} if series else {"color": PALETTE[0]}
        layers += [
            alt.Chart(data).mark_rule(color="rgba(255,255,255,0.22)", strokeDash=[4, 4])
            .encode(x=x_enc, opacity=alt.condition(nearest, alt.value(1), alt.value(0))),
            base.mark_point(filled=True, stroke="#0b1020", strokeWidth=2, **point_kw).encode(
                size=alt.condition(nearest, alt.value(110), alt.value(36)), tooltip=tips,
            ).add_params(nearest),
        ]
        return _theme(alt.layer(*layers).properties(title=title, height=360))

    hover = alt.selection_point(on="pointerover", clear="pointerout", empty=True)
    bar_kw = dict(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    if series:
        data = df.groupby([x, series], as_index=False)[y].sum()
        out = alt.Chart(data).mark_bar(**bar_kw).encode(
            x=alt.X(field=x, type="nominal", title=None, axis=x_axis),
            xOffset=alt.XOffset(field=series, type="nominal"),
            y=alt.Y(field=y, type="quantitative", title=y_title, axis=y_axis),
            color=alt.Color(field=series, type="nominal", scale=palette, legend=legend),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.45)),
            tooltip=[x_tip, alt.Tooltip(field=series, type="nominal"), y_tip],
        )
    else:
        data = df.groupby(x, as_index=False)[y].sum().sort_values(y, ascending=False).head(20)
        grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                            stops=[alt.GradientStop(color="#6366f1", offset=0),
                                   alt.GradientStop(color="#22d3ee", offset=1)])
        out = alt.Chart(data).mark_bar(color=grad, **bar_kw).encode(
            x=alt.X(field=x, type="nominal", sort="-y", title=None, axis=x_axis),
            y=alt.Y(field=y, type="quantitative", title=y_title, axis=y_axis),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.45)),
            tooltip=[x_tip, y_tip],
        )
    return _theme(out.add_params(hover).properties(title=title, height=360))


def _theme(chart):
    return (
        chart.configure(background="transparent", font="Inter", padding={"left": 4, "right": 12, "top": 4, "bottom": 4})
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="rgba(255,255,255,0.06)", domainColor="rgba(255,255,255,0.12)",
                        tickColor="rgba(255,255,255,0.12)", labelColor="#9aa3b8", titleColor="#9aa3b8",
                        labelFontSize=11, titleFontSize=11, titleFontWeight=500)
        .configure_axisX(grid=False)
        .configure_legend(labelColor="#c9cfdd", titleColor="#9aa3b8", orient="bottom", symbolType="circle",
                          labelFontSize=12, columnPadding=16)
        .configure_title(color="#e5e7f0", fontSize=15, fontWeight=700, anchor="start", offset=16)
    )


def fmt_compact(n: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            return f"{n / div:.1f}{suffix}"
    return f"{n:,.0f}"


def show_frame(df: pd.DataFrame) -> None:
    """Dataframe with thousands separators on numeric columns."""
    config = {c: st.column_config.NumberColumn(format="localized") for c in df.columns
              if pd.api.types.is_numeric_dtype(df[c]) and not str(c).endswith("_id")}
    try:
        stretch(st.dataframe, df, hide_index=True, column_config=config)
    except Exception:
        stretch(st.dataframe, df, hide_index=True)


# ---------------------------------------------------------------- rendering
def stats_strip(resp: dict) -> None:
    results = resp.get("results") or []
    usage = resp.get("usage") or {}
    ok = sum(1 for r in results if r.get("status") == "ok")
    rows = sum(r.get("row_count", 0) for r in results if r.get("status") == "ok")
    items = []
    if results:
        items.append(f"{mi('task_alt')} <b>{ok}/{len(results)}</b> queries ok")
        items.append(f"{mi('table_rows')} <b>{fmt_int(rows)}</b> rows")
    items.append(f"{mi('bolt')} <b>{usage.get('calls', 0)}</b> LLM calls")
    items.append(f"{mi('data_object')} <b>{fmt_int(usage.get('total_tokens', 0))}</b> tokens")
    items.append(f"{mi('timer')} <b>{resp.get('elapsed_ms', 0) / 1000:.1f}s</b>")
    st.markdown('<div class="ip-stats">' + "".join(f'<span class="ip-stat">{i}</span>' for i in items) + "</div>",
                unsafe_allow_html=True)


def render_trace(trace: list[dict], key: str) -> None:
    steps = []
    for step in trace:
        ic, name = AGENT_INFO.get(step["agent"], ("circle", step["agent"]))
        cls = "err" if any(w in step["message"] for w in ("unavailable", "failed", "blocked", "exhausted")) else ""
        steps.append(f'<div class="ip-trace-step {cls}"><span class="ip-trace-ic">{mi(ic)}</span>'
                     f'<b>{esc(name)}</b><span class="msg">{esc(step["message"])}</span></div>')
    st.markdown('<div class="ip-trace">' + "".join(steps) + "</div>", unsafe_allow_html=True)
    if st.toggle("Show raw step details", key=f"raw_{key}"):
        st.json(trace, expanded=False)


def render_response(resp: dict, key: str, question: str | None = None) -> None:
    if resp.get("llm_error"):
        st.error(resp["llm_error"], icon=icon("power_off"))
        cols = st.columns([1, 1.6, 3.4])
        if question and cols[0].button("Retry", icon=icon("refresh"), key=f"retry_{key}", type="primary"):
            llm_status.clear()
            st.session_state.queued = question
            st.rerun()
        if cols[1].button("Check connection", icon=icon("wifi_find"), key=f"check_{key}"):
            llm_status.clear()
            st.rerun()
        if resp.get("results") and any(r["status"] == "ok" for r in resp["results"]):
            st.markdown(no_math(resp.get("answer") or ""))
    elif resp.get("answer"):
        st.markdown(no_math(resp["answer"]))

    results = resp.get("results") or []
    chart = resp.get("chart")
    if resp.get("intent") not in (None, "data_question") and not results:
        return
    stats_strip(resp)

    names = []
    if chart:
        names.append("Chart")
    if results:
        names += ["Data", "SQL"]
    if resp.get("plan"):
        names.append("Plan")
    if resp.get("trace"):
        names.append("Agent trace")
    if not names:
        return
    tab_icons = {"Chart": "bar_chart", "Data": "table_chart", "SQL": "code", "Plan": "account_tree",
                 "Agent trace": "timeline"}
    tabs = dict(zip(names, st.tabs([f"{icon(tab_icons[n])} {n}" for n in names])))

    if "Chart" in tabs:
        with tabs["Chart"]:
            drawn = False
            try:
                c = interactive_chart(chart, results)
                if c is not None:
                    stretch(st.altair_chart, c, theme=None)
                    drawn = True
            except Exception:
                drawn = False
            if not drawn and chart.get("image_b64"):
                stretch(st.image, base64.b64decode(chart["image_b64"]))
            elif not drawn:
                st.info("This result can't be drawn as a chart. See the Data tab.", icon=icon("info"))
            if drawn:
                st.caption("Hover the chart for exact values")

    if "Data" in tabs:
        with tabs["Data"]:
            for i, rec in enumerate(results):
                badge_cls, label = STATUS_BADGE.get(rec["status"], ("info", rec["status"]))
                st.markdown(f'<div class="ip-step-head"><span class="ip-step-n">{i + 1}</span>'
                            f'<span class="ip-step-q">{esc(rec["sub_question"])}</span>'
                            f'<span class="ip-badge {badge_cls}">{label}</span></div>', unsafe_allow_html=True)
                if rec["status"] == "ok":
                    df = pd.DataFrame(rec["rows"], columns=rec["columns"])
                    show_frame(df)
                    meta = f"{rec['row_count']:,} rows{' (truncated)' if rec.get('truncated') else ''}"
                    if rec.get("masked_columns"):
                        meta += f" · masked: {', '.join(rec['masked_columns'])}"
                    c1, c2 = st.columns([4, 1], vertical_alignment="center")
                    c1.caption(meta)
                    c2.download_button("CSV", df.to_csv(index=False).encode("utf-8"), icon=icon("download"),
                                       file_name=f"querentcrew_step{i + 1}.csv", mime="text/csv",
                                       key=f"dl_{key}_{i}", width="stretch")
                else:
                    st.warning(rec.get("error") or rec["status"], icon=icon("warning"))

    if "SQL" in tabs:
        with tabs["SQL"]:
            for i, rec in enumerate(results):
                badge_cls, label = STATUS_BADGE.get(rec["status"], ("info", rec["status"]))
                timing = f" · {rec.get('elapsed_ms', 0)} ms" if rec["status"] == "ok" else ""
                st.markdown(f'<div class="ip-step-head"><span class="ip-step-n">{i + 1}</span>'
                            f'<span class="ip-step-q">{esc(rec["sub_question"])}</span>'
                            f'<span class="ip-badge {badge_cls}">{label}</span>'
                            f'<span class="ip-step-meta">{rec.get("attempts", 0)} attempt(s){timing}</span></div>',
                            unsafe_allow_html=True)
                if rec.get("sql"):
                    st.code(rec["sql"].strip(), language="sql")
                else:
                    st.caption("No SQL was produced for this step.")

    if "Plan" in tabs:
        with tabs["Plan"]:
            st.markdown('<div class="ip-plan">' + "".join(
                f'<div class="ip-plan-step"><span class="ip-step-n">{i}</span><span>{esc(sub)}</span></div>'
                for i, sub in enumerate(resp["plan"], 1)) + "</div>", unsafe_allow_html=True)
            insights = resp.get("insights") or {}
            if insights.get("caveats"):
                st.markdown('<div class="ip-section-title">Caveats</div>', unsafe_allow_html=True)
                for c in insights["caveats"]:
                    st.markdown(f"- {no_math(c)}")

    if "Agent trace" in tabs:
        with tabs["Agent trace"]:
            render_trace(resp["trace"], key)


# ---------------------------------------------------------------- running
def run_with_progress(fn, label: str):
    """Run an ask/resume call while streaming each agent step into a status box."""
    with st.status(label, expanded=True) as box:
        def on_step(node: str, entry: dict) -> None:
            if node == "advance":
                return
            ic, name = AGENT_INFO.get(node, ("circle", node))
            box.markdown(f"{icon(ic)} **{name}** · {no_math(entry.get('message', ''))}")
            box.update(label=f"{name}…")

        try:
            resp = fn(on_step)
        except Exception as exc:
            box.update(label="Something went wrong", state="error", expanded=True)
            raise exc
        if resp.status == "awaiting_approval":
            box.update(label="Waiting for your approval", state="running", expanded=False)
        elif resp.llm_error:
            box.update(label="Language model unavailable", state="error", expanded=False)
        else:
            box.update(label=f"Done in {resp.elapsed_ms / 1000:.1f}s", state="complete", expanded=False)
    return resp


def handle(resp, question: str) -> None:
    data = resp.to_dict()
    usage = data.get("usage") or {}
    totals = st.session_state.totals
    totals["calls"] += usage.get("calls", 0)
    totals["tokens"] += usage.get("total_tokens", 0)
    totals["seconds"] += data.get("elapsed_ms", 0) / 1000
    if resp.status == "awaiting_approval":
        st.session_state.pending = {**data, "question": question}
    else:
        st.session_state.pending = None
        st.session_state.messages.append({"role": "assistant", "content": data.get("answer", ""),
                                          "response": data, "question": question})


def export_markdown() -> str:
    lines = ["# QuerentCrew conversation", f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lines += [f"## ❓ {msg['content']}", ""]
            continue
        lines += [msg.get("content") or "", ""]
        for i, rec in enumerate((msg.get("response") or {}).get("results") or [], 1):
            if rec.get("sql"):
                lines += [f"<details><summary>SQL step {i}: {rec['sub_question']}</summary>", "",
                          "```sql", rec["sql"].strip(), "```", "", "</details>", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- dialogs
@st.dialog("Table preview", width="large")
def preview_dialog(pilot: QuerentCrew, table: str) -> None:
    info = pilot.schema()
    st.markdown(f"### `{table}`")
    st.caption(f"{info['row_counts'].get(table, 0):,} rows · first 50 shown")
    cols, rows = pilot.preview(table, 50)
    show_frame(pd.DataFrame(rows, columns=cols))
    schema = pd.DataFrame(info["tables"][table], columns=["column", "type"])
    st.markdown("**Columns**")
    stretch(st.dataframe, schema, hide_index=True)


# ---------------------------------------------------------------- layout
def sidebar(pilot: QuerentCrew, info: dict, status: tuple[bool, str]) -> None:
    with st.sidebar:
        st.markdown(f'<div class="ip-brand"><div class="ip-logo">{mi("insights")}</div><div>'
                    '<div class="ip-brand-name">QuerentCrew</div>'
                    '<div class="ip-brand-sub">You ask · the crew answers</div></div></div>',
                    unsafe_allow_html=True)

        ok, message = status
        dot = "ok" if ok else "err"
        text = "Model connected" if ok else "Model unreachable"
        model = esc(pilot.settings.llm_model)
        st.markdown(f'<div class="ip-status"><span class="ip-dot {dot}"></span><div class="ip-status-body">'
                    f'<b>{text}</b><code title="{model}">{model}</code></div></div>', unsafe_allow_html=True)

        if not ok:
            st.caption(message)
        if st.button("Re-check connection", icon=icon("refresh"), key="recheck", width="stretch"):
            llm_status.clear()
            st.rerun()

        st.markdown('<div class="ip-section-title">Conversation</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("New", icon=icon("add_comment"), width="stretch", help="Start a new conversation"):
            reset_conversation()
            st.rerun()
        c2.download_button("Export", export_markdown().encode("utf-8"), file_name="querentcrew_chat.md",
                           mime="text/markdown", width="stretch", icon=icon("download"),
                           disabled=not st.session_state.messages, help="Download this chat as Markdown")
        t = st.session_state.totals
        tiles = [("Questions", t["questions"]), ("Calls", t["calls"]), ("Tokens", fmt_int(t["tokens"]))]
        st.markdown('<div class="ip-tiles">' + "".join(
            f'<div class="ip-tile"><div class="ip-tile-v">{v}</div><div class="ip-tile-l">{lbl}</div></div>'
            for lbl, v in tiles) + "</div>", unsafe_allow_html=True)

        st.markdown('<div class="ip-section-title">Data explorer</div>', unsafe_allow_html=True)
        for table, cols in info["tables"].items():
            if st.button(f"{table} · {info['row_counts'][table]:,}", icon=icon("table"), key=f"tbl_{table}",
                         width="stretch", help=", ".join(c[0] for c in cols)):
                preview_dialog(pilot, table)

        with st.expander("Import a CSV", icon=icon("upload_file")):
            upload = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
            if upload is not None and st.button("Import", type="primary", width="stretch"):
                try:
                    table, rows = pilot.import_csv(upload, upload.name)
                    st.toast(f"Imported {rows:,} rows into `{table}`", icon=icon("check_circle"))
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

        with st.expander("Governance", icon=icon("shield")):
            s = pilot.settings
            st.markdown(
                f"- PII columns: `{', '.join(s.pii_columns) or 'none'}`\n"
                f"- Approval for PII queries: **{'on' if s.require_approval_for_pii else 'off'}**\n"
                f"- Approval for every query: **{'on' if s.require_approval_for_all else 'off'}**\n"
                f"- Max rows per query: **{s.max_result_rows}**\n"
                f"- SQL self-correction retries: **{s.max_sql_retries}**\n"
                "- Read-only database connection, single SELECT only"
            )


def hero(pilot: QuerentCrew, info: dict, compact: bool = False) -> None:
    total_rows = sum(info["row_counts"].values())
    chips = [f"{mi('psychology')} {esc(pilot.settings.llm_model)}", f"{mi('database')} {len(info['tables'])} tables",
             f"{mi('table_rows')} {total_rows:,} rows", f"{mi('shield')} Guarded SQL",
             f"{mi('autorenew')} Self-correcting", f"{mi('how_to_reg')} Human-in-the-loop"]
    if compact:
        chips = chips[:3]
    eyebrow = ("" if compact else '<div class="ip-eyebrow"><span class="ip-dot ok"></span>'
               "Your AI analyst crew</div>")
    title = "QuerentCrew" if compact else "Ask your business data anything"
    st.markdown(
        f'<div class="ip-hero{" compact" if compact else ""}">{eyebrow}<h1>{title}</h1>'
        "<p>A team of AI agents plans the analysis, writes safe SQL, fixes its own mistakes, "
        "crunches the numbers and reports back with interactive charts.</p>"
        '<div class="ip-chips">' + "".join(f'<span class="ip-chip">{c}</span>' for c in chips) + "</div></div>",
        unsafe_allow_html=True,
    )


def example_grid() -> None:
    st.markdown('<div class="ip-section-title">Start with a question</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for i, (ic, tag, q) in enumerate(EXAMPLES):
        with cols[i % 3], st.container(key=f"excard{i}"):
            st.markdown(f'<div class="ip-card-head"><span class="ip-card-icon">{mi(ic)}</span>'
                        f'<span class="ip-card-tag">{tag}</span><span class="ip-card-go">{mi("arrow_outward")}</span>'
                        '</div>', unsafe_allow_html=True)
            if st.button(q, key=f"ex_{i}", width="stretch"):
                st.session_state.queued = q
                st.rerun()


def scroll_to_top() -> None:
    """Chat apps auto-scroll to the bottom; on the welcome screen that hides the top of the hero."""
    with st.container(key="ip_scrolltop"):
        components.html(
            "<script>const go=()=>{const c=window.parent.document.querySelector("
            "'[data-testid=\"stAppScrollToBottomContainer\"]');if(c)c.scrollTop=0;};"
            "[0,150,400,900].forEach(t=>setTimeout(go,t));</script>",
            height=0,
        )


def approval_card(pilot: QuerentCrew, pending: dict) -> None:
    req = pending["approval_request"] or {}
    with st.chat_message("assistant", avatar=icon("shield")):
        st.markdown('<span class="ip-badge warn">Human approval required</span>', unsafe_allow_html=True)
        st.markdown(f"**Step:** {esc(req.get('sub_question', ''))}")
        for reason in req.get("reasons", []):
            st.markdown(f"- {reason}")
        if req.get("pii_columns"):
            st.info(f"This query reads personal data: **{', '.join(req['pii_columns'])}**. "
                    "Values will be masked in the results.", icon=icon("lock"))
        st.code(req.get("sql", ""), language="sql")
        comment = st.text_input("Comment (optional)", key="approval_comment", placeholder="Why approve / reject?")
        c1, c2, _ = st.columns([1, 1, 3])
        decision = None
        if c1.button("Approve and run", icon=icon("check"), type="primary", width="stretch"):
            decision = True
        if c2.button("Reject", icon=icon("block"), width="stretch"):
            decision = False
        if decision is not None:
            resp = run_with_progress(
                lambda cb: pilot.resume(pending["thread_id"], approved=decision,
                                        comment=comment or ("approved in UI" if decision else "rejected in UI"),
                                        on_step=cb),
                "Continuing the analysis…",
            )
            handle(resp, pending.get("question", ""))
            st.rerun()


def main() -> None:
    st.markdown(load_css(), unsafe_allow_html=True)
    init_state()
    try:
        pilot = get_pilot()
        info = pilot.schema()
    except Exception as exc:
        st.error(f"Could not start QuerentCrew: {exc}")
        st.info("Check your .env (LLM_MODEL and the matching API key) and run `python scripts/seed_database.py`.")
        st.stop()

    status = llm_status(pilot, pilot.settings.llm_model)
    sidebar(pilot, info, status)

    # Read the input first (it is pinned to the bottom anyway) so a new question
    # immediately switches the page into conversation layout.
    pending = st.session_state.pending
    question = st.chat_input("Ask a question about your sales data…", disabled=bool(pending))
    question = (question or st.session_state.pop("queued", None) or "").strip()
    chatting = bool(st.session_state.messages or pending or question)

    hero(pilot, info, compact=chatting)

    if not status[0]:
        st.warning(f"**Language model unreachable.** {status[1]}", icon=icon("power_off"))

    if not chatting:
        example_grid()
        scroll_to_top()

    for n, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar=AVATAR_USER):
                user_bubble(msg["content"])
        else:
            with st.chat_message("assistant", avatar=AVATAR_BOT):
                if msg.get("response"):
                    render_response(msg["response"], key=str(n), question=msg.get("question"))
                else:
                    st.markdown(msg["content"])

    if pending:
        approval_card(pilot, pending)

    if not chatting:
        st.markdown('<div class="ip-footer">QuerentCrew · read-only SQL · personal data masked · '
                    'every step traced</div>', unsafe_allow_html=True)
    if question and not pending:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.totals["questions"] += 1
        with st.chat_message("user", avatar=AVATAR_USER):
            user_bubble(question)
        with st.chat_message("assistant", avatar=AVATAR_BOT):
            try:
                resp = run_with_progress(
                    lambda cb: pilot.ask(question, st.session_state.thread_id, on_step=cb),
                    "Thinking…",
                )
                handle(resp, question)
                if resp.llm_error:
                    llm_status.clear()
            except Exception as exc:
                st.session_state.messages.append({"role": "assistant", "content": f"Unexpected error: {exc}",
                                                  "question": question})
        st.rerun()


main()
