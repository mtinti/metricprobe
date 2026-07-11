"""Plotly figure builders — the SINGLE source of figures for the static
report, the markdown dashboard, and (later) the app.

Every builder consumes a probe's PRESENTATION frames (suppression already
applied by viz.presentation — a blanked count arrives as NA and renders as a
gap, never a value) and returns a plotly Figure, or None when the probe has
no data for that figure. figures_for_probe() assembles the applicable set
under stable keys.

Proxy timestamps (config `proxy: true`) are labelled honestly in the title of
every figure; `expect_batchy: true` adjusts wording, not math.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

FIGURE_ORDER = (
    "volume",
    "completion_curves",
    "completion_heatmap",
    "percentiles",
    "dual_overlay",
    "dual_delta",
    "batch",
    "parity",
)

_STATE_COLORS = {"mature": "#2f6fb2", "immature": "#8ebbe3", "open": "#d3d3d3"}
_PERCENTILES = (50, 90, 95, 99)


def _title(text: str, probe: str, proxy: bool) -> str:
    suffix = " — PROXY timestamps" if proxy else ""
    return f"{text} — {probe}{suffix}"


def volume_figure(month_volumes: pd.DataFrame, probe: str, proxy: bool) -> go.Figure:
    """Rows per event month, colored by maturity state; deficit months are
    flagged; interior gaps show as missing bars (explicitly RED in statuses)."""
    df = month_volumes.sort_values("month")
    fig = go.Figure()
    for state, color in _STATE_COLORS.items():
        subset = df[df["state"] == state]
        if subset.empty:
            continue
        fig.add_bar(
            x=list(subset["month"]),
            y=[None if pd.isna(v) else float(v) for v in subset["volume"]],
            name=state,
            marker_color=color,
        )
    deficits = df[df["deficit"].fillna(False).astype(bool)]
    if not deficits.empty:
        fig.add_scatter(
            x=list(deficits["month"]),
            y=[None if pd.isna(v) else float(v) for v in deficits["volume"]],
            mode="markers",
            name="arrival deficit",
            marker={"symbol": "triangle-down", "size": 12, "color": "#c0392b"},
        )
    band = df[df["expected_low"].notna()]
    if not band.empty:
        fig.add_scatter(
            x=list(band["month"]),
            y=[float(v) for v in band["expected_low"]],
            mode="lines",
            name="expected fill (low)",
            line={"dash": "dot", "color": "#7f8c8d"},
        )
    fig.update_layout(
        title=_title("Rows per event month", probe, proxy),
        xaxis_title="event month",
        yaxis_title="rows",
        barmode="overlay",
    )
    return fig


def _curves(cells: pd.DataFrame) -> dict[str, pd.Series]:
    """month -> cumulative fraction indexed by lag_day (suppressed cells are
    dropped: their mass stays hidden)."""
    usable = cells[(cells["lag_day"] >= 0) & cells["row_count"].notna()]
    curves: dict[str, pd.Series] = {}
    for month, group in usable.groupby(usable["event_month"].astype(str)):
        counts = group.set_index(group["lag_day"].astype(int))["row_count"].sort_index()
        total = counts.sum()
        if total > 0:
            curves[str(month)[:7]] = counts.cumsum() / total
    return curves


def completion_curves_figure(
    month_lag_cells: pd.DataFrame,
    month_volumes: pd.DataFrame | None,
    probe: str,
    proxy: bool,
) -> go.Figure | None:
    """Per-month cumulative completion curves; mature months drawn solid with
    a pointwise median + p10-p90 band, immature/open months faint."""
    curves = _curves(month_lag_cells)
    if not curves:
        return None
    states = {}
    if month_volumes is not None:
        states = {
            str(m)[:7]: s
            for m, s in zip(
                month_volumes["month"], month_volumes["state"], strict=True
            )
        }
    fig = go.Figure()
    mature_frames = []
    for month, curve in sorted(curves.items()):
        state = states.get(month, "mature")
        faint = state != "mature"
        fig.add_scatter(
            x=list(curve.index),
            y=[round(float(v) * 100, 3) for v in curve.values],
            mode="lines",
            name=f"{month} ({state})" if faint else month,
            line={"width": 1, "color": "#c9d7e4"} if faint else {"width": 1.5},
            opacity=0.6 if faint else 1.0,
        )
        if not faint:
            mature_frames.append(curve)
    if mature_frames:
        grid = sorted({day for curve in mature_frames for day in curve.index})
        aligned = pd.DataFrame(
            {i: curve.reindex(grid).ffill().fillna(0.0) for i, curve in enumerate(mature_frames)}
        )
        quantile_lines = ((0.5, "median", "solid"), (0.1, "p10", "dot"), (0.9, "p90", "dot"))
        for quantile, name, dash in quantile_lines:
            fig.add_scatter(
                x=grid,
                y=[round(float(v) * 100, 3) for v in aligned.quantile(quantile, axis=1)],
                mode="lines",
                name=name,
                line={"width": 3 if name == "median" else 1, "color": "#1a1a2e", "dash": dash},
            )
    fig.update_layout(
        title=_title("Completion curves (cumulative % of final rows)", probe, proxy),
        xaxis_title="lag (days)",
        yaxis_title="% of final rows",
        yaxis_range=[0, 105],
    )
    return fig


def completion_heatmap_figure(
    month_lag_cells: pd.DataFrame, probe: str, proxy: bool
) -> go.Figure | None:
    """Event-month x lag-WEEK fill fractions."""
    usable = month_lag_cells[
        (month_lag_cells["lag_day"] >= 0) & month_lag_cells["row_count"].notna()
    ].copy()
    if usable.empty:
        return None
    usable["month"] = usable["event_month"].astype(str).str[:7]
    usable["lag_week"] = (usable["lag_day"].astype(int) // 7).astype(int)
    grid = usable.pivot_table(
        index="month", columns="lag_week", values="row_count", aggfunc="sum"
    ).fillna(0)
    fractions = grid.div(grid.sum(axis=1), axis=0)
    fig = go.Figure(
        go.Heatmap(
            z=[[round(float(v), 4) for v in row] for row in fractions.values],
            x=[f"wk {c}" for c in fractions.columns],
            y=list(fractions.index),
            colorscale="Blues",
            colorbar={"title": "fraction"},
        )
    )
    fig.update_layout(
        title=_title("Arrival mass by event month x lag week", probe, proxy),
        xaxis_title="lag (weeks)",
        yaxis_title="event month",
    )
    return fig


def percentile_summary_figure(
    completion_percentiles: pd.DataFrame,
    completion_summary: pd.DataFrame | None,
    probe: str,
    proxy: bool,
) -> go.Figure | None:
    """Days-to-pXX per month (dot lines) with the mature mean +/- std band."""
    df = completion_percentiles
    if df.empty:
        return None
    fig = go.Figure()
    for pct in _PERCENTILES:
        subset = df[df["pct"] == pct].sort_values("month")
        if subset.empty:
            continue
        fig.add_scatter(
            x=list(subset["month"].astype(str).str[:7]),
            y=[None if pd.isna(v) else int(v) for v in subset["days"]],
            mode="lines+markers",
            name=f"p{pct}",
        )
    if completion_summary is not None and not completion_summary.empty:
        summary = completion_summary.iloc[0]
        months = sorted(df["month"].astype(str).str[:7].unique())
        for pct in _PERCENTILES:
            mean, std = summary.get(f"p{pct}_mean"), summary.get(f"p{pct}_std")
            if mean is None or pd.isna(mean):
                continue
            fig.add_scatter(
                x=months,
                y=[round(float(mean), 2)] * len(months),
                mode="lines",
                name=f"p{pct} mature mean",
                line={"dash": "dash", "width": 1},
                error_y={
                    "type": "constant",
                    "value": round(float(std), 2),
                    "visible": False,
                },
                showlegend=False,
            )
    fig.update_layout(
        title=_title("Days to percentile by event month", probe, proxy),
        xaxis_title="event month",
        yaxis_title="days",
    )
    return fig


def dual_overlay_figure(
    month_lag_cells: pd.DataFrame,
    dual_lag_cells: pd.DataFrame,
    probe: str,
    proxy: bool,
) -> go.Figure | None:
    """Median cumulative curve on each side: upstream (event -> source insert)
    vs local (event -> load) — the provider-lag vs ingestion-lag split."""
    sides = (("load side", _curves(month_lag_cells)), ("source side", _curves(dual_lag_cells)))
    fig = go.Figure()
    drew = False
    for name, curves in sides:
        if not curves:
            continue
        grid = sorted({day for curve in curves.values() for day in curve.index})
        aligned = pd.DataFrame(
            {m: c.reindex(grid).ffill().fillna(0.0) for m, c in curves.items()}
        )
        fig.add_scatter(
            x=grid,
            y=[round(float(v) * 100, 3) for v in aligned.median(axis=1)],
            mode="lines",
            name=name,
        )
        drew = True
    if not drew:
        return None
    fig.update_layout(
        title=_title("Dual lag: source vs load completion (median month)", probe, proxy),
        xaxis_title="lag (days)",
        yaxis_title="% of final rows",
        yaxis_range=[0, 105],
    )
    return fig


def dual_delta_figure(dual_delta: pd.DataFrame, probe: str, proxy: bool) -> go.Figure | None:
    """Histogram of per-row source->load delta days (local ingestion lag)."""
    usable = dual_delta[dual_delta["row_count"].notna()].sort_values("delta_day")
    if usable.empty:
        return None
    fig = go.Figure(
        go.Bar(
            x=[int(v) for v in usable["delta_day"]],
            y=[int(v) for v in usable["row_count"]],
            name="rows",
        )
    )
    fig.update_layout(
        title=_title("Source -> load delta (days)", probe, proxy),
        xaxis_title="delta (days)",
        yaxis_title="rows",
    )
    return fig


def batch_figure(
    batch_months: pd.DataFrame, probe: str, proxy: bool, expect_batchy: bool
) -> go.Figure | None:
    """Runs per month plus batch-level completion days (p50/p95 from month
    end); a step-function feed is expected wording when flagged."""
    if batch_months.empty:
        return None
    df = batch_months.sort_values("month")
    months = list(df["month"].astype(str).str[:7])
    fig = go.Figure()
    fig.add_bar(
        x=months,
        y=[None if pd.isna(v) else int(v) for v in df["runs"]],
        name="runs",
        yaxis="y",
        marker_color="#8ebbe3",
    )
    for pct in (50, 95):
        column = f"days_to_p{pct}"
        if column in df.columns:
            fig.add_scatter(
                x=months,
                y=[None if pd.isna(v) else int(v) for v in df[column]],
                mode="lines+markers",
                name=f"p{pct} days (batch)",
                yaxis="y2",
            )
    wording = "step-function feed (expected)" if expect_batchy else "batch feed"
    fig.update_layout(
        title=_title(f"Batch runs and completion — {wording}", probe, proxy),
        xaxis_title="event month",
        yaxis={"title": "runs per month"},
        yaxis2={"title": "days from month end", "overlaying": "y", "side": "right"},
    )
    return fig


def parity_figure(parity_months: pd.DataFrame, probe: str, proxy: bool) -> go.Figure | None:
    """Diverging bars: per-month count difference vs the parity partner."""
    if parity_months.empty:
        return None
    df = parity_months.sort_values("month")
    diffs = [None if pd.isna(v) else int(v) for v in df["diff"]]
    colors = [
        "#c0392b" if verdict != "match" else "#2f6fb2" for verdict in df["verdict"]
    ]
    fig = go.Figure(
        go.Bar(x=list(df["month"].astype(str).str[:7]), y=diffs, marker_color=colors)
    )
    partner = str(df["partner"].iloc[0]) if "partner" in df.columns else "partner"
    fig.update_layout(
        title=_title(f"Parity diff vs {partner}", probe, proxy),
        xaxis_title="event month",
        yaxis_title="left - right rows",
    )
    return fig


def figures_for_probe(
    probe_frames: dict[str, pd.DataFrame],
    probe: str,
    proxy: bool = False,
    expect_batchy: bool = False,
) -> dict[str, go.Figure]:
    """Every applicable figure for one probe, under stable keys (FIGURE_ORDER)."""
    figures: dict[str, go.Figure] = {}

    def put(key: str, figure: go.Figure | None) -> None:
        if figure is not None:
            figures[key] = figure

    if "month_volumes" in probe_frames:
        put("volume", volume_figure(probe_frames["month_volumes"], probe, proxy))
    if "month_lag_cells" in probe_frames:
        put(
            "completion_curves",
            completion_curves_figure(
                probe_frames["month_lag_cells"],
                probe_frames.get("month_volumes"),
                probe,
                proxy,
            ),
        )
        put(
            "completion_heatmap",
            completion_heatmap_figure(probe_frames["month_lag_cells"], probe, proxy),
        )
    if "completion_percentiles" in probe_frames:
        put(
            "percentiles",
            percentile_summary_figure(
                probe_frames["completion_percentiles"],
                probe_frames.get("completion_summary"),
                probe,
                proxy,
            ),
        )
    if "dual_lag_cells" in probe_frames and "month_lag_cells" in probe_frames:
        put(
            "dual_overlay",
            dual_overlay_figure(
                probe_frames["month_lag_cells"],
                probe_frames["dual_lag_cells"],
                probe,
                proxy,
            ),
        )
    if "dual_delta" in probe_frames:
        put("dual_delta", dual_delta_figure(probe_frames["dual_delta"], probe, proxy))
    if "batch_months" in probe_frames:
        put(
            "batch",
            batch_figure(probe_frames["batch_months"], probe, proxy, expect_batchy),
        )
    if "parity_months" in probe_frames:
        put("parity", parity_figure(probe_frames["parity_months"], probe, proxy))
    return dict(sorted(figures.items(), key=lambda kv: FIGURE_ORDER.index(kv[0])))
