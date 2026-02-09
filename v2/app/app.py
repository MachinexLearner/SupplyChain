"""
Supply Chain Intelligence Dashboard
====================================
Gradio app for the Oshkosh Defense supply chain platform.
Deploy on Databricks Apps or run locally with `python app.py`.

Connects to Unity Catalog tables via Databricks SQL Connector.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from functools import lru_cache

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════════
# Data layer — Databricks SQL Connector
# ═══════════════════════════════════════════════════════════════════════════════

CATALOG = os.environ.get("CATALOG", "supply_chain")

# Table registry
T = {
    "demand":       f"{CATALOG}.gold.oshkosh_monthly_demand_signals",
    "dod":          f"{CATALOG}.gold.dod_metrics_inputs_monthly",
    "trade_risk":   f"{CATALOG}.gold.trade_tariff_risk_monthly",
    "prophet":      f"{CATALOG}.gold.prophet_forecasts",
    "arima":        f"{CATALOG}.gold.arima_forecasts",
    "rf":           f"{CATALOG}.gold.random_forest_forecasts",
    "rf_fi":        f"{CATALOG}.gold.random_forest_feature_importance",
    "suppliers":    f"{CATALOG}.silver.supplier_geolocations",
    "commodity":    f"{CATALOG}.silver.commodity_prices_monthly",
    "weather":      f"{CATALOG}.silver.weather_risk_monthly",
    "contracts":    f"{CATALOG}.bronze.fpds_contracts",
    "gscpi":        f"{CATALOG}.bronze.nyfed_gscpi",
    "wto":          f"{CATALOG}.bronze.wto_trade_barometer",
}

_conn = None


def _get_connection():
    """Lazy singleton SQL connection."""
    global _conn
    if _conn is not None:
        return _conn
    try:
        from databricks import sql as dbsql
        from databricks.sdk.core import Config

        cfg = Config()
        http_path = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
        if not http_path:
            raise RuntimeError(
                "Set DATABRICKS_WAREHOUSE_ID env var to your SQL warehouse HTTP path "
                "(e.g. /sql/1.0/warehouses/abc123)"
            )
        _conn = dbsql.connect(
            server_hostname=cfg.host,
            http_path=http_path,
            credentials_provider=lambda: cfg.authenticate,
        )
        return _conn
    except Exception as e:
        print(f"SQL connection unavailable: {e}")
        return None


def query(table_key: str, sql_override: str | None = None, limit: int = 50_000) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame. Falls back to empty DF."""
    conn = _get_connection()
    if conn is None:
        return _demo_data(table_key)
    try:
        q = sql_override or f"SELECT * FROM {T[table_key]} LIMIT {limit}"
        with conn.cursor() as cur:
            cur.execute(q)
            return cur.fetchall_arrow().to_pandas()
    except Exception as e:
        print(f"Query failed for {table_key}: {e}")
        return pd.DataFrame()


# ── Demo / fallback data (so the UI renders even without a warehouse) ────────

def _demo_data(key: str) -> pd.DataFrame:
    """Generate plausible demo data for local testing without a SQL warehouse."""
    rng = np.random.default_rng(42)
    months = pd.date_range("2022-01-01", periods=36, freq="MS")

    if key == "demand":
        base = 50_000_000 + np.cumsum(rng.normal(500_000, 2_000_000, 36))
        return pd.DataFrame({
            "month": months,
            "total_obligations_usd": base,
            "prime_obligations_usd": base * 0.75,
            "subaward_obligations_usd": base * 0.25,
            "geo_risk_index": rng.uniform(1, 8, 36),
            "tariff_risk_index": rng.uniform(1, 6, 36),
            "commodity_cost_pressure": rng.uniform(-2, 12, 36),
            "weather_disruption_index": rng.uniform(0, 5, 36),
            "combined_risk_index": rng.uniform(2, 8, 36),
        })

    if key in ("prophet", "arima", "rf"):
        future = pd.date_range(months[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
        all_m = months.append(future)
        base_hist = 50_000_000 + np.cumsum(rng.normal(500_000, 2_000_000, 36))
        base_fut = base_hist[-1] + np.cumsum(rng.normal(300_000, 1_500_000, 12))
        vals = np.concatenate([base_hist, base_fut])
        noise = {"prophet": 0, "arima": 2_000_000, "rf": -1_500_000}[key]
        return pd.DataFrame({
            "month": all_m,
            "forecast_demand_usd": vals + noise,
            "forecast_lower": vals + noise - rng.uniform(3e6, 8e6, 48),
            "forecast_upper": vals + noise + rng.uniform(3e6, 8e6, 48),
        })

    if key == "rf_fi":
        features = [
            "demand_lag_1m", "demand_rolling_mean_12m", "demand_rolling_mean_6m",
            "demand_rolling_mean_3m", "demand_trend_3m", "months_since_start",
            "geo_risk_index", "tariff_risk_index", "commodity_cost_pressure",
            "demand_lag_3m", "demand_pct_change_1m", "is_q4",
            "weather_disruption_index", "demand_rolling_std_6m", "combined_risk_interaction",
            "demand_lag_6m", "month_sin", "month_cos", "demand_lag_12m", "quarter",
        ]
        imps = sorted(rng.dirichlet(np.ones(len(features))) * 1.0, reverse=True)
        return pd.DataFrame({
            "feature": features,
            "importance": imps,
            "rank": range(1, len(features) + 1),
            "model_type": "RandomForest",
        })

    if key == "dod":
        return pd.DataFrame({
            "month": months,
            "requirements_objective_proxy": 50_000_000 + rng.normal(0, 3e6, 36),
            "risk_adjusted_ro": 55_000_000 + rng.normal(0, 3e6, 36),
            "approved_acquisition_objective_proxy": 120_000_000 + rng.normal(0, 5e6, 36),
            "days_of_supply_proxy": rng.uniform(35, 80, 36),
            "nmcs_risk_indicator": rng.choice(["LOW_RISK", "ELEVATED_RISK", "HIGH_RISK"], 36, p=[0.5, 0.35, 0.15]),
            "demand_volatility_category": rng.choice(["LOW", "MEDIUM", "HIGH"], 36),
            "coefficient_of_variation": rng.uniform(0.1, 0.5, 36),
            "forecast_method_recommendation": "Ensemble",
        })

    if key == "commodity":
        commodities = [
            ("Crude Oil WTI", "Energy", "Vehicle fuel & lubricants", 72),
            ("Steel HRC", "Industrial Metals", "Armor & chassis", 850),
            ("Aluminum", "Industrial Metals", "Vehicle body panels", 2400),
            ("Copper", "Industrial Metals", "Wiring & electronics", 8500),
            ("Lithium", "Battery Materials", "Hybrid powertrains", 25000),
        ]
        rows = []
        for m in months:
            for name, cat, use, base_price in commodities:
                price = base_price * (1 + rng.normal(0, 0.05))
                rows.append({
                    "month": m, "commodity_name": name, "category": cat,
                    "defense_use": use, "close_price": price,
                    "pct_change_1mo": rng.normal(0, 3),
                    "pct_change_3mo": rng.normal(0, 6),
                    "cost_pressure_score": rng.normal(2, 5),
                })
        return pd.DataFrame(rows)

    if key == "suppliers":
        names = [f"Supplier_{i}" for i in range(50)]
        states = rng.choice(["WI", "TX", "MI", "OH", "CA", "PA", "IN", "AL", "VA", "GA"], 50)
        cats = rng.choice(["POWERTRAIN", "ARMOR", "ELECTRONICS", "SUSPENSION", "MATERIALS", "HYDRAULICS"], 50)
        sizes = rng.choice(["SMALL", "LARGE", "OTHER"], 50, p=[0.5, 0.35, 0.15])
        return pd.DataFrame({
            "supplier_name": names, "uei": [f"UEI{i:06d}" for i in range(50)],
            "cage_code": [f"C{i:04d}" for i in range(50)],
            "city": ["City"] * 50, "state": states, "country": ["USA"] * 50,
            "lat": rng.uniform(30, 48, 50), "lon": rng.uniform(-120, -75, 50),
            "region_group": rng.choice(["US_MIDWEST", "US_SOUTH", "US_WEST", "US_NORTHEAST"], 50),
            "distance_to_nearest_oshkosh_facility_km": rng.uniform(50, 2000, 50),
            "subsystem_category": cats, "company_size": sizes,
            "naics_code_primary": rng.choice(["336992", "332312", "334511", "336350"], 50),
        })

    if key == "contracts":
        n = 100
        return pd.DataFrame({
            "contract_id": [f"W56HZV-{rng.integers(20,25)}-C-{rng.integers(1000,9999)}" for _ in range(n)],
            "vendor_name": rng.choice(["Oshkosh Defense", "BAE Systems", "General Dynamics", "L3Harris", "Textron"], n),
            "obligated_amount": rng.uniform(100_000, 50_000_000, n),
            "fiscal_year": rng.choice([2022, 2023, 2024, 2025], n),
            "psc_code": rng.choice(["2320", "2350", "2510", "2590", "2915"], n),
            "naics_code": rng.choice(["336992", "336120"], n),
            "description": ["Defense vehicle contract"] * n,
            "signed_date": pd.date_range("2022-01-01", periods=n, freq="12D").strftime("%Y-%m-%d"),
            "vendor_state": rng.choice(["WI", "VA", "TX", "MI"], n),
        })

    if key == "gscpi":
        return pd.DataFrame({
            "as_of_date": months,
            "value": np.cumsum(rng.normal(-0.05, 0.3, 36)),
            "indicator_name": "GSCPI",
        })

    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#1B6EF3",
    "prophet": "#1B6EF3",
    "arima": "#F59E0B",
    "rf": "#10B981",
    "risk_geo": "#EF4444",
    "risk_tariff": "#F59E0B",
    "risk_commodity": "#8B5CF6",
    "risk_weather": "#06B6D4",
    "good": "#10B981",
    "warn": "#F59E0B",
    "bad": "#EF4444",
    "bg": "#0F172A",
    "card": "#1E293B",
    "text": "#E2E8F0",
    "muted": "#94A3B8",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
    margin=dict(l=50, r=30, t=60, b=50),
)


def _apply_layout(fig, **kwargs):
    merged = {**LAYOUT_DEFAULTS, **kwargs}
    fig.update_layout(**merged)
    fig.update_xaxes(gridcolor="#334155", zeroline=False)
    fig.update_yaxes(gridcolor="#334155", zeroline=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def build_dashboard():
    demand = query("demand")
    dod = query("dod")
    gscpi = query("gscpi")

    kpis = {}
    trend_fig = go.Figure()
    health_fig = go.Figure()

    if not demand.empty:
        demand["month"] = pd.to_datetime(demand["month"])
        demand = demand.sort_values("month")
        recent3 = demand.tail(3)["total_obligations_usd"]
        recent12 = demand.tail(12)["total_obligations_usd"]
        prior3 = demand.tail(6).head(3)["total_obligations_usd"]
        qoq = (recent3.mean() - prior3.mean()) / prior3.mean() * 100 if prior3.mean() > 0 else 0

        kpis["Last Quarter Avg"] = f"${recent3.mean():,.0f}"
        kpis["12-Month Avg"] = f"${recent12.mean():,.0f}"
        kpis["QoQ Change"] = f"{qoq:+.1f}%"

        # Demand trend chart
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=demand["month"], y=demand["total_obligations_usd"],
            mode="lines+markers", name="Total Obligations",
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=4),
        ))
        _apply_layout(trend_fig, title="Monthly Demand (Total Obligations USD)",
                      yaxis_title="USD", xaxis_title="")

    if not dod.empty:
        dod["month"] = pd.to_datetime(dod["month"])
        latest = dod.sort_values("month").iloc[-1]
        kpis["Days of Supply"] = f"{latest.get('days_of_supply_proxy', 0):.0f}"
        kpis["NMCS Risk"] = str(latest.get("nmcs_risk_indicator", "N/A"))

    if not gscpi.empty:
        gscpi["as_of_date"] = pd.to_datetime(gscpi["as_of_date"])
        latest_g = gscpi.sort_values("as_of_date").iloc[-1]
        kpis["GSCPI"] = f"{float(latest_g['value']):.2f}"

    # Health gauge
    scores = _compute_health_scores(demand, dod)
    if scores:
        overall = np.mean(list(scores.values()))
        health_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall,
            title={"text": "Overall Health Score", "font": {"size": 18, "color": COLORS["text"]}},
            number={"font": {"size": 48, "color": COLORS["text"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                "bar": {"color": COLORS["good"] if overall >= 70 else COLORS["warn"] if overall >= 40 else COLORS["bad"]},
                "bgcolor": COLORS["card"],
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                    {"range": [70, 100], "color": "rgba(16,185,129,0.15)"},
                ],
            },
        ))
        _apply_layout(health_fig, height=300)

    # KPI markdown
    kpi_md = "| Metric | Value |\n|--------|-------|\n"
    for k, v in kpis.items():
        kpi_md += f"| **{k}** | {v} |\n"

    # Health breakdown markdown
    health_md = ""
    if scores:
        health_md = "| Dimension | Score | Status |\n|-----------|-------|--------|\n"
        for dim, sc in sorted(scores.items(), key=lambda x: -x[1]):
            status = "Good" if sc >= 70 else "Fair" if sc >= 40 else "Poor"
            health_md += f"| {dim} | {sc:.0f}/100 | {status} |\n"

    return kpi_md, trend_fig, health_fig, health_md


def _compute_health_scores(demand, dod):
    scores = {}
    if not demand.empty:
        demand = demand.sort_values("month")
        y = demand.tail(12)["total_obligations_usd"].values
        if len(y) > 1:
            cv = y.std() / y.mean()
            scores["Demand Stability"] = max(0, min(100, 100 - cv * 200))
        risk_cols = [c for c in ["geo_risk_index", "tariff_risk_index", "weather_disruption_index"]
                     if c in demand.columns]
        if risk_cols:
            avg_risk = demand.tail(3)[risk_cols].mean().mean()
            scores["Risk Environment"] = max(0, min(100, 100 - avg_risk * 10))
    if not dod.empty:
        dod = dod.sort_values("month")
        latest = dod.iloc[-1]
        dos = float(latest.get("days_of_supply_proxy", 0))
        scores["DoD Readiness"] = min(100, dos / 60 * 100)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Forecasting
# ═══════════════════════════════════════════════════════════════════════════════

def build_forecasting():
    demand = query("demand")
    prophet = query("prophet")
    arima = query("arima")
    rf = query("rf")

    fig = go.Figure()

    # Actuals
    if not demand.empty:
        demand["month"] = pd.to_datetime(demand["month"])
        demand = demand.sort_values("month")
        fig.add_trace(go.Scatter(
            x=demand["month"], y=demand["total_obligations_usd"],
            mode="lines", name="Actuals",
            line=dict(color=COLORS["text"], width=2),
        ))

    # Forecasts
    for df, name, color in [
        (prophet, "Prophet", COLORS["prophet"]),
        (arima, "ARIMA", COLORS["arima"]),
        (rf, "Random Forest", COLORS["rf"]),
    ]:
        if not df.empty:
            df["month"] = pd.to_datetime(df["month"])
            df = df.sort_values("month")
            fig.add_trace(go.Scatter(
                x=df["month"], y=df["forecast_demand_usd"],
                mode="lines", name=name,
                line=dict(color=color, width=2.5, dash="dot"),
            ))
            if "forecast_lower" in df.columns and "forecast_upper" in df.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([df["month"], df["month"][::-1]]),
                    y=pd.concat([df["forecast_upper"], df["forecast_lower"][::-1]]),
                    fill="toself", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba")
                    if "rgb" in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{name} 95% CI", showlegend=False,
                ))

    _apply_layout(fig, title="Multi-Model Demand Forecast Comparison",
                  yaxis_title="USD", xaxis_title="", height=500,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Table — show the last 6 months of each forecast (future or most recent)
    table_rows = []
    for df, name in [(prophet, "Prophet"), (arima, "ARIMA"), (rf, "Random Forest")]:
        if not df.empty:
            df["month"] = pd.to_datetime(df["month"])
            future = df.sort_values("month").tail(6)
            for _, r in future.iterrows():
                table_rows.append({
                    "Model": name,
                    "Month": r["month"].strftime("%b %Y"),
                    "Forecast": f"${r['forecast_demand_usd']:,.0f}",
                    "Lower": f"${r.get('forecast_lower', 0):,.0f}",
                    "Upper": f"${r.get('forecast_upper', 0):,.0f}",
                })
    table_df = pd.DataFrame(table_rows) if table_rows else pd.DataFrame()

    return fig, table_df


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Demand Drivers (Random Forest Feature Importance)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    "demand_lag_1m": "Prior Month Demand",
    "demand_rolling_mean_12m": "12-Month Rolling Avg",
    "demand_rolling_mean_6m": "6-Month Rolling Avg",
    "demand_rolling_mean_3m": "3-Month Rolling Avg",
    "demand_trend_3m": "3-Month Trend",
    "demand_trend_6m": "6-Month Trend",
    "demand_pct_change_1m": "MoM % Change",
    "demand_pct_change_3m": "3-Mo % Change",
    "demand_lag_3m": "3-Month Lag",
    "demand_lag_6m": "6-Month Lag",
    "demand_lag_12m": "12-Month Lag",
    "demand_rolling_std_6m": "6-Mo Volatility",
    "demand_rolling_std_3m": "3-Mo Volatility",
    "demand_rolling_std_12m": "12-Mo Volatility",
    "demand_rolling_min_3m": "3-Mo Min",
    "demand_rolling_max_3m": "3-Mo Max",
    "geo_risk_index": "Geopolitical Risk",
    "tariff_risk_index": "Tariff / Trade Risk",
    "commodity_cost_pressure": "Commodity Cost Pressure",
    "weather_disruption_index": "Weather Disruption",
    "combined_risk_interaction": "Combined Risk (Geo x Tariff)",
    "geo_risk_lag_1m": "Geopolitical Risk (Lagged)",
    "tariff_risk_lag_1m": "Tariff Risk (Lagged)",
    "commodity_lag_1m": "Commodity Pressure (Lagged)",
    "commodity_trend_3m": "Commodity 3-Mo Trend",
    "weather_lag_1m": "Weather Disruption (Lagged)",
    "is_q4": "Fiscal Year-End (Q4)",
    "month_of_year": "Month of Year",
    "quarter": "Quarter",
    "month_sin": "Seasonality (Sin)",
    "month_cos": "Seasonality (Cos)",
    "months_since_start": "Time Trend",
}

FEATURE_CATEGORIES = {
    "Demand History": ["demand_lag", "demand_rolling", "demand_trend", "demand_pct"],
    "Risk Signals": ["geo_risk", "tariff_risk", "commodity", "weather", "combined_risk"],
    "Seasonality": ["is_q4", "month_of_year", "quarter", "month_sin", "month_cos"],
    "Time Trend": ["months_since_start"],
}


def _categorize_feature(name):
    for cat, prefixes in FEATURE_CATEGORIES.items():
        if any(name.startswith(p) for p in prefixes):
            return cat
    return "Other"


def build_demand_drivers(top_n=20):
    fi = query("rf_fi")
    if fi.empty:
        return go.Figure(), go.Figure(), go.Figure(), pd.DataFrame(), "No feature importance data available."

    fi = fi.sort_values("importance", ascending=False).head(top_n)
    fi["label"] = fi["feature"].map(lambda f: FEATURE_LABELS.get(f, f.replace("_", " ").title()))
    fi["category"] = fi["feature"].map(_categorize_feature)

    cat_colors = {
        "Demand History": "#3B82F6",
        "Risk Signals": "#EF4444",
        "Seasonality": "#F59E0B",
        "Time Trend": "#8B5CF6",
        "Other": "#6B7280",
    }
    fi["color"] = fi["category"].map(cat_colors)

    # ── Main bar chart ───────────────────────────────────────────────────────
    bar_fig = go.Figure(go.Bar(
        y=fi["label"][::-1],
        x=fi["importance"][::-1],
        orientation="h",
        marker=dict(
            color=fi["color"][::-1],
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        text=[f"{v:.4f}" for v in fi["importance"][::-1]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    _apply_layout(bar_fig,
                  title=f"Top {top_n} Demand Drivers — Random Forest Feature Importance",
                  xaxis_title="Importance Score",
                  height=max(450, top_n * 28),
                  yaxis=dict(tickfont=dict(size=12)))

    # ── Category donut chart ─────────────────────────────────────────────────
    cat_sums = fi.groupby("category")["importance"].sum().sort_values(ascending=False)
    donut_fig = go.Figure(go.Pie(
        labels=cat_sums.index,
        values=cat_sums.values,
        hole=0.55,
        marker=dict(colors=[cat_colors.get(c, "#6B7280") for c in cat_sums.index]),
        textinfo="label+percent",
        textfont=dict(size=12),
    ))
    _apply_layout(donut_fig, title="Importance by Category", height=350,
                  showlegend=False)

    # ── Cumulative importance chart ──────────────────────────────────────────
    fi_sorted = fi.sort_values("importance", ascending=False)
    fi_sorted["cumulative"] = fi_sorted["importance"].cumsum()
    fi_sorted["cumulative_pct"] = fi_sorted["cumulative"] / fi_sorted["importance"].sum() * 100

    cum_fig = make_subplots(specs=[[{"secondary_y": True}]])
    cum_fig.add_trace(go.Bar(
        x=fi_sorted["label"], y=fi_sorted["importance"],
        name="Individual", marker_color=COLORS["primary"],
    ), secondary_y=False)
    cum_fig.add_trace(go.Scatter(
        x=fi_sorted["label"], y=fi_sorted["cumulative_pct"],
        name="Cumulative %", mode="lines+markers",
        line=dict(color=COLORS["rf"], width=2.5),
        marker=dict(size=6),
    ), secondary_y=True)
    cum_fig.add_hline(y=80, line_dash="dash", line_color=COLORS["warn"],
                      annotation_text="80% threshold", secondary_y=True)
    _apply_layout(cum_fig, title="Pareto Analysis — Cumulative Feature Importance",
                  height=400, xaxis_tickangle=-45)
    cum_fig.update_yaxes(title_text="Importance", secondary_y=False)
    cum_fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])

    # ── Table ────────────────────────────────────────────────────────────────
    table_df = fi[["rank", "label", "category", "importance"]].rename(columns={
        "rank": "Rank", "label": "Feature", "category": "Category", "importance": "Score"
    })
    table_df["Score"] = table_df["Score"].map(lambda x: f"{x:.4f}")

    # ── Insights text ────────────────────────────────────────────────────────
    top_cat = cat_sums.index[0]
    top_cat_pct = cat_sums.values[0] / cat_sums.sum() * 100
    n_80 = (fi_sorted["cumulative_pct"] <= 80).sum() + 1
    insights = textwrap.dedent(f"""\
    ### Key Insights

    - **{top_cat}** features account for **{top_cat_pct:.0f}%** of total model importance
    - The top **{n_80}** features explain 80% of forecast variance (Pareto)
    - **{fi.iloc[0]['label']}** is the single strongest predictor ({fi.iloc[0]['importance']:.4f})
    """)

    risk_feats = fi[fi["category"] == "Risk Signals"]
    if not risk_feats.empty:
        risk_pct = risk_feats["importance"].sum() / fi["importance"].sum() * 100
        insights += f"- Risk signals contribute **{risk_pct:.1f}%** — "
        if risk_pct > 30:
            insights += "demand is highly sensitive to external risk factors\n"
        elif risk_pct > 15:
            insights += "meaningful but not dominant risk sensitivity\n"
        else:
            insights += "demand is primarily momentum-driven\n"

    return bar_fig, donut_fig, cum_fig, table_df, insights


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Risk Monitor
# ═══════════════════════════════════════════════════════════════════════════════

def build_risk_monitor():
    demand = query("demand")
    commodity = query("commodity")

    risk_fig = go.Figure()
    commodity_fig = go.Figure()
    pressure_fig = go.Figure()

    if not demand.empty:
        demand["month"] = pd.to_datetime(demand["month"])
        demand = demand.sort_values("month")

        for col, name, color in [
            ("geo_risk_index", "Geopolitical", COLORS["risk_geo"]),
            ("tariff_risk_index", "Tariff / Trade", COLORS["risk_tariff"]),
            ("commodity_cost_pressure", "Commodity Cost", COLORS["risk_commodity"]),
            ("weather_disruption_index", "Weather", COLORS["risk_weather"]),
        ]:
            if col in demand.columns:
                risk_fig.add_trace(go.Scatter(
                    x=demand["month"], y=demand[col],
                    mode="lines", name=name,
                    line=dict(color=color, width=2),
                ))
        _apply_layout(risk_fig, title="Risk Indices Over Time",
                      yaxis_title="Index Value", height=420,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))

    if not commodity.empty:
        commodity["month"] = pd.to_datetime(commodity["month"])
        for name in commodity["commodity_name"].unique():
            sub = commodity[commodity["commodity_name"] == name].sort_values("month")
            commodity_fig.add_trace(go.Scatter(
                x=sub["month"], y=sub["close_price"],
                mode="lines", name=name, line=dict(width=2),
            ))
        _apply_layout(commodity_fig, title="Defense-Critical Commodity Prices",
                      yaxis_title="Price (USD)", height=420,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))

        # Cost pressure
        latest_m = commodity["month"].max()
        latest = commodity[commodity["month"] == latest_m]
        pressure_fig = go.Figure(go.Bar(
            x=latest["commodity_name"],
            y=latest["cost_pressure_score"],
            marker_color=[
                COLORS["bad"] if v > 5 else COLORS["warn"] if v > 0 else COLORS["good"]
                for v in latest["cost_pressure_score"]
            ],
            text=[f"{v:.1f}" for v in latest["cost_pressure_score"]],
            textposition="outside",
        ))
        _apply_layout(pressure_fig, title="Current Cost Pressure by Commodity",
                      yaxis_title="Pressure Score", height=350)

    return risk_fig, commodity_fig, pressure_fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Suppliers
# ═══════════════════════════════════════════════════════════════════════════════

def build_suppliers(name_filter="", state_filter="", subsystem_filter=""):
    df = query("suppliers")
    if df.empty:
        return pd.DataFrame(), go.Figure(), go.Figure(), "No supplier data."

    mask = pd.Series(True, index=df.index)
    if name_filter:
        mask &= df["supplier_name"].str.contains(name_filter, case=False, na=False)
    if state_filter:
        mask &= df["state"].str.upper() == state_filter.upper()
    if subsystem_filter:
        mask &= df["subsystem_category"].str.contains(subsystem_filter, case=False, na=False)
    filtered = df[mask]

    # Map
    map_fig = go.Figure()
    if "lat" in filtered.columns and "lon" in filtered.columns:
        map_fig = px.scatter_geo(
            filtered, lat="lat", lon="lon", color="subsystem_category",
            hover_name="supplier_name",
            hover_data=["state", "company_size", "distance_to_nearest_oshkosh_facility_km"],
            scope="usa", title="Supplier Locations",
        )
        map_fig.update_layout(**LAYOUT_DEFAULTS, height=450, geo=dict(
            bgcolor=COLORS["bg"], lakecolor=COLORS["card"],
            landcolor=COLORS["card"], showland=True,
        ))

    # Subsystem distribution
    sub_counts = filtered["subsystem_category"].value_counts()
    pie_fig = go.Figure(go.Pie(
        labels=sub_counts.index, values=sub_counts.values,
        hole=0.4, textinfo="label+value",
    ))
    _apply_layout(pie_fig, title="Suppliers by Subsystem", height=350, showlegend=False)

    # Table
    table_df = filtered[["supplier_name", "state", "subsystem_category", "company_size",
                          "distance_to_nearest_oshkosh_facility_km"]].copy()
    table_df.columns = ["Supplier", "State", "Subsystem", "Size", "Distance (km)"]
    if "Distance (km)" in table_df.columns:
        table_df["Distance (km)"] = table_df["Distance (km)"].round(0)

    summary = f"**{len(filtered)}** suppliers"
    if "company_size" in filtered.columns:
        small = (filtered["company_size"].str.contains("SMALL", case=False, na=False)).sum()
        summary += f"  |  Small business: **{small}** ({small/len(filtered)*100:.0f}%)" if len(filtered) > 0 else ""
    if "region_group" in filtered.columns:
        n_regions = filtered["region_group"].nunique()
        summary += f"  |  Regions: **{n_regions}**"

    return table_df, map_fig, pie_fig, summary


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Contracts
# ═══════════════════════════════════════════════════════════════════════════════

def build_contracts(vendor_filter="", fy_filter="", min_amount=0):
    df = query("contracts")
    if df.empty:
        return pd.DataFrame(), go.Figure(), "No contract data."

    df["obligated_amount"] = pd.to_numeric(df["obligated_amount"], errors="coerce").fillna(0)
    mask = pd.Series(True, index=df.index)
    if vendor_filter:
        mask &= df["vendor_name"].str.contains(vendor_filter, case=False, na=False)
    if fy_filter and str(fy_filter).isdigit():
        mask &= df["fiscal_year"].astype(str) == str(fy_filter)
    if min_amount > 0:
        mask &= df["obligated_amount"] >= min_amount
    filtered = df[mask].sort_values("obligated_amount", ascending=False)

    # Top vendors chart
    top_vendors = filtered.groupby("vendor_name")["obligated_amount"].sum().nlargest(10)
    bar_fig = go.Figure(go.Bar(
        x=top_vendors.values, y=top_vendors.index, orientation="h",
        marker_color=COLORS["primary"],
        text=[f"${v:,.0f}" for v in top_vendors.values],
        textposition="outside",
    ))
    _apply_layout(bar_fig, title="Top Vendors by Obligated Amount", height=400,
                  xaxis_title="Total Obligated (USD)")

    # Table
    table_df = filtered.head(50)[["contract_id", "vendor_name", "obligated_amount",
                                   "fiscal_year", "psc_code", "signed_date"]].copy()
    table_df.columns = ["Contract ID", "Vendor", "Amount ($)", "FY", "PSC", "Signed"]
    table_df["Amount ($)"] = table_df["Amount ($)"].map(lambda x: f"${x:,.0f}")

    summary = f"**{len(filtered)}** contracts  |  Total: **${filtered['obligated_amount'].sum():,.0f}**"
    return table_df, bar_fig, summary


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: DoD Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def build_dod_metrics():
    dod = query("dod")
    if dod.empty:
        return go.Figure(), go.Figure(), go.Figure(), "No DoD metrics data."

    dod["month"] = pd.to_datetime(dod["month"])
    dod = dod.sort_values("month")
    latest = dod.iloc[-1]

    # Days of Supply gauge
    dos = float(latest.get("days_of_supply_proxy", 0))
    dos_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dos,
        title={"text": "Days of Supply"},
        delta={"reference": 60, "relative": False, "valueformat": ".0f"},
        number={"suffix": " days", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 120]},
            "bar": {"color": COLORS["good"] if dos >= 60 else COLORS["warn"] if dos >= 30 else COLORS["bad"]},
            "bgcolor": COLORS["card"],
            "threshold": {"line": {"color": COLORS["text"], "width": 3}, "thickness": 0.75, "value": 60},
        },
    ))
    _apply_layout(dos_fig, height=280)

    # DoS over time
    dos_trend = go.Figure()
    dos_trend.add_trace(go.Scatter(
        x=dod["month"], y=dod["days_of_supply_proxy"],
        mode="lines+markers", name="Days of Supply",
        line=dict(color=COLORS["primary"], width=2),
    ))
    dos_trend.add_hline(y=60, line_dash="dash", line_color=COLORS["good"],
                        annotation_text="Target: 60 days")
    dos_trend.add_hline(y=30, line_dash="dash", line_color=COLORS["bad"],
                        annotation_text="Critical: 30 days")
    _apply_layout(dos_trend, title="Days of Supply — Trend", yaxis_title="Days", height=350)

    # RO over time
    ro_fig = go.Figure()
    if "requirements_objective_proxy" in dod.columns:
        ro_fig.add_trace(go.Scatter(
            x=dod["month"], y=dod["requirements_objective_proxy"],
            mode="lines", name="RO", line=dict(color=COLORS["primary"], width=2),
        ))
    if "risk_adjusted_ro" in dod.columns:
        ro_fig.add_trace(go.Scatter(
            x=dod["month"], y=dod["risk_adjusted_ro"],
            mode="lines", name="Risk-Adjusted RO",
            line=dict(color=COLORS["warn"], width=2, dash="dash"),
        ))
    _apply_layout(ro_fig, title="Requirements Objective (RO)", yaxis_title="USD", height=350,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))

    # NMCS summary
    nmcs = str(latest.get("nmcs_risk_indicator", "N/A"))
    cv = float(latest.get("coefficient_of_variation", 0))
    vol = str(latest.get("demand_volatility_category", "N/A"))
    rec = str(latest.get("forecast_method_recommendation", "N/A"))
    summary = textwrap.dedent(f"""\
    ### Latest DoD Metrics ({latest['month'].strftime('%B %Y')})

    | Metric | Value |
    |--------|-------|
    | **NMCS Risk** | {nmcs} |
    | **Demand Volatility** | {vol} (CV: {cv:.2f}) |
    | **Recommended Model** | {rec} |
    | **Days of Supply** | {dos:.0f} days |
    """)

    return dos_fig, dos_trend, ro_fig, summary


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Macro Economy
# ═══════════════════════════════════════════════════════════════════════════════

def build_macro():
    gscpi = query("gscpi")

    gscpi_fig = go.Figure()
    if not gscpi.empty:
        gscpi["as_of_date"] = pd.to_datetime(gscpi["as_of_date"])
        gscpi = gscpi.sort_values("as_of_date")
        gscpi["value"] = pd.to_numeric(gscpi["value"], errors="coerce")

        gscpi_fig.add_trace(go.Scatter(
            x=gscpi["as_of_date"], y=gscpi["value"],
            mode="lines", name="GSCPI",
            line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(27,110,243,0.1)",
        ))
        gscpi_fig.add_hline(y=0, line_color=COLORS["muted"], line_dash="dash")
        gscpi_fig.add_hline(y=1, line_color=COLORS["bad"], line_dash="dot",
                            annotation_text="Elevated pressure")
        gscpi_fig.add_hline(y=-1, line_color=COLORS["good"], line_dash="dot",
                            annotation_text="Low pressure")
        _apply_layout(gscpi_fig,
                      title="NY Fed Global Supply Chain Pressure Index (GSCPI)",
                      yaxis_title="Standard Deviations from Mean", height=420)

    latest_val = float(gscpi.iloc[-1]["value"]) if not gscpi.empty else 0
    level = "Elevated" if latest_val > 1 else "Normal" if latest_val > -1 else "Low"
    summary = textwrap.dedent(f"""\
    ### Global Supply Chain Pressure

    | Indicator | Value | Level |
    |-----------|-------|-------|
    | **GSCPI** | {latest_val:.2f} | {level} |

    *The GSCPI measures global supply chain conditions. Values above 0 indicate above-average pressure;
    above 1.0 signals elevated stress. Below -1.0 indicates unusually low pressure.*
    """)

    return gscpi_fig, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0F172A",
    body_background_fill_dark="#0F172A",
    block_background_fill="#1E293B",
    block_background_fill_dark="#1E293B",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_label_text_color="#E2E8F0",
    block_label_text_color_dark="#E2E8F0",
    block_title_text_color="#E2E8F0",
    block_title_text_color_dark="#E2E8F0",
    body_text_color="#E2E8F0",
    body_text_color_dark="#E2E8F0",
    input_background_fill="#1E293B",
    input_background_fill_dark="#1E293B",
    button_primary_background_fill="#1B6EF3",
    button_primary_background_fill_dark="#1B6EF3",
    button_primary_text_color="#FFFFFF",
)

CSS = """
.gradio-container { max-width: 1400px !important; }
.tab-nav button { font-size: 15px !important; font-weight: 600 !important; }
footer { display: none !important; }
"""


def create_app():
    with gr.Blocks(theme=THEME, css=CSS, title="Supply Chain Intelligence") as app:

        gr.Markdown(
            "# Supply Chain Intelligence Dashboard\n"
            "*Oshkosh Defense — Real-time pipeline analytics from Unity Catalog*",
        )

        # ── Dashboard ────────────────────────────────────────────────────────
        with gr.Tab("Dashboard", id="dashboard"):
            dash_btn = gr.Button("Refresh Dashboard", variant="primary", size="sm")
            with gr.Row():
                with gr.Column(scale=1):
                    dash_kpis = gr.Markdown("Loading...")
                    dash_health_gauge = gr.Plot(label="Health Score")
                with gr.Column(scale=2):
                    dash_trend = gr.Plot(label="Demand Trend")
            dash_health_table = gr.Markdown("")

            def _refresh_dash():
                kpi_md, trend, gauge, health_md = build_dashboard()
                return kpi_md, trend, gauge, health_md

            dash_btn.click(_refresh_dash, outputs=[dash_kpis, dash_trend, dash_health_gauge, dash_health_table])
            app.load(_refresh_dash, outputs=[dash_kpis, dash_trend, dash_health_gauge, dash_health_table])

        # ── Forecasting ──────────────────────────────────────────────────────
        with gr.Tab("Forecasting", id="forecasting"):
            fc_btn = gr.Button("Refresh Forecasts", variant="primary", size="sm")
            fc_chart = gr.Plot(label="Multi-Model Forecast")
            fc_table = gr.Dataframe(label="Forecast Details", interactive=False)

            def _refresh_fc():
                fig, tbl = build_forecasting()
                return fig, tbl

            fc_btn.click(_refresh_fc, outputs=[fc_chart, fc_table])
            app.load(_refresh_fc, outputs=[fc_chart, fc_table])

        # ── Demand Drivers ───────────────────────────────────────────────────
        with gr.Tab("Demand Drivers", id="drivers"):
            gr.Markdown("### Random Forest Feature Importance\n"
                        "Understand which factors drive the demand forecast model's predictions.")
            dd_topn = gr.Slider(5, 30, value=20, step=1, label="Number of features to show")
            dd_btn = gr.Button("Analyze Drivers", variant="primary", size="sm")

            dd_bar = gr.Plot(label="Feature Importance")
            with gr.Row():
                with gr.Column():
                    dd_donut = gr.Plot(label="By Category")
                with gr.Column():
                    dd_insights = gr.Markdown("")
            dd_pareto = gr.Plot(label="Pareto Analysis")
            dd_table = gr.Dataframe(label="Feature Details", interactive=False)

            def _refresh_dd(topn):
                bar, donut, pareto, tbl, insights = build_demand_drivers(int(topn))
                return bar, donut, pareto, tbl, insights

            dd_btn.click(_refresh_dd, inputs=[dd_topn],
                         outputs=[dd_bar, dd_donut, dd_pareto, dd_table, dd_insights])
            app.load(lambda: _refresh_dd(20),
                     outputs=[dd_bar, dd_donut, dd_pareto, dd_table, dd_insights])

        # ── Risk Monitor ─────────────────────────────────────────────────────
        with gr.Tab("Risk Monitor", id="risk"):
            risk_btn = gr.Button("Refresh Risk Data", variant="primary", size="sm")
            risk_chart = gr.Plot(label="Risk Indices")
            with gr.Row():
                commodity_chart = gr.Plot(label="Commodity Prices")
                pressure_chart = gr.Plot(label="Cost Pressure")

            def _refresh_risk():
                return build_risk_monitor()

            risk_btn.click(_refresh_risk, outputs=[risk_chart, commodity_chart, pressure_chart])
            app.load(_refresh_risk, outputs=[risk_chart, commodity_chart, pressure_chart])

        # ── Suppliers ────────────────────────────────────────────────────────
        with gr.Tab("Suppliers", id="suppliers"):
            with gr.Row():
                sup_name = gr.Textbox(label="Supplier Name", placeholder="Search...")
                sup_state = gr.Textbox(label="State", placeholder="e.g. WI")
                sup_sub = gr.Textbox(label="Subsystem", placeholder="e.g. ARMOR")
                sup_btn = gr.Button("Search", variant="primary")

            sup_summary = gr.Markdown("")
            with gr.Row():
                sup_map = gr.Plot(label="Map")
                sup_pie = gr.Plot(label="By Subsystem")
            sup_table = gr.Dataframe(label="Suppliers", interactive=False)

            def _search_sup(name, state, sub):
                tbl, m, pie, summary = build_suppliers(name, state, sub)
                return tbl, m, pie, summary

            sup_btn.click(_search_sup, inputs=[sup_name, sup_state, sup_sub],
                          outputs=[sup_table, sup_map, sup_pie, sup_summary])
            app.load(lambda: _search_sup("", "", ""),
                     outputs=[sup_table, sup_map, sup_pie, sup_summary])

        # ── Contracts ────────────────────────────────────────────────────────
        with gr.Tab("Contracts", id="contracts"):
            with gr.Row():
                ct_vendor = gr.Textbox(label="Vendor Name", placeholder="Search...")
                ct_fy = gr.Textbox(label="Fiscal Year", placeholder="e.g. 2024")
                ct_min = gr.Number(label="Min Amount ($)", value=0)
                ct_btn = gr.Button("Search", variant="primary")

            ct_summary = gr.Markdown("")
            ct_chart = gr.Plot(label="Top Vendors")
            ct_table = gr.Dataframe(label="Contracts", interactive=False)

            def _search_ct(vendor, fy, min_amt):
                tbl, fig, summary = build_contracts(vendor, fy, min_amt)
                return tbl, fig, summary

            ct_btn.click(_search_ct, inputs=[ct_vendor, ct_fy, ct_min],
                         outputs=[ct_table, ct_chart, ct_summary])
            app.load(lambda: _search_ct("", "", 0),
                     outputs=[ct_table, ct_chart, ct_summary])

        # ── DoD Metrics ──────────────────────────────────────────────────────
        with gr.Tab("DoD Metrics", id="dod"):
            dod_btn = gr.Button("Refresh Metrics", variant="primary", size="sm")
            with gr.Row():
                dod_gauge = gr.Plot(label="Days of Supply")
                dod_summary = gr.Markdown("")
            with gr.Row():
                dod_trend = gr.Plot(label="DoS Trend")
                dod_ro = gr.Plot(label="Requirements Objective")

            def _refresh_dod():
                gauge, trend, ro, summary = build_dod_metrics()
                return gauge, trend, ro, summary

            dod_btn.click(_refresh_dod, outputs=[dod_gauge, dod_trend, dod_ro, dod_summary])
            app.load(_refresh_dod, outputs=[dod_gauge, dod_trend, dod_ro, dod_summary])

        # ── Macro Economy ────────────────────────────────────────────────────
        with gr.Tab("Macro Economy", id="macro"):
            macro_btn = gr.Button("Refresh", variant="primary", size="sm")
            macro_chart = gr.Plot(label="GSCPI")
            macro_summary = gr.Markdown("")

            def _refresh_macro():
                return build_macro()

            macro_btn.click(_refresh_macro, outputs=[macro_chart, macro_summary])
            app.load(_refresh_macro, outputs=[macro_chart, macro_summary])

        # ── Footer ───────────────────────────────────────────────────────────
        gr.Markdown(
            "<center style='color:#64748B; font-size:12px; padding-top:16px;'>"
            "Supply Chain Intelligence Platform v2 &mdash; "
            "Data sourced from USAspending, FPDS, SAM.gov, Federal Register, "
            "Yahoo Finance, Meteostat, World Bank, NY Fed, WTO"
            "</center>"
        )

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
    )
