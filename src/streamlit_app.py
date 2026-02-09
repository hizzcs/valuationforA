from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.alerts import evaluate_alerts, persist_alerts  # noqa: E402
from src.data_pipeline import TuShareClient, ensure_duckdb_schema, load_inputs, load_macro_series, load_prices  # noqa: E402
from src.industry_valuation import estimate_industry_type, estimate_industry_valuation  # noqa: E402
from src.reporting import log_risk_profile, save_valuation  # noqa: E402
from src.risk_params import build_risk_profile  # noqa: E402
from src.scenario_engine import build_distribution_params, run_monte_carlo, tornado_analysis  # noqa: E402
from src.valuation_core import revenue_driven_dcf, roic_driven_dcf, two_stage_dcf  # noqa: E402
from src.workbench_domain import (  # noqa: E402
    METHOD_LABELS,
    UI_METHOD_MAPPING,
    assumption_frame,
    build_dashboard_snapshot,
    build_method_results,
    method_comparison_frame,
)


st.set_page_config(page_title="A股绝对估值 - 研究工作台", layout="wide")


def apply_theme() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

          html, body, [class*="css"] {
            font-family: 'Noto Sans SC', sans-serif;
          }

          .stApp {
            background:
              radial-gradient(1200px 480px at 8% -10%, rgba(34, 111, 195, 0.17), transparent 65%),
              radial-gradient(920px 360px at 95% -8%, rgba(245, 160, 66, 0.16), transparent 70%),
              linear-gradient(180deg, #f6f8fb 0%, #f2f6ff 100%);
          }

          div[data-testid="stMetric"] {
            border: 1px solid rgba(20, 32, 60, 0.08);
            border-radius: 12px;
            padding: 10px 14px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 8px 20px rgba(20, 32, 60, 0.06);
          }

          div[data-testid="stMetricLabel"] {
            font-weight: 700;
          }

          .mono-note {
            font-family: 'IBM Plex Mono', monospace;
            color: #2a3d64;
            font-size: 0.84rem;
          }

          .block-title {
            padding: 0.3rem 0;
            border-bottom: 2px solid rgba(46, 104, 191, 0.22);
            margin-bottom: 0.8rem;
            font-weight: 800;
            color: #1f2b47;
            letter-spacing: 0.01em;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_badge(status: str) -> str:
    if status == "pass":
        return "✅"
    if status == "warn":
        return "⚠️"
    return "❌"


def render_scenarios_table(scenarios: dict, scenarios_per_share: dict | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(list(scenarios.items()), columns=["情景", "企业价值(总额)"])
    if scenarios_per_share:
        frame["每股内在价值"] = frame["情景"].map(scenarios_per_share)
    frame["企业价值(总额)"] = frame["企业价值(总额)"].apply(lambda x: f"{x:,.0f}")
    if "每股内在价值" in frame.columns:
        frame["每股内在价值"] = frame["每股内在价值"].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
        )
    return frame


def render_tornado_chart(tornado: dict, title: str = "参数敏感性 (Tornado)") -> go.Figure | None:
    if not tornado:
        return None
    frame = pd.DataFrame(list(tornado.items()), columns=["参数", "影响"])
    frame = frame.dropna().sort_values("影响", ascending=True)
    if frame.empty:
        return None
    fig = px.bar(
        frame,
        y="参数",
        x="影响",
        orientation="h",
        color="影响",
        color_continuous_scale="Tealgrn",
        title=title,
    )
    fig.update_layout(height=360, margin=dict(l=24, r=24, t=55, b=24), coloraxis_showscale=False)
    return fig


def render_cash_flow_chart(cash_flows: list[float], discount_factors: list[float], title: str) -> go.Figure:
    years = list(range(1, len(cash_flows) + 1))
    frame = pd.DataFrame(
        {
            "年份": years,
            "自由现金流": cash_flows,
            "折现因子": discount_factors,
        }
    )
    frame["现值"] = frame["自由现金流"] * frame["折现因子"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["年份"],
            y=frame["自由现金流"],
            name="自由现金流",
            marker_color="#2f6db2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["年份"],
            y=frame["现值"],
            name="现值",
            mode="lines+markers",
            line=dict(color="#e07a22", width=3),
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        xaxis_title="预测年度",
        yaxis_title="金额",
        legend_title="序列",
        margin=dict(l=24, r=24, t=55, b=24),
    )
    return fig


def render_weekly_kline(prices: pd.DataFrame) -> go.Figure | None:
    if prices.empty:
        return None
    recent = prices.sort_values("trade_date").tail(22).copy()
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(recent.columns)):
        recent["open"] = recent["close"].shift(1).fillna(recent["close"])
        recent["high"] = recent[["open", "close"]].max(axis=1)
        recent["low"] = recent[["open", "close"]].min(axis=1)
    recent["ma5"] = recent["close"].rolling(5).mean()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=recent["trade_date"],
                open=recent["open"],
                high=recent["high"],
                low=recent["low"],
                close=recent["close"],
                increasing_line_color="#d43c31",
                decreasing_line_color="#198754",
                name="K线",
            )
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=recent["trade_date"],
            y=recent["ma5"],
            name="MA5",
            line=dict(color="#e07a22", width=2),
        )
    )
    fig.update_layout(
        title="近一个月价格行为（含 MA5）",
        xaxis_title="交易日",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=24, r=24, t=55, b=24),
    )
    return fig


def render_method_comparison_chart(frame: pd.DataFrame) -> go.Figure | None:
    if frame.empty:
        return None
    chart = px.bar(
        frame,
        x="method",
        y="enterprise_value",
        color="method",
        title="估值方法对比（企业价值）",
        text="enterprise_value",
        color_discrete_sequence=["#2f6db2", "#e07a22", "#4f8f63"],
    )
    chart.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    chart.update_layout(showlegend=False, height=360, margin=dict(l=24, r=24, t=55, b=24))
    return chart


def render_distribution_hist(samples: list[float], p5: float, p50: float, p95: float, reference: float) -> go.Figure:
    frame = pd.DataFrame({"估值样本": samples})
    fig = px.histogram(
        frame,
        x="估值样本",
        nbins=45,
        title="Monte Carlo 估值分布",
        color_discrete_sequence=["#3a74b6"],
        opacity=0.85,
    )
    for x, label, color in [
        (p5, "P5", "#c0392b"),
        (p50, "P50", "#1f6f8b"),
        (p95, "P95", "#2e7d32"),
        (reference, "选中模型", "#f39c12"),
    ]:
        fig.add_vline(x=x, line_width=2, line_dash="dash", line_color=color)
        fig.add_annotation(x=x, y=1.05, yref="paper", text=label, showarrow=False, font=dict(color=color))

    fig.update_layout(height=360, margin=dict(l=24, r=24, t=55, b=24), bargap=0.03)
    return fig


def render_value_band_chart(p5: float, p50: float, p95: float, reference: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p5, p95],
            y=["估值区间", "估值区间"],
            mode="lines",
            line=dict(color="#2f6db2", width=12),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p50],
            y=["估值区间"],
            mode="markers+text",
            marker=dict(color="#e07a22", size=14),
            text=["中位数"],
            textposition="top center",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[reference],
            y=["估值区间"],
            mode="markers+text",
            marker=dict(color="#198754", size=14, symbol="diamond"),
            text=["选中模型"],
            textposition="bottom center",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="估值区间定位（P5-P95）",
        xaxis_title="企业价值",
        yaxis_title="",
        height=220,
        margin=dict(l=24, r=24, t=55, b=24),
    )
    return fig


def render_margin_indicator(margin_of_safety: float | None) -> go.Figure | None:
    if margin_of_safety is None:
        return None
    value = margin_of_safety * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": "安全边际"},
            gauge={
                "axis": {"range": [-60, 120]},
                "bar": {"color": "#2f6db2"},
                "steps": [
                    {"range": [-60, 0], "color": "#f6d6d5"},
                    {"range": [0, 30], "color": "#e4f1ff"},
                    {"range": [30, 120], "color": "#e5f6ea"},
                ],
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=24, r=24, t=50, b=24))
    return fig


def render_risk_radar(trace: dict) -> go.Figure:
    metrics = {
        "Beta": min(max(float(trace.get("beta", 1.0)) / 2.5, 0.0), 1.0),
        "WACC": min(max(float(trace.get("cost_of_equity", 0.1)) / 0.25, 0.0), 1.0),
        "DebtCost": min(max(float(trace.get("cost_of_debt", 0.05)) / 0.12, 0.0), 1.0),
        "StdErr": min(max(float(trace.get("std_err", 0.02)) / 0.12, 0.0), 1.0),
        "R2": min(max(float(trace.get("r2", 0.5)), 0.0), 1.0),
    }
    theta = list(metrics.keys())
    values = list(metrics.values())
    theta.append(theta[0])
    values.append(values[0])

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                line=dict(color="#2f6db2", width=2),
                fillcolor="rgba(47, 109, 178, 0.28)",
            )
        ]
    )
    fig.update_layout(
        title="风险剖面雷达图（归一化）",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=360,
        margin=dict(l=24, r=24, t=55, b=24),
    )
    return fig


def render_sensitivity_lines(sensitivity: dict | None) -> None:
    if not sensitivity:
        return
    for param, points in sensitivity.items():
        frame = pd.DataFrame(points, columns=["参数值", "内在价值"])
        frame = frame.dropna()
        if frame.empty:
            continue
        fig = px.line(
            frame,
            x="参数值",
            y="内在价值",
            title=f"{param} 敏感性曲线",
            markers=True,
            color_discrete_sequence=["#e07a22"],
        )
        fig.update_layout(height=300, margin=dict(l=24, r=24, t=55, b=24))
        st.plotly_chart(fig, use_container_width=True)


def render_method_guide() -> None:
    st.subheader("方法说明与适用场景")
    st.caption("系统按确定性公式计算，不使用黑箱预测。")

    with st.expander("收入驱动 DCF（Revenue-driven FCFF）", expanded=False):
        st.markdown(
            r"""
            - \(Revenue_t = Revenue_{t-1}\times(1+g_t)\)
            - \(NOPAT_t = Revenue_t \times Operating\ Margin\)
            - \(FCFF_t = NOPAT_t - Reinvestment_t\)
            - \(EV = \sum_{t=1}^{n}\frac{FCFF_t}{(1+WACC)^t} + \frac{TV}{(1+WACC)^n}\)
            - 每股价值：\(\frac{EV-Net\ Debt}{Shares}\)
            """
        )

    with st.expander("ROIC 驱动 DCF", expanded=False):
        st.markdown(
            r"""
            - \(NOPAT_t = Invested\ Capital_t \times ROIC_t\)
            - \(FCFF_t = NOPAT_t - Reinvestment_t\)
            - 适合评估资本效率驱动型公司。
            """
        )

    with st.expander("两阶段 DCF", expanded=False):
        st.markdown(
            r"""
            - 高增速阶段 + 稳态阶段
            - 统一折现并计算终值
            - 适合“高增向稳态”过渡公司
            """
        )

    with st.expander("行业特定估值", expanded=False):
        st.markdown(
            """
            - 按行业属性修正 β/WACC/增长参数
            - 输出行业化情景与敏感性
            """
        )


apply_theme()
st.title("A股绝对估值研究工作台")
st.caption("将领域估值、风险画像和可视化叙事统一在同一工作流中。")
ensure_duckdb_schema()
client = TuShareClient()


def main() -> None:
    sidebar = st.sidebar
    ticker = sidebar.text_input("TuShare TS 代码", value="600000.SH")
    as_of = sidebar.date_input("估值日期", value=date(2024, 12, 31))
    valuation_method = sidebar.selectbox(
        "主视图方法",
        ["收入驱动DCF", "ROIC驱动DCF", "两阶段DCF", "行业特定估值"],
    )
    scenario_draws = sidebar.slider("Monte Carlo 路径数", min_value=1000, max_value=10000, step=1000, value=5000)
    run_button = sidebar.button("执行估值")

    if not run_button:
        st.info("在左侧输入参数并点击“执行估值”，系统将生成领域快照、风险图谱和估值分布。")
        render_method_guide()
        return

    with st.spinner("加载与计算中..."):
        try:
            validated = load_inputs(client, ticker, as_of)
        except ValueError as exc:
            st.error(
                f"{exc}\n\n建议：将估值日期调整到最近财报披露日之后，或确认 TuShare 财报接口可用。"
            )
            return
        run_ticker = validated.ticker
        if run_ticker.upper() != ticker.upper():
            st.info(f"输入代码 {ticker.upper()} 无数据，已自动切换到 {run_ticker} 继续估值。")
        prices = load_prices(client, ticker, as_of=as_of)
        market_prices = load_prices(client, ticker, as_of=None)
        if prices.empty and run_ticker.upper() != ticker.upper():
            prices = load_prices(client, run_ticker, as_of=as_of)
        if market_prices.empty and run_ticker.upper() != ticker.upper():
            market_prices = load_prices(client, run_ticker, as_of=None)

        csi300 = client.call_api("csi300")
        csi300["trade_date"] = pd.to_datetime(csi300["trade_date"])
        csi300 = csi300[csi300["trade_date"] <= pd.Timestamp(as_of)]

        bonds = client.call_api("bonds")
        bonds["obs_date"] = pd.to_datetime(bonds["obs_date"])
        bonds = bonds[bonds["obs_date"] <= pd.Timestamp(as_of)]

        macro = load_macro_series(client, as_of=as_of)
        if prices.empty or csi300.empty:
            st.error("截止估值日缺少价格数据，无法估算 Beta。请调整估值日期或补充数据。")
            return

        risk = build_risk_profile(run_ticker, as_of, prices, csi300, bonds, validated.statements)
        method_results = build_method_results(validated, risk)

        industry_val = None
        if valuation_method == "行业特定估值":
            industry_val = estimate_industry_valuation(run_ticker, as_of, validated, risk)
            st.session_state["industry_val"] = industry_val
        else:
            st.session_state.pop("industry_val", None)

        selected_method_key = UI_METHOD_MAPPING.get(valuation_method, "two_stage")
        selected_result = method_results.get(selected_method_key, method_results["revenue"])

        params = build_distribution_params(prices, macro, selected_result)
        mc = run_monte_carlo(selected_result, params, draws=scenario_draws)

        tornado_config = {
            "revenue": (revenue_driven_dcf, ["revenue_growth", "operating_margin", "wacc"]),
            "roic": (roic_driven_dcf, ["roic", "reinvestment_rate", "wacc"]),
            "two_stage": (two_stage_dcf, ["high_growth_rate", "stable_growth_rate", "wacc"]),
        }
        tornado_func, tornado_params = tornado_config.get(selected_method_key, tornado_config["revenue"])
        tornado = tornado_analysis(selected_result, validated, risk, tornado_func, tornado_params)

    latest_price = None
    latest_date = None
    if not market_prices.empty:
        latest_row = market_prices.sort_values("trade_date").iloc[-1]
        latest_price = float(latest_row["close"])
        latest_date = pd.Timestamp(latest_row["trade_date"]).date()

    snapshot = build_dashboard_snapshot(
        ticker=run_ticker,
        as_of_date=as_of,
        selected_method_label=valuation_method if valuation_method in UI_METHOD_MAPPING else "两阶段DCF",
        validated=validated,
        risk=risk,
        results=method_results,
        scenario_summary=mc,
        market_price=latest_price,
    )
    method_frame = method_comparison_frame(snapshot.methods)
    selected_snapshot = snapshot.methods[snapshot.selected_method_key]
    industry_type = estimate_industry_type(ticker, validated)

    st.markdown('<div class="block-title">估值总览</div>', unsafe_allow_html=True)
    metric_cols = st.columns(6)
    metric_cols[0].metric("数据质量", snapshot.data_quality_grade)
    metric_cols[1].metric("数据模式", snapshot.source_mode)
    metric_cols[2].metric("行业类型", industry_type.value)
    metric_cols[3].metric("选中模型", METHOD_LABELS.get(snapshot.selected_method_key, snapshot.selected_method_key))
    metric_cols[4].metric(
        "最新市场价",
        f"{latest_price:,.2f}" if latest_price is not None else "N/A",
        help=f"数据日: {latest_date.isoformat()}" if latest_date else "",
    )
    metric_cols[5].metric(
        "选中模型每股",
        f"{selected_snapshot.intrinsic_value_per_share:,.2f}"
        if selected_snapshot.intrinsic_value_per_share is not None
        else "N/A",
    )

    tabs = st.tabs(["总览", "模型拆解", "风险可视化", "方法说明", "数据与落库"])

    with tabs[0]:
        top_left, top_right = st.columns([1.6, 1.4])

        with top_left:
            comp_fig = render_method_comparison_chart(method_frame)
            if comp_fig:
                st.plotly_chart(comp_fig, use_container_width=True)

        with top_right:
            band_fig = render_value_band_chart(
                snapshot.scenario_band.p5,
                snapshot.scenario_band.p50,
                snapshot.scenario_band.p95,
                selected_snapshot.enterprise_value,
            )
            st.plotly_chart(band_fig, use_container_width=True)

        lower_left, lower_right = st.columns([1, 2])
        with lower_left:
            margin_fig = render_margin_indicator(snapshot.margin_of_safety)
            if margin_fig is None:
                st.info("安全边际需同时具备市场价格和每股内在价值。")
            else:
                st.plotly_chart(margin_fig, use_container_width=True)
                st.markdown(
                    f'<div class="mono-note">margin_of_safety = intrinsic_per_share / market_price - 1</div>',
                    unsafe_allow_html=True,
                )

        with lower_right:
            if market_prices.empty:
                st.warning("无可用价格数据，无法展示K线。")
            else:
                kline_fig = render_weekly_kline(market_prices)
                if kline_fig:
                    st.plotly_chart(kline_fig, use_container_width=True)

        st.markdown("#### 质量校验")
        q_cols = st.columns(max(len(snapshot.verification), 1))
        for idx, (metric, status) in enumerate(snapshot.verification.items()):
            q_cols[idx % len(q_cols)].metric(metric, render_badge(status))

    with tabs[1]:
        if industry_val is not None:
            st.markdown("#### 行业特定估值输出")
            c1, c2 = st.columns(2)
            c1.metric("行业基础情景(总额)", f"{industry_val.base_value:,.0f}")
            c2.metric(
                "行业基础情景(每股)",
                f"{industry_val.base_value_per_share:,.2f}"
                if industry_val.base_value_per_share is not None
                else "N/A",
            )
            st.dataframe(render_scenarios_table(industry_val.scenarios, industry_val.scenarios_per_share), use_container_width=True)
            tornado_fig = render_tornado_chart(industry_val.tornado, title="行业参数敏感性 (Tornado)")
            if tornado_fig:
                st.plotly_chart(tornado_fig, use_container_width=True)
            render_sensitivity_lines(industry_val.sensitivity)

        st.markdown("#### 基准方法对比")
        if not method_frame.empty:
            display = method_frame.copy()
            display["enterprise_value"] = display["enterprise_value"].map(lambda x: f"{x:,.0f}")
            display["equity_value"] = display["equity_value"].map(lambda x: f"{x:,.0f}")
            display["intrinsic_value_per_share"] = display["intrinsic_value_per_share"].map(
                lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            )
            display = display.rename(
                columns={
                    "method": "估值方法",
                    "enterprise_value": "企业价值(总额)",
                    "equity_value": "股权价值(总额)",
                    "intrinsic_value_per_share": "每股内在价值",
                }
            )
            st.dataframe(display[["估值方法", "企业价值(总额)", "股权价值(总额)", "每股内在价值"]], use_container_width=True)

        st.markdown("#### 选中模型现金流拆解")
        st.plotly_chart(
            render_cash_flow_chart(
                selected_snapshot.cash_flows,
                selected_snapshot.discount_factors,
                title=f"{METHOD_LABELS.get(snapshot.selected_method_key, snapshot.selected_method_key)} 现金流轨迹",
            ),
            use_container_width=True,
        )

        st.markdown("#### 选中模型假设")
        assumption_df = assumption_frame(selected_snapshot)
        if assumption_df.empty:
            st.info("无可展示参数。")
        else:
            st.dataframe(assumption_df, use_container_width=True)

    with tabs[2]:
        risk_left, risk_right = st.columns(2)
        with risk_left:
            st.plotly_chart(render_risk_radar(snapshot.risk.trace), use_container_width=True)
        with risk_right:
            st.plotly_chart(
                render_distribution_hist(
                    mc.samples,
                    snapshot.scenario_band.p5,
                    snapshot.scenario_band.p50,
                    snapshot.scenario_band.p95,
                    selected_snapshot.enterprise_value,
                ),
                use_container_width=True,
            )

        risk_metrics = st.columns(4)
        risk_metrics[0].metric("P5", f"{snapshot.scenario_band.p5:,.0f}")
        risk_metrics[1].metric("P50", f"{snapshot.scenario_band.p50:,.0f}")
        risk_metrics[2].metric("P95", f"{snapshot.scenario_band.p95:,.0f}")
        risk_metrics[3].metric("WACC", f"{snapshot.risk.wacc:.2%}")

        tornado_fig = render_tornado_chart(tornado)
        if tornado_fig:
            st.plotly_chart(tornado_fig, use_container_width=True)
            st.caption("龙卷风图用于识别单因子变化下最敏感的估值驱动变量。")

        with st.expander("风险参数追踪", expanded=False):
            st.json(snapshot.risk.trace)

    with tabs[3]:
        render_method_guide()

    with tabs[4]:
        st.subheader("数据来源透明度")
        st.table(
            pd.DataFrame(
                [
                    {
                        "metric": "financials",
                        "source": validated.metadata.get("source", "fixture"),
                        "grade": validated.data_quality_grade,
                    },
                    {
                        "metric": "prices",
                        "source": "tushare" if client._pro else "fixture",
                        "grade": "-",
                    },
                    {
                        "metric": "macro",
                        "source": "tushare" if client._pro else "fixture",
                        "grade": "-",
                    },
                ]
            )
        )

        if st.button("保存到 DuckDB"):
            try:
                save_valuation(selected_result, mc, risk)
                log_risk_profile(risk)
                alerts = evaluate_alerts(selected_result, mc)
                persist_alerts(alerts)
                if alerts:
                    st.warning(f"触发 {len(alerts)} 条告警，请检查 DuckDB 的 valuation_alerts。")
                else:
                    st.success("已保存估值记录并完成告警检查。")
            except RuntimeError as exc:
                st.error(f"无法保存：{exc}")


if __name__ == "__main__":
    main()
