from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_pipeline import (  # noqa: E402
    TuShareClient,
    ensure_duckdb_schema,
    load_inputs,
    load_macro_series,
    load_prices,
)
from src.risk_params import build_risk_profile  # noqa: E402
from src.scenario_engine import build_distribution_params, run_monte_carlo, tornado_analysis  # noqa: E402
from src.reporting import log_risk_profile, save_valuation  # noqa: E402
from src.valuation_core import revenue_driven_dcf, roic_driven_dcf, two_stage_dcf  # noqa: E402
from src.industry_valuation import estimate_industry_type, estimate_industry_valuation  # noqa: E402
from src.alerts import evaluate_alerts, persist_alerts  # noqa: E402


st.set_page_config(page_title="A股绝对估值 - 投资经理工作台", layout="wide")
st.title("📊 A股绝对估值 - 投资经理工作台")
ensure_duckdb_schema()

client = TuShareClient()

def render_badge(status: str) -> str:
    if status == "pass":
        return "✅"
    if status == "warn":
        return "⚠️"
    return "❌"


def render_scenarios_table(scenarios: dict, scenarios_per_share: dict | None = None):
    df = pd.DataFrame(list(scenarios.items()), columns=['情景', '企业价值(总额)'])
    if scenarios_per_share:
        df['每股内在价值'] = df['情景'].map(scenarios_per_share)
    df['企业价值(总额)'] = df['企业价值(总额)'].apply(lambda x: f"{x:,.0f}")
    if '每股内在价值' in df.columns:
        df['每股内在价值'] = df['每股内在价值'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
    return df

def render_tornado_chart(tornado: dict):
    if not tornado:
        return None
    df = pd.DataFrame(list(tornado.items()), columns=['参数', '影响'])
    df = df.sort_values('影响', ascending=True)
    fig = px.bar(df, y='参数', x='影响', orientation='h', 
                title='参数影响敏感性分析 (Tornado图)',
                color='影响',
                color_continuous_scale='RdBu')
    fig.update_layout(height=400)
    return fig

def render_cash_flow_chart(cash_flows, discount_factors):
    years = list(range(1, len(cash_flows)+1))
    df = pd.DataFrame({
        '年份': years,
        '自由现金流(FCFF)': cash_flows,
        '折现因子(DF)': discount_factors
    })
    df['现值(FCFF*DF)'] = df['自由现金流(FCFF)'] * df['折现因子(DF)']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['年份'],
        y=df['自由现金流(FCFF)'],
        name='自由现金流',
        marker_color='blue'
    ))
    fig.add_trace(go.Scatter(
        x=df['年份'],
        y=df['现值(FCFF*DF)'],
        name='现值',
        mode='lines+markers',
        yaxis='y2',
        marker_color='red'
    ))
    fig.add_trace(go.Scatter(
        x=df['年份'],
        y=df['折现因子(DF)'],
        name='折现因子',
        mode='lines+markers',
        yaxis='y3',
        marker_color='green'
    ))
    
    fig.update_layout(
        title='自由现金流与现值分析',
        yaxis=dict(title='自由现金流', side='left'),
        yaxis2=dict(title='现值', overlaying='y', side='right'),
        yaxis3=dict(title='折现因子', overlaying='y', side='right', position=0.95),
        height=500
    )
    
    return fig


def render_weekly_kline(prices: pd.DataFrame):
    if prices.empty:
        return None
    recent = prices.sort_values("trade_date").tail(7).copy()
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(recent.columns)):
        recent["open"] = recent["close"].shift(1).fillna(recent["close"])
        recent["high"] = recent[["open", "close"]].max(axis=1)
        recent["low"] = recent[["open", "close"]].min(axis=1)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=recent["trade_date"],
                open=recent["open"],
                high=recent["high"],
                low=recent["low"],
                close=recent["close"],
                increasing_line_color="#d32f2f",
                decreasing_line_color="#2e7d32",
                name="K线",
            )
        ]
    )
    fig.update_layout(
        title="近1周K线（最新可得交易日）",
        xaxis_title="交易日",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
        height=420,
    )
    return fig


def render_method_guide() -> None:
    st.subheader("方法说明与适用场景")
    st.caption("以下说明用于解释模型输入与输出，系统按确定性公式计算，不使用黑箱预测。")

    with st.expander("收入驱动 DCF（Revenue-driven FCFF）", expanded=False):
        st.markdown(
            r"""
            **核心公式**
            - \(Revenue_t = Revenue_{t-1}\times(1+g_t)\)
            - \(NOPAT_t = Revenue_t \times Operating\ Margin\)
            - \(FCFF_t = NOPAT_t - Reinvestment_t\)
            - \(EV = \sum_{t=1}^{n}\frac{FCFF_t}{(1+WACC)^t} + \frac{TV}{(1+WACC)^n}\)
            - \(TV_{perpetual}=\frac{FCFF_n(1+g)}{WACC-g}\)
            - \(Equity\ Value\ Per\ Share = \frac{EV - Net\ Debt}{Shares}\)

            **适用场景**
            - 收入和利润率相对可预测的成熟行业（消费、制造、公用事业等）。
            - 管理层提供了中短期收入目标，便于逐年拆解。
            """
        )

    with st.expander("ROIC 驱动 DCF", expanded=False):
        st.markdown(
            r"""
            **核心逻辑**
            - 先预测投入资本 \(Invested\ Capital_t\) 与回报率 \(ROIC_t\)。
            - \(NOPAT_t = Invested\ Capital_t \times ROIC_t\)
            - 再由 \(FCFF_t = NOPAT_t - Reinvestment_t\) 折现得到企业价值。

            **适用场景**
            - 资本开支和周转是核心驱动（重资产、周期行业）。
            - 希望把“增长质量（ROIC-WACC）”单独拉出来评估时。
            """
        )

    with st.expander("两阶段 DCF", expanded=False):
        st.markdown(
            r"""
            **核心逻辑**
            - 高增长阶段：使用较高增长率；
            - 稳定阶段：向长期稳态增长率收敛；
            - 对两阶段现金流与终值统一折现。

            **适用场景**
            - 公司处于“高增速向稳态过渡”区间（科技成长、行业拐点）。
            """
        )

    with st.expander("行业特定估值", expanded=False):
        st.markdown(
            """
            **核心逻辑**
            - 先识别行业属性（金融/消费/周期/科技等）；
            - 对 β、WACC、增长阶段与情景参数做行业化修正；
            - 输出基础情景、乐观/悲观情景与敏感性分析。

            **适用场景**
            - 单一通用参数难以刻画行业差异时（如金融监管、周期波动、研发驱动）。
            """
        )

    st.caption("参考资料：Damodaran《Investment Valuation》、Koller等《Valuation: Measuring and Managing the Value of Companies》、Brealey等《Principles of Corporate Finance》。")

def main() -> None:
    sidebar = st.sidebar
    ticker = sidebar.text_input("TuShare TS 代码", value="600000.SH")
    as_of = sidebar.date_input("估值日期", value=date(2024, 12, 31))
    
    # 估值方法选择
    valuation_method = sidebar.selectbox(
        "估值方法",
        ["收入驱动DCF", "ROIC驱动DCF", "两阶段DCF", "行业特定估值"]
    )
    
    scenario_draws = sidebar.slider("模拟路径数", min_value=1000, max_value=10000, step=1000, value=5000)
    run_button = sidebar.button("执行估值")

    if not run_button:
        st.info("在侧边栏输入参数并点击“执行估值”。")
        return

    with st.spinner("加载数据..."):
        validated = load_inputs(client, ticker, as_of)
        prices = load_prices(client, ticker, as_of=as_of)
        market_prices = load_prices(client, ticker, as_of=None)
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
        risk = build_risk_profile(ticker, as_of, prices, csi300, bonds, validated.statements)
        
        # 根据选择的方法执行估值
        if valuation_method == "行业特定估值":
            industry_val_result = estimate_industry_valuation(
                ticker, as_of, validated, risk
            )
            st.session_state['industry_val'] = industry_val_result
        else:
            st.session_state.pop('industry_val', None)
        
        # 基础估值
        base_result = revenue_driven_dcf(validated, risk)
        roic_result = roic_driven_dcf(validated, risk)
        two_stage_result = two_stage_dcf(validated, risk)
        
        params = build_distribution_params(prices, macro, base_result)
        mc = run_monte_carlo(base_result, params, draws=scenario_draws)
        
        # 龙卷风分析
        tornado = tornado_analysis(base_result, validated, risk, revenue_driven_dcf, 
                                  ["revenue_growth", "operating_margin", "wacc"])

    st.subheader("数据校验")
    grade_col, source_col, industry_col, shares_col = st.columns(4)
    grade_col.metric("数据质量", validated.data_quality_grade)
    source_col.metric("数据模式", validated.metadata.get("source_mode", "fixture"))
    industry_type = estimate_industry_type(ticker, validated)
    industry_col.metric("行业类型", industry_type.value)
    shares_col.metric("总股本(股)", f"{validated.shares_outstanding:,.0f}" if validated.shares_outstanding > 0 else "N/A")
    
    cols = st.columns(max(len(validated.verification), 1))
    for idx, (metric, status) in enumerate(validated.verification.items()):
        cols[idx % len(cols)].metric(metric, render_badge(status))

    st.subheader("风险画像")
    st.json(risk.trace)

    st.subheader("市场价格")
    if market_prices.empty:
        st.warning("无可用价格数据，无法展示K线。")
    else:
        latest_row = market_prices.sort_values("trade_date").iloc[-1]
        latest_price = float(latest_row["close"])
        latest_date = pd.Timestamp(latest_row["trade_date"]).date()
        today = datetime.now().date()
        if latest_date == today:
            st.metric("当日最新价", f"{latest_price:,.2f}")
        else:
            st.metric(f"最近收盘价（{latest_date.isoformat()}）", f"{latest_price:,.2f}")

        kline_fig = render_weekly_kline(market_prices)
        if kline_fig:
            st.plotly_chart(kline_fig)

    st.subheader("估值结果")
    if 'industry_val' in st.session_state:
        industry_val = st.session_state['industry_val']
        st.markdown("### 🎯 行业特定估值")
        st.write(f"内在价值: {industry_val.base_value:,.0f}")
        st.write(
            f"每股内在价值: {industry_val.base_value_per_share:,.2f}"
            if industry_val.base_value_per_share is not None
            else "每股内在价值: N/A（缺少可靠总股本）"
        )
        
        st.markdown("#### 情景分析")
        st.table(render_scenarios_table(industry_val.scenarios, industry_val.scenarios_per_share))
        
        if industry_val.tornado:
            st.markdown("#### 参数影响敏感性分析")
            st.plotly_chart(render_tornado_chart(industry_val.tornado))
            
        if industry_val.sensitivity:
            st.markdown("#### 敏感性分析")
            for param, values in industry_val.sensitivity.items():
                df = pd.DataFrame(values, columns=['参数值', '内在价值'])
                fig = px.line(df, x='参数值', y='内在价值', title=f"{param} 敏感性分析")
                st.plotly_chart(fig)
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "收入驱动DCF（每股）",
            f"{base_result.intrinsic_value_per_share:,.2f}" if base_result.intrinsic_value_per_share is not None else "N/A",
            help="公式：每股内在价值=(企业价值-净负债)/总股本",
        )
        col2.metric(
            "ROIC驱动DCF（每股）",
            f"{roic_result.intrinsic_value_per_share:,.2f}" if roic_result.intrinsic_value_per_share is not None else "N/A",
            help="公式：每股内在价值=(企业价值-净负债)/总股本",
        )
        col3.metric(
            "两阶段DCF（每股）",
            f"{two_stage_result.intrinsic_value_per_share:,.2f}" if two_stage_result.intrinsic_value_per_share is not None else "N/A",
            help="公式：每股内在价值=(企业价值-净负债)/总股本",
        )
        st.caption(
            f"企业价值（总额）：收入驱动 {base_result.intrinsic_value:,.0f}，ROIC驱动 {roic_result.intrinsic_value:,.0f}，两阶段 {two_stage_result.intrinsic_value:,.0f}"
        )
        
        st.markdown("#### 自由现金流分析 (收入驱动)")
        st.plotly_chart(render_cash_flow_chart(base_result.cash_flows, base_result.discount_factors))
        
        st.markdown("#### 两阶段DCF参数")
        st.write(two_stage_result.assumptions)

    st.subheader("Monte Carlo 风险分析")
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    mc_col1.metric("5%分位", f"{mc.percentile_5:,.0f}")
    mc_col2.metric("50%分位 (中位数)", f"{mc.percentile_50:,.0f}")
    mc_col3.metric("95%分位", f"{mc.percentile_95:,.0f}")
    
    st.markdown("#### 估值分布")
    fig = px.histogram(pd.DataFrame({'内在价值': mc.samples}), 
                      x='内在价值', nbins=50,
                      title='Monte Carlo 内在价值分布',
                      labels={'内在价值': '内在价值', 'count': '频率'})
    st.plotly_chart(fig)

    st.subheader("参数敏感性分析")
    st.plotly_chart(render_tornado_chart(tornado))
    st.caption("龙卷风图展示“单一参数变化”对估值的边际影响，适合识别最敏感输入。")

    render_method_guide()

    st.subheader("数据来源透明度")
    st.table(
        pd.DataFrame(
            [
                {"metric": "financials", "source": validated.metadata["source"], "grade": validated.data_quality_grade},
                {"metric": "prices", "source": "tushare" if client._pro else "fixture", "grade": "-"},
                {"metric": "macro", "source": "tushare" if client._pro else "fixture", "grade": "-"},
            ]
        )
    )

    if st.button("保存到 DuckDB"):
        try:
            save_valuation(base_result, mc, risk)
            log_risk_profile(risk)
            alerts = evaluate_alerts(base_result, mc)
            persist_alerts(alerts)
            if alerts:
                st.warning(f"触发 {len(alerts)} 条告警，请检查 DuckDB 中的 valuation_alerts。")
            else:
                st.success("已保存估值记录并检测告警。")
        except RuntimeError as exc:
            st.error(f"无法保存：{exc}")


if __name__ == "__main__":
    main()
