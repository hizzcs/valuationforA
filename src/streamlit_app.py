import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import (
    fetch_financials,
    fetch_daily_price,
    fetch_basic,
    fetch_daily_basic,
    fetch_index_weight,
    fetch_all_stock_basic,
)
from valuation import (
    build_inputs,
    simple_dcf,
    calc_wacc,
    equity_value_per_share,
    filter_non_financial,
    rank_stocks_by_mispricing,
)
from reporting import generate_report

st.set_page_config(page_title="valuationforA", layout="wide")

st.markdown(
    """
<style>
h1, h2, h3 { font-weight: 700; }
.block-container { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("A股估值工作台 (Tushare Only)")

# Presets
presets = {
    "保守": {"wacc": 0.10, "terminal_growth": 0.01, "growth_delta": -0.02},
    "基准": {"wacc": 0.08, "terminal_growth": 0.02, "growth_delta": 0.0},
    "激进": {"wacc": 0.06, "terminal_growth": 0.03, "growth_delta": 0.02},
}

with st.sidebar:
    st.header("核心参数")
    ts_code = st.text_input("股票代码", value="600519.SH")

    preset = st.selectbox("参数模板", list(presets.keys()))
    preset_vals = presets[preset]

    wacc = st.slider("WACC", 0.04, 0.14, preset_vals["wacc"], 0.005)
    terminal_growth = st.slider("终值增长率", 0.00, 0.05, preset_vals["terminal_growth"], 0.002)
    growth_delta = st.slider("收入/现金流增速调整", -0.05, 0.05, preset_vals["growth_delta"], 0.005)

    with st.expander("WACC 细化 (可选)"):
        risk_free = st.slider("无风险利率", 0.01, 0.06, 0.03, 0.001)
        beta = st.slider("Beta", 0.5, 2.0, 1.0, 0.05)
        erp = st.slider("股权风险溢价", 0.03, 0.08, 0.05, 0.005)
        cost_of_debt = st.slider("税前债务成本", 0.02, 0.08, 0.04, 0.002)
        tax_rate = st.slider("税率", 0.10, 0.30, 0.20, 0.01)
        debt_ratio = st.slider("债务占比", 0.0, 0.6, 0.2, 0.05)
        if st.button("使用WACC推导值"):
            wacc = calc_wacc(risk_free, beta, erp, cost_of_debt, tax_rate, debt_ratio)
            st.success(f"WACC 计算结果：{wacc:.3f}")

    st.caption("需要环境变量 TUSHARE_TOKEN")

# Data
try:
    basic = fetch_basic(ts_code)
    financials = fetch_financials(ts_code)
    price_df = fetch_daily_price(ts_code)
    daily_basic = fetch_daily_basic(ts_code)
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

name = basic["name"].iloc[0] if basic is not None and not basic.empty else ts_code
st.subheader(f"{name} ({ts_code})")

# Valuation inputs
shares_outstanding = 1.0
net_debt = 0.0
if daily_basic is not None and not daily_basic.empty:
    latest_mv = float(daily_basic["total_mv"].iloc[-1])
    latest_price_for_mv = float(price_df["close"].iloc[-1]) if not price_df.empty else None
    if latest_price_for_mv and latest_price_for_mv > 0:
        shares_outstanding = (latest_mv * 1e4) / latest_price_for_mv  # total_mv in 10k CNY

# TODO (needs human input): net debt from balance sheet / latest report
inputs = build_inputs(financials, wacc, terminal_growth, growth_delta, shares_outstanding, net_debt)
enterprise_value = simple_dcf(inputs)
value = equity_value_per_share(enterprise_value, net_debt, shares_outstanding)

# KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("DCF估值", f"{value:,.0f}")
if not price_df.empty:
    latest_price = float(price_df["close"].iloc[-1])
    c2.metric("最新价", f"{latest_price:,.2f}")
    mispricing = (value / latest_price - 1) if latest_price else 0
    c3.metric("估值偏离", f"{mispricing * 100:.2f}%")
    c4.metric("安全边际", f"{(1-mispricing)*100:.1f}%" if latest_price else "-")
else:
    c2.metric("最新价", "-")
    c3.metric("估值偏离", "-")
    c4.metric("安全边际", "-")

# Tabs
overview_tab, sensitivity_tab, chart_tab, data_tab, ranking_tab = st.tabs(["概览", "敏感性", "图表", "数据", "推荐"])

with overview_tab:
    st.subheader("关键指标")
    st.write({
        "base_revenue": inputs.base_revenue,
        "net_profit": inputs.net_profit,
        "cashflow": inputs.cashflow,
        "shares_outstanding": f"{inputs.shares_outstanding:,.0f}",
        "net_debt": inputs.net_debt,
    })

with sensitivity_tab:
    st.subheader("敏感性分析")
    wacc_values = [wacc - 0.01, wacc, wacc + 0.01]
    g_values = [terminal_growth - 0.01, terminal_growth, terminal_growth + 0.01]
    grid = []
    for w in wacc_values:
        for g in g_values:
            if w <= g:
                val = None
            else:
                temp_inputs = build_inputs(financials, w, g, growth_delta, shares_outstanding, net_debt)
                ev = simple_dcf(temp_inputs)
                val = equity_value_per_share(ev, net_debt, shares_outstanding)
            grid.append({"WACC": w, "Terminal": g, "Value": val})
    df_grid = pd.DataFrame(grid)
    fig = px.imshow(df_grid.pivot(index="Terminal", columns="WACC", values="Value"), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

with chart_tab:
    st.subheader("价格走势")
    if not price_df.empty:
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=price_df["trade_date"], y=price_df["close"], name="Close"))
        figp.update_layout(title="历史价格", yaxis_title="Price")
        st.plotly_chart(figp, use_container_width=True)

with data_tab:
    st.subheader("原始数据")
    st.write("收入/利润")
    st.dataframe(financials["income"].head(10))
    st.write("现金流")
    st.dataframe(financials["cashflow"].head(10))
    if not price_df.empty:
        st.write("价格")
        st.dataframe(price_df.tail(10))


@st.cache_data(ttl=3600, show_spinner=False)
def _quick_valuation(ts_code: str, wacc: float, terminal_growth: float, growth_delta: float):
    financials = fetch_financials(ts_code)
    price_df = fetch_daily_price(ts_code)
    daily_basic = fetch_daily_basic(ts_code)
    if price_df.empty or daily_basic is None or daily_basic.empty:
        return None

    latest_mv = float(daily_basic["total_mv"].iloc[-1])
    latest_price_for_mv = float(price_df["close"].iloc[-1])
    shares_outstanding = (latest_mv * 1e4) / latest_price_for_mv if latest_price_for_mv > 0 else 1.0
    net_debt = 0.0

    inputs = build_inputs(financials, wacc, terminal_growth, growth_delta, shares_outstanding, net_debt)
    enterprise_value = simple_dcf(inputs)
    per_share = equity_value_per_share(enterprise_value, net_debt, shares_outstanding)
    price = float(price_df["close"].iloc[-1])
    mispricing = (per_share / price - 1) if price > 0 else None

    return {
        "per_share_value": per_share,
        "price": price,
        "mispricing": mispricing,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _build_universe(index_code: str, limit: int, exclude_financial: bool):
    weights = fetch_index_weight(index_code, limit=limit)
    basic_all = fetch_all_stock_basic()
    merged = weights.merge(basic_all, left_on="con_code", right_on="ts_code", how="left")
    if exclude_financial:
        merged = filter_non_financial(merged)
    return merged


with ranking_tab:
    st.subheader("价值推荐（粗筛）")
    st.caption("当前为简化 DCF + 统一参数，适合初筛；非最终投资结论。")

    index_map = {
        "上证50": "000016.SH",
        "沪深300": "000300.SH",
        "中证500": "000905.SH",
    }
    idx_name = st.selectbox("指数池", list(index_map.keys()), index=1)
    limit = st.slider("取前N权重成分", 20, 200, 50, 10)
    top_n = st.slider("输出TopN", 5, 50, 15, 5)
    exclude_fin = st.checkbox("剔除金融", value=True)

    if st.button("开始筛选"):
        universe = _build_universe(index_map[idx_name], limit, exclude_fin)
        results = []
        progress = st.progress(0.0)
        for i, row in universe.iterrows():
            ts_code = row["con_code"]
            val = _quick_valuation(ts_code, wacc, terminal_growth, growth_delta)
            if val:
                results.append({
                    "ts_code": ts_code,
                    "name": row.get("name", ""),
                    "industry": row.get("industry", ""),
                    "weight": row.get("weight", None),
                    **val,
                })
            progress.progress((i + 1) / max(1, len(universe)))
        res_df = pd.DataFrame(results)
        ranked = rank_stocks_by_mispricing(res_df, top_n=top_n)
        st.dataframe(ranked, use_container_width=True)

# Report
assumptions = {
    "WACC": wacc,
    "terminal_growth": terminal_growth,
    "growth_delta": growth_delta,
}
inputs_payload = {
    "base_revenue": inputs.base_revenue,
    "net_profit": inputs.net_profit,
    "cashflow": inputs.cashflow,
    "shares_outstanding": inputs.shares_outstanding,
    "net_debt": inputs.net_debt,
}
report_payload = {
    "ts_code": ts_code,
    "name": name,
    "valuation": value,
    "mispricing": mispricing if not price_df.empty else None,
    "inputs": inputs_payload,
    "assumptions": assumptions,
}
report_md = generate_report(report_payload)

st.download_button("下载估值报告 (Markdown)", report_md, file_name=f"{ts_code}_valuation.md")
