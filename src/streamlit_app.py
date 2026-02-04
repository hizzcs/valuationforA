import os
import streamlit as st
import pandas as pd
import plotly.express as px

from data_pipeline import fetch_financials, fetch_daily_price, fetch_basic
from valuation import build_inputs, simple_dcf

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

with st.sidebar:
    st.header("核心参数")
    ts_code = st.text_input("股票代码", value="600519.SH")
    wacc = st.slider("WACC", 0.04, 0.14, 0.08, 0.005)
    terminal_growth = st.slider("终值增长率", 0.00, 0.05, 0.02, 0.002)
    growth_delta = st.slider("收入/现金流增速调整", -0.05, 0.05, 0.0, 0.005)

    st.caption("需要环境变量 TUSHARE_TOKEN")

# Data
try:
    basic = fetch_basic(ts_code)
    financials = fetch_financials(ts_code)
    price_df = fetch_daily_price(ts_code)
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

name = basic["name"].iloc[0] if basic is not None and not basic.empty else ts_code
st.subheader(f"{name} ({ts_code})")

# Valuation
inputs = build_inputs(financials, wacc, terminal_growth, growth_delta)
value = simple_dcf(inputs)

# KPI
c1, c2, c3 = st.columns(3)
c1.metric("DCF估值", f"{value:,.0f}")
if not price_df.empty:
    latest_price = float(price_df["close"].iloc[-1])
    c2.metric("最新价", f"{latest_price:,.2f}")
    mispricing = (value / latest_price - 1) if latest_price else 0
    c3.metric("估值偏离", f"{mispricing * 100:.2f}%")
else:
    c2.metric("最新价", "-")
    c3.metric("估值偏离", "-")

# Tabs
overview_tab, sensitivity_tab, data_tab = st.tabs(["概览", "敏感性", "数据"])

with overview_tab:
    st.subheader("关键指标")
    st.write({
        "base_revenue": inputs.base_revenue,
        "net_profit": inputs.net_profit,
        "cashflow": inputs.cashflow,
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
                temp_inputs = build_inputs(financials, w, g, growth_delta)
                val = simple_dcf(temp_inputs)
            grid.append({"WACC": w, "Terminal": g, "Value": val})
    df_grid = pd.DataFrame(grid)
    fig = px.imshow(df_grid.pivot(index="Terminal", columns="WACC", values="Value"), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

with data_tab:
    st.subheader("原始数据")
    st.write("收入/利润")
    st.dataframe(financials["income"].head(10))
    st.write("现金流")
    st.dataframe(financials["cashflow"].head(10))
    if not price_df.empty:
        st.write("价格")
        st.dataframe(price_df.tail(10))
