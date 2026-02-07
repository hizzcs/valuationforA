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
from src.industry_valuation import estimate_industry_type, estimate_industry_valuation, IndustryType, IndustryValuationResult  # noqa: E402
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


def render_scenarios_table(scenarios: dict):
    df = pd.DataFrame(list(scenarios.items()), columns=['情景', '内在价值'])
    df['内在价值'] = df['内在价值'].apply(lambda x: f"{x:,.0f}")
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
    grade_col, source_col, industry_col = st.columns(3)
    grade_col.metric("数据质量", validated.data_quality_grade)
    source_col.metric("数据模式", validated.metadata.get("source_mode", "fixture"))
    industry_type = estimate_industry_type(ticker, validated)
    industry_col.metric("行业类型", industry_type.value)
    
    cols = st.columns(max(len(validated.verification), 1))
    for idx, (metric, status) in enumerate(validated.verification.items()):
        cols[idx % len(cols)].metric(metric, render_badge(status))

    st.subheader("风险画像")
    st.json(risk.trace)

    st.subheader("估值结果")
    if 'industry_val' in st.session_state:
        industry_val = st.session_state['industry_val']
        st.markdown("### 🎯 行业特定估值")
        st.write(f"内在价值: {industry_val.base_value:,.0f}")
        
        st.markdown("#### 情景分析")
        st.table(render_scenarios_table(industry_val.scenarios))
        
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
        col1.metric("收入驱动DCF", f"{base_result.intrinsic_value:,.0f}")
        col2.metric("ROIC驱动DCF", f"{roic_result.intrinsic_value:,.0f}")
        col3.metric("两阶段DCF", f"{two_stage_result.intrinsic_value:,.0f}")
        
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
