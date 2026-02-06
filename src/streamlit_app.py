from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
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
from src.scenario_engine import build_distribution_params, run_monte_carlo  # noqa: E402
from src.reporting import log_risk_profile, save_valuation  # noqa: E402
from src.valuation_core import revenue_driven_dcf, roic_driven_dcf  # noqa: E402
from src.alerts import evaluate_alerts, persist_alerts  # noqa: E402


st.set_page_config(page_title="A股绝对估值", layout="wide")
st.title("A股 TuShare 估值工作台")
ensure_duckdb_schema()

client = TuShareClient()

def render_badge(status: str) -> str:
    if status == "pass":
        return "✅"
    if status == "warn":
        return "⚠️"
    return "❌"


def main() -> None:
    sidebar = st.sidebar
    ticker = sidebar.text_input("TuShare TS 代码", value="600000.SH")
    as_of = sidebar.date_input("估值日期", value=date(2024, 12, 31))
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
        base_result = revenue_driven_dcf(validated, risk)
        roic_result = roic_driven_dcf(validated, risk)
        params = build_distribution_params(prices, macro, base_result)
        mc = run_monte_carlo(base_result, params, draws=scenario_draws)

    st.subheader("数据校验")
    grade_col, source_col = st.columns(2)
    grade_col.metric("数据质量", validated.data_quality_grade)
    source_col.metric("数据模式", validated.metadata.get("source_mode", "fixture"))
    cols = st.columns(max(len(validated.verification), 1))
    for idx, (metric, status) in enumerate(validated.verification.items()):
        cols[idx % len(cols)].metric(metric, render_badge(status))

    st.subheader("风险画像")
    st.json(risk.trace)

    st.subheader("估值结果 - 收入驱动")
    st.write(f"内在价值: {base_result.intrinsic_value:,.0f}")
    st.write(pd.DataFrame({"FCFF": base_result.cash_flows, "DF": base_result.discount_factors}))

    st.subheader("估值结果 - ROIC 驱动")
    st.write(f"内在价值: {roic_result.intrinsic_value:,.0f}")

    st.subheader("Monte Carlo 百分位")
    st.write(
        {
            "5%": mc.percentile_5,
            "50%": mc.percentile_50,
            "95%": mc.percentile_95,
            "draws": mc.draws,
        }
    )

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
