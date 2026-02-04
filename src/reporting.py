from datetime import datetime
from typing import Dict

import pandas as pd


def generate_report(payload: Dict) -> str:
    ts_code = payload.get("ts_code")
    name = payload.get("name")
    valuation = payload.get("valuation")
    mispricing = payload.get("mispricing")
    inputs = payload.get("inputs")
    assumptions = payload.get("assumptions")

    lines = []
    lines.append(f"# 估值报告 - {name} ({ts_code})")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 结论")
    if mispricing is not None:
        lines.append(f"- 估值偏离：{mispricing*100:.2f}%")
    lines.append(f"- DCF 估值：{valuation:,.0f}")
    lines.append("")
    lines.append("## 核心假设")
    for k, v in assumptions.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 输入数据")
    for k, v in inputs.items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)
