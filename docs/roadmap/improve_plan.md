# A股估值平台路线图

## 短期（<=1 个月）
- 扩展 TuShare 数据覆盖：行业指标、个股分红、北向资金
- 完善 `load_inputs` 验证项：经营活动现金流与资本开支拆分、资产负债表勾稽
- 引入自动化 CI（GitHub Actions）执行 `scripts/check_env.py` 与测试，并输出 DuckDB schema drift 报告
- 将 Streamlit 中的 Monte Carlo 可视化扩展为分布图 + 场景说明

## 中期（1-3 个月）
- 增加回测：`valuation_backtest` 表记录估值与随后 3/6/12 个月收益，并生成曲线
- Monte Carlo 多因子：结合 CPI、PMI、社融等宏观序列构建联合分布
- Streamlit 报告导出 PDF，并附数据源签名

## 长期（>3 个月）
- 接入因子库，支持对比 Wind/Choice 数据
- 构建自动告警服务，扫描 DuckDB 中最新估值并推送钉钉/企业微信
- 国际化：在 A 股主干稳定后再评估港股或美股扩展
