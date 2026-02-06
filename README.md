# A股 TuShare 估值工作台

全新的“多市场绝对估值工作台”版本，专注于中国 A 股，以 TuShare Pro 提供的行情、财务与宏观数据驱动估值。系统强调可验证性、风险透明度与 DuckDB 持久化，支持离线样本运行。

## 目录结构

- `src/`：核心模块（数据管道、风险参数、估值引擎、场景、报告、告警、Streamlit UI）
- `scripts/`：环境检查、DuckDB 初始化、样本数据导入
- `tests/`：单元测试与固定 CSV 样本
- `duckdb/valuation.duckdb`：本地持久化数据库
- `data/fixtures/`：可放置额外样本或宏观数据
- `.streamlit/`：界面主题配置
- `docs/`：路线图与说明

## 快速开始

1. 安装依赖：`python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt`
2. 可选设置 TuShare Token：`export TUSHARE_TOKEN=你的token`
3. 运行环境检查：`./.venv/bin/python scripts/check_env.py`（验证 Token、TuShare 端点与 DuckDB 表结构）
4. 执行测试：`./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
5. 启动前端：`./.venv/bin/streamlit run src/streamlit_app.py`

即使缺少 Token，也会自动落到 `tests/data/*.csv` 的离线样本；Streamlit UI 会显示“fixture mode” 徽章提醒数据来源。

## DuckDB 表设计

- `valuation_runs`：估值输出及场景百分位
- `risk_profiles`：WACC/β/Rf 追踪
- `valuation_alerts`：数据质量与异常监控
- `valuation_backtest`：事后检验
- `raw_prices` / `raw_financials` / `macro_series`：原始缓存

## 功能亮点

- **数据验证**：`load_inputs` 对收入、净利、现金流、净负债等多指标交叉校验，形成 `A/B/C` 质量等级与徽章面板。
- **风险画像**：基于 TuShare 行情与 CSI300 估 β，国债曲线推导无风险利率，并记录回归窗口、样本量、StdErr 以便追踪。
- **双驱动 DCF**：收入/ROIC 两种模型均直接取自 TuShare 财报推导的成长、ROIC、再投资与 fade 年数。
- **场景/Monte Carlo**：使用历史价格波动与宏观 CSV 波动构造分布，输出 5/50/95 百分位和输入参数，供 alert 使用。
- **透明落地**：估值、风险、场景、数据质量、来源模式统一写入 DuckDB；Streamlit 展示数据源可信度，并可一键保存记录 + 告警。

## 后续路线

详见 `docs/roadmap/improve_plan.md`，包含数据覆盖扩展、回测框架与自动化告警等规划。
