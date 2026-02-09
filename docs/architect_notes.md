# Architecture Notes (valuationforA)

## Agent-Swarm Execution Model

本次改造采用“角色化多 agent 分工”落地，而非单文件堆逻辑：

- `Domain Agent`：负责估值领域编排，输出统一快照对象，隔离业务计算与 UI。
- `UI Agent`：负责信息架构与视觉层，重建为多标签仪表盘（总览/模型拆解/风险/方法/落库）。
- `Viz Agent`：负责图表叙事，从单图展示升级为方法对比 + 区间定位 + 分布 + 雷达 + Tornado。

## Domain Design (Applied)

新增 `src/workbench_domain.py`，将 UI 所需的业务聚合收敛到领域层：

- `MethodSnapshot`：单个估值方法快照（企业价值、股权价值、每股价值、假设、现金流）。
- `ScenarioBand`：Monte Carlo 区间（P5/P50/P95）。
- `DashboardSnapshot`：页面级聚合对象（数据质量、风险、方法、价格、安全边际）。
- `build_method_results` / `build_dashboard_snapshot`：统一入口，避免在 Streamlit 页面里拼装业务对象。

结果：展示层只消费快照，不直接处理估值方法细节，后续接 API 或替换前端时成本更低。

## UI Design (Applied)

`src/streamlit_app.py` 重构为分区仪表盘：

- **总览**：质量/来源/行业/市场价/选中模型，每股价值一屏可读。
- **模型拆解**：行业特定输出 + 基准方法对比 + 现金流轨迹 + 参数表。
- **风险可视化**：风险雷达、分布直方图、Tornado、参数追踪。
- **方法说明**：估值公式与适用场景。
- **数据与落库**：来源透明度 + DuckDB 持久化与告警。

并补充了统一 CSS 主题（中文字体、渐变背景、指标卡片），提升桌面和移动端阅读一致性。

## Visualization Improvements (Applied)

- 方法对比柱状图（企业价值横向比较）
- 估值区间定位图（P5-P95 + 选中模型位置）
- Monte Carlo 分布直方图（含 P5/P50/P95 与基准线）
- 风险雷达图（Beta/WACC/债务成本/StdErr/R2 归一化）
- 现金流轨迹图（自由现金流 + 现值）

## Testing

- 新增 `tests/test_workbench_domain.py`：覆盖领域层快照构建、方法对比、安全边际计算。
- 全量单测通过：`python -m unittest discover -s tests -p 'test_*.py'`。
