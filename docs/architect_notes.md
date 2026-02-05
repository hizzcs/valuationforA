# Architecture Notes (valuationforA)

## Investor Manager Perspective (Model Gaps)
- Need FCFF/FCFE separation and explicit capital structure assumptions.
- Per-share valuation should reflect net debt and shares outstanding.
- Provide scenario ranges (bear/base/bull) and margin of safety.
- Make assumptions transparent and auditable.

## Architect Perspective (Structure Improvements)
- Separate data ingestion, model logic, and presentation layers.
- Add caching and retry for Tushare calls.
- Introduce config-driven model parameters.
- Keep reporting module independent for reuse.

## Action Items (Applied)
- Added per-share valuation via shares outstanding and net debt placeholders.
- Added daily_basic fetch for market cap proxy.
- Added reporting module and UI integration.
- TODO: implement FCFF/FCFE models + scenario engine.
