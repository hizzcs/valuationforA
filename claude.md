# Claude Code Guide (valuationforA)

## Project Goal
Build an A-share valuation workspace with clean, reproducible data pipelines and transparent assumptions.

## Repo Structure
- `src/` core logic
- `data/` local data (ignored)
- `notebooks/` exploration
- `docs/` notes/specs

## Conventions
- Prefer deterministic scripts under `src/`.
- Keep data sources and assumptions explicit.
- Log all data pulls with timestamps and parameters.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Next Steps
- Define data ingestion pipeline
- Draft valuation model assumptions
- Add basic tests
