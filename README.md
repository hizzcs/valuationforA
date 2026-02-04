# valuationforA

A-share valuation workspace (Tushare only).

## Structure
- `src/` core code
- `data/` local data (ignored)
- `notebooks/` exploration
- `docs/` notes and specs

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TUSHARE_TOKEN=your_token
streamlit run src/streamlit_app.py
```

## Notes
- Single data source: Tushare
- If data fails to load, check token and network
