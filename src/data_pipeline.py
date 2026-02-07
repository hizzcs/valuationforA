"""Data ingestion, verification, and caching utilities for the valuation platform."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional

try:
    import duckdb
except Exception:  # pragma: no cover - duckdb optional for tests
    duckdb = None  # type: ignore
import pandas as pd
from loguru import logger

ts = None  # lazy-loaded in _get_tushare()
yf = None  # lazy-loaded in _get_yfinance()
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "data"
DUCKDB_PATH = REPO_ROOT / "duckdb" / "valuation.duckdb"
ENV_PATH = REPO_ROOT / ".env"


def _load_env_token(key: str = "TUSHARE_TOKEN", env_path: Path = ENV_PATH) -> Optional[str]:
    """Lightweight .env reader to avoid requiring python-dotenv."""
    if not env_path.exists():
        return None
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() != key:
                continue
            token = v.strip().strip('"').strip("'")
            if token:
                return token
    except Exception:
        return None
    return None


def _get_tushare():
    global ts
    if ts is not None:
        return ts
    try:  # Optional TuShare import for environments with the package installed
        import tushare as _ts
        ts = _ts
    except Exception:  # pragma: no cover - offline environments
        ts = False  # type: ignore[assignment]
    return None if ts is False else ts


def _get_yfinance():
    global yf
    if yf is not None:
        return yf
    try:  # Optional yfinance fallback for CSI300
        import yfinance as _yf
        yf = _yf
    except Exception:  # pragma: no cover
        yf = False  # type: ignore[assignment]
    return None if yf is False else yf


@dataclass
class ValidatedInputs:
    ticker: str
    as_of_date: date
    revenue: float
    net_profit: float
    net_debt: float
    operating_cf: float
    invested_capital: float
    shares_outstanding: float
    verification: Dict[str, str]
    data_quality_grade: str
    research_expense: float = 0.0
    market_share: float = 0.0
    technology_life_cycle: float = 0.0
    capital_ratio: float = 0.0
    npl_ratio: float = 0.0
    net_interest_margin: float = 0.0
    consumer_upgrade: float = 0.0
    competition_intensity: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)
    statements: pd.DataFrame = field(default_factory=pd.DataFrame)


class TuShareClient:
    """Wrapper around TuShare that logs every call and falls back to fixtures."""

    def __init__(self, token: Optional[str] = None, fixtures_dir: Path = FIXTURE_DIR):
        if token is not None:
            resolved_token = token.strip() or None
        else:
            env_token = os.getenv("TUSHARE_TOKEN")
            if env_token is not None:  # Explicitly set (including empty) should take precedence over .env.
                resolved_token = env_token.strip() or None
            else:
                resolved_token = _load_env_token()
        self.token = resolved_token
        self.fixtures_dir = fixtures_dir
        self._pro = None
        self.last_call: Dict[str, str] = {}
        ts_module = _get_tushare() if self.token else None
        if self.token and ts_module is not None:
            try:
                self._pro = ts_module.pro_api(self.token)
                logger.info("Initialized TuShare client with provided token.")
            except Exception as exc:  # pragma: no cover - runtime failure path
                logger.warning("Failed to init TuShare, falling back to fixtures: {}", exc)
        else:
            logger.info("TuShare token missing; operating in fixture mode.")

    @property
    def mode(self) -> str:
        return "live" if self._pro is not None else "fixture"

    def call_api(self, endpoint: str, **params) -> pd.DataFrame:
        start = datetime.now(timezone.utc)
        cache_key = _hash_params(endpoint, params)
        if self._pro is not None:
            try:
                func = getattr(self._pro, endpoint)
                df = func(**params)
                duration = (datetime.now(timezone.utc) - start).total_seconds()
                self.last_call = {
                    "endpoint": endpoint,
                    "mode": "live",
                    "cache_key": cache_key,
                    "duration": f"{duration:.3f}",
                }
                logger.info(
                    "TuShare call endpoint={} rows={} cache_key={} duration={:.3f}s mode=live",
                    endpoint,
                    len(df) if hasattr(df, "__len__") else "na",
                    cache_key,
                    duration,
                )
                return pd.DataFrame(df)
            except Exception as exc:  # pragma: no cover
                logger.error("TuShare call failed ({}): {} -- falling back to fixtures", endpoint, exc)
        fixture_path = self.fixtures_dir / f"{endpoint}.csv"
        if not fixture_path.exists():
            yf_module = _get_yfinance()
            if endpoint == "csi300" and yf_module is not None:
                logger.warning("Fixture missing for CSI300, downloading via yfinance fallback.")
                data = yf_module.download("000300.SS", period="1y", progress=False)
                df = (
                    data.reset_index()
                    .rename(columns={"Date": "trade_date", "Close": "close"})[["trade_date", "close"]]
                )
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
                duration = (datetime.now(timezone.utc) - start).total_seconds()
                self.last_call = {
                    "endpoint": endpoint,
                    "mode": "fallback",
                    "cache_key": cache_key,
                    "duration": f"{duration:.3f}",
                }
                logger.info(
                    "CSI300 yfinance fallback rows={} cache_key={} duration={:.3f}s",
                    len(df),
                    cache_key,
                    duration,
                )
                return df
            raise FileNotFoundError(f"Fixture {fixture_path} missing for endpoint {endpoint}")
        df = pd.read_csv(fixture_path)
        ts_code = params.get("ts_code")
        if ts_code and "ts_code" in df.columns:
            df = df[df["ts_code"] == ts_code]
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        self.last_call = {
            "endpoint": endpoint,
            "mode": "fixture",
            "cache_key": cache_key,
            "duration": f"{duration:.3f}",
        }
        logger.info(
            "Loaded fixture endpoint={} rows={} cache_key={} duration={:.3f}s mode=fixture",
            endpoint,
            len(df),
            cache_key,
            duration,
        )
        return df


def ensure_duckdb_schema(path: Path = DUCKDB_PATH) -> None:
    if duckdb is None:
        logger.warning("duckdb package missing; skipping schema creation.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        con = duckdb.connect(path.as_posix())
    except Exception as exc:
        # A zero-byte file is common when repos accidentally commit an empty placeholder.
        if path.exists() and path.stat().st_size == 0:
            logger.warning("DuckDB file {} is empty; recreating it.", path)
            path.unlink(missing_ok=True)
            con = duckdb.connect(path.as_posix())
        elif path.exists() and "not a valid duckdb database file" in str(exc).lower():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            backup = path.with_name(f"{path.name}.invalid-{stamp}")
            path.rename(backup)
            logger.warning("Invalid DuckDB file moved to {}; recreating {}.", backup, path)
            con = duckdb.connect(path.as_posix())
        else:
            raise
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS valuation_runs (
                id UUID DEFAULT uuid(),
                ticker TEXT,
                as_of_date DATE,
                method TEXT,
                intrinsic_value DOUBLE,
                percentile_5 DOUBLE,
                percentile_50 DOUBLE,
                percentile_95 DOUBLE,
                scenario_seed INTEGER,
                scenario_inputs JSON,
                wacc_details JSON,
                data_quality_grade TEXT,
                source_mode TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_profiles (
                id UUID DEFAULT uuid(),
                ticker TEXT,
                as_of_date DATE,
                beta DOUBLE,
                risk_free DOUBLE,
                cost_of_equity DOUBLE,
                cost_of_debt DOUBLE,
                wacc DOUBLE,
                observations INTEGER,
                std_err DOUBLE,
                trace JSON,
                window_start DATE,
                window_end DATE,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS valuation_alerts (
                id UUID DEFAULT uuid(),
                ticker TEXT,
                alert_type TEXT,
                message TEXT,
                severity TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS valuation_backtest (
                id UUID DEFAULT uuid(),
                ticker TEXT,
                valuation_date DATE,
                forward_return DOUBLE,
                realized_vs_intrinsic DOUBLE,
                notes TEXT,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_prices (
                ticker TEXT,
                trade_date DATE,
                close DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_financials (
                ticker TEXT,
                end_date DATE,
                revenue DOUBLE,
                net_profit DOUBLE,
                net_debt DOUBLE,
                operating_cf DOUBLE,
                invested_capital DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS macro_series (
                series TEXT,
                obs_date DATE,
                value DOUBLE
            )
            """
        )
    finally:
        con.close()


def cache_dataframe(table: str, df: pd.DataFrame, path: Path = DUCKDB_PATH) -> None:
    if df.empty:
        return
    if duckdb is None:
        logger.warning("duckdb package missing; skip caching table={}", table)
        return
    ensure_duckdb_schema(path)
    con = duckdb.connect(path.as_posix())
    try:
        table_cols = [row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        if not table_cols:
            raise ValueError(f"Unknown table {table}")

        normalized = df.copy()
        if "ticker" in table_cols and "ticker" not in normalized.columns and "ts_code" in normalized.columns:
            normalized = normalized.rename(columns={"ts_code": "ticker"})
        for col in table_cols:
            if col not in normalized.columns:
                normalized[col] = pd.NA
        normalized = normalized[table_cols]
        con.register("df", normalized)

        identifier_col = None
        if "ticker" in table_cols and "ticker" in normalized.columns:
            identifier_col = "ticker"
        elif "ts_code" in table_cols and "ts_code" in normalized.columns:
            identifier_col = "ts_code"
        if identifier_col:
            ids = normalized[identifier_col].dropna().unique().tolist()
            if ids:
                placeholders = ",".join("?" for _ in ids)
                con.execute(f"DELETE FROM {table} WHERE {identifier_col} IN ({placeholders})", ids)
        else:
            con.execute(f"DELETE FROM {table}")
        quoted_cols = ", ".join(f'"{col}"' for col in table_cols)
        con.execute(f"INSERT INTO {table} ({quoted_cols}) SELECT {quoted_cols} FROM df")
    finally:
        con.close()


def load_financials(client: TuShareClient, ticker: str, as_of: Optional[date] = None) -> pd.DataFrame:
    df = client.call_api("financials", ts_code=ticker)
    if df.empty:
        return df
    df["end_date"] = pd.to_datetime(df["end_date"])
    if as_of is not None:
        df = df[df["end_date"] <= pd.Timestamp(as_of)]
    if df.empty:
        return df
    df = df.sort_values("end_date")
    for col in ["revenue", "net_profit", "net_debt", "operating_cf", "invested_capital"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["interest_expense", "total_debt", "shares_outstanding", "total_share", "total_shares", "share_total"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cache_dataframe("raw_financials", df)
    return df


def load_prices(client: TuShareClient, ticker: str, as_of: Optional[date] = None) -> pd.DataFrame:
    df = client.call_api("daily", ts_code=ticker)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if as_of is not None:
        df = df[df["trade_date"] <= pd.Timestamp(as_of)]
    df = df.sort_values("trade_date")[["ts_code", "trade_date", "close"]]
    cache_dataframe("raw_prices", df.rename(columns={"ts_code": "ticker"}))
    return df


def load_macro_series(client: TuShareClient, as_of: Optional[date] = None) -> pd.DataFrame:
    df = client.call_api("macro")  # fixture only
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    if as_of is not None:
        df = df[df["obs_date"] <= pd.Timestamp(as_of)]
    cache_dataframe("macro_series", df)
    return df


def load_inputs(client: Optional[TuShareClient], ticker: str, as_of: date) -> ValidatedInputs:
    client = client or TuShareClient()
    fin = load_financials(client, ticker, as_of=as_of)
    if fin.empty:
        raise ValueError(f"No financial statements available for {ticker} on or before {as_of}")
    latest = fin.iloc[-1]
    previous = fin.iloc[-2] if len(fin) > 1 else latest
    verification: Dict[str, str] = {}

    def _ratio_status(metric: str, value: float, lo: float, hi: float, warn_pad: float = 0.2) -> None:
        if pd.isna(value):
            verification[metric] = "fail"
            return
        if lo <= value <= hi:
            verification[metric] = "pass"
        elif (lo - warn_pad) <= value <= (hi + warn_pad):
            verification[metric] = "warn"
        else:
            verification[metric] = "fail"

    def _diff_status(metric: str, value: float, pass_threshold: float, warn_threshold: float) -> None:
        if pd.isna(value):
            verification[metric] = "fail"
            return
        if value <= pass_threshold:
            verification[metric] = "pass"
        elif value <= warn_threshold:
            verification[metric] = "warn"
        else:
            verification[metric] = "fail"

    def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
        value = pd.to_numeric(row.get(key, default), errors="coerce")
        if pd.isna(value):
            return default
        return float(value)

    def _normalize_shares(raw: float) -> float:
        # TuShare's `total_share` is often reported in 10k-share units.
        # If the value is suspiciously small, scale it to shares.
        if raw <= 0:
            return 0.0
        return raw * 10000 if raw < 1e8 else raw

    def _extract_shares(financials: pd.DataFrame) -> float:
        latest_row = financials.iloc[-1]
        for col in ("shares_outstanding", "total_shares", "share_total", "total_share"):
            value = _safe_float(latest_row, col, default=0.0)
            if value > 0:
                return _normalize_shares(value)

        # Fallback to daily_basic when financial statements don't carry share count.
        try:
            daily_basic = client.call_api("daily_basic", ts_code=ticker)
            if not daily_basic.empty:
                basic = daily_basic.copy()
                if "trade_date" in basic.columns:
                    basic["trade_date"] = pd.to_datetime(basic["trade_date"])
                    basic = basic[basic["trade_date"] <= pd.Timestamp(as_of)]
                if not basic.empty:
                    latest_basic = basic.sort_values("trade_date").iloc[-1] if "trade_date" in basic.columns else basic.iloc[-1]
                    value = _safe_float(latest_basic, "total_share", default=0.0)
                    if value > 0:
                        return _normalize_shares(value)
        except (FileNotFoundError, AttributeError, KeyError):
            pass
        return 0.0

    revenue = _safe_float(latest, "revenue")
    net_profit = _safe_float(latest, "net_profit")
    net_debt = _safe_float(latest, "net_debt")
    operating_cf = _safe_float(latest, "operating_cf")
    invested_capital = _safe_float(latest, "invested_capital")
    shares_outstanding = _extract_shares(fin)
    industry = str(latest.get("industry", latest.get("industry_name", ""))).strip()

    cash_conversion = operating_cf / max(net_profit, 1e-6)
    _ratio_status("cash_conversion", cash_conversion, 0.7, 1.3, warn_pad=0.6)

    profit_margin = net_profit / max(revenue, 1.0)
    _ratio_status("net_profit_margin", profit_margin, 0.05, 0.35, warn_pad=0.15)

    if len(fin) > 1:
        rev_growth = (latest.get("revenue", 0.0) - previous.get("revenue", 0.0)) / max(
            previous.get("revenue", 1.0), 1.0
        )
        profit_growth = (latest.get("net_profit", 0.0) - previous.get("net_profit", 0.0)) / max(
            previous.get("net_profit", 1.0), 1.0
        )
        growth_diff = abs(rev_growth - profit_growth)
        _diff_status("growth_alignment", growth_diff, 0.05, 0.15)

    debt_to_capital = net_debt / max(invested_capital, 1.0)
    _ratio_status("net_debt_to_invested", debt_to_capital, 0.0, 1.0, warn_pad=0.5)

    leverage_to_cash = net_debt / max(abs(operating_cf), 1.0)
    _ratio_status("net_debt_vs_cf", leverage_to_cash, -2.5, 2.5, warn_pad=1.0)

    passes = sum(1 for status in verification.values() if status == "pass")
    warns = sum(1 for status in verification.values() if status == "warn")
    fails = sum(1 for status in verification.values() if status == "fail")
    total = max(len(verification), 1)
    score = (passes + 0.5 * warns) / total
    if fails == 0 and score >= 0.75:
        grade = "A"
    elif score >= 0.5:
        grade = "B"
    else:
        grade = "C"

    source_mode = client.last_call.get("mode", client.mode)
    statement_end = latest.get("end_date")
    if isinstance(statement_end, pd.Timestamp):
        statement_end = statement_end.date().isoformat()
    metadata = {
        "source": "tushare" if source_mode == "live" else source_mode,
        "source_mode": source_mode,
        "statement_end_date": str(statement_end),
        "statement_rows": str(len(fin)),
        "verification_summary": f"pass:{passes}|warn:{warns}|fail:{fails}",
        "last_endpoint": client.last_call.get("endpoint", "financials"),
        "industry": industry,
        "shares_outstanding": str(shares_outstanding),
    }
    if source_mode != "live":
        metadata["fixture_dir"] = str(client.fixtures_dir)
    if not industry:
        try:
            basic = client.call_api("stock_basic", ts_code=ticker)
            if not basic.empty:
                metadata["industry"] = str(
                    basic.iloc[0].get("industry", basic.iloc[0].get("industry_name", ""))
                ).strip()
        except (FileNotFoundError, AttributeError, KeyError):
            pass

    return ValidatedInputs(
        ticker=ticker,
        as_of_date=as_of,
        revenue=revenue,
        net_profit=net_profit,
        net_debt=net_debt,
        operating_cf=operating_cf,
        invested_capital=invested_capital,
        shares_outstanding=shares_outstanding,
        verification=verification,
        data_quality_grade=grade,
        research_expense=_safe_float(latest, "research_expense"),
        market_share=_safe_float(latest, "market_share"),
        technology_life_cycle=_safe_float(latest, "technology_life_cycle"),
        capital_ratio=_safe_float(latest, "capital_ratio"),
        npl_ratio=_safe_float(latest, "npl_ratio"),
        net_interest_margin=_safe_float(latest, "net_interest_margin"),
        consumer_upgrade=_safe_float(latest, "consumer_upgrade"),
        competition_intensity=_safe_float(latest, "competition_intensity"),
        metadata=metadata,
        statements=fin,
    )


def _hash_params(endpoint: str, params: Dict) -> str:
    raw = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


__all__ = [
    "TuShareClient",
    "ValidatedInputs",
    "ensure_duckdb_schema",
    "cache_dataframe",
    "load_financials",
    "load_prices",
    "load_macro_series",
    "load_inputs",
]
