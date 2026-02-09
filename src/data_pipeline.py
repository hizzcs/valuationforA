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

    @staticmethod
    def _quarter_end_date(quarter: str) -> Optional[pd.Timestamp]:
        if not isinstance(quarter, str) or "Q" not in quarter:
            return None
        try:
            year = int(quarter[:4])
            q = int(quarter[-1])
            if q == 1:
                return pd.Timestamp(year=year, month=3, day=31)
            if q == 2:
                return pd.Timestamp(year=year, month=6, day=30)
            if q == 3:
                return pd.Timestamp(year=year, month=9, day=30)
            if q == 4:
                return pd.Timestamp(year=year, month=12, day=31)
        except Exception:
            return None
        return None

    def _call_live_alias(self, endpoint: str, params: Dict[str, object]) -> Optional[pd.DataFrame]:
        if self._pro is None:
            return None

        if endpoint == "csi300":
            # Official CSI300 endpoint.
            query = {"ts_code": "000300.SH"}
            if "start_date" in params:
                query["start_date"] = params["start_date"]
            if "end_date" in params:
                query["end_date"] = params["end_date"]
            if "limit" in params:
                query["limit"] = params["limit"]
            return pd.DataFrame(self._pro.index_daily(**query))

        if endpoint == "bonds":
            # 国债收益率曲线，提取接近10Y点并转换成小数（如 2.1 -> 0.021）
            query = {"curve_type": "0"}
            if "limit" in params:
                query["limit"] = params["limit"]
            if "start_date" in params:
                query["start_date"] = params["start_date"]
            if "end_date" in params:
                query["end_date"] = params["end_date"]
            tenor = float(params.get("tenor", 10))
            raw = pd.DataFrame(self._pro.yc_cb(**query))
            if raw.empty:
                return raw
            raw["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce")
            raw["curve_term"] = pd.to_numeric(raw.get("curve_term"), errors="coerce")
            raw["yield"] = pd.to_numeric(raw.get("yield"), errors="coerce")
            raw = raw.dropna(subset=["trade_date", "curve_term", "yield"])
            if raw.empty:
                return pd.DataFrame(columns=["obs_date", "value", "curve_term", "curve_name", "ts_code"])
            raw["term_distance"] = (raw["curve_term"] - tenor).abs()
            selected = (
                raw.sort_values(["trade_date", "term_distance"])
                .groupby("trade_date", as_index=False)
                .first()
            )
            out = selected.rename(columns={"trade_date": "obs_date", "yield": "value"})
            out["value"] = out["value"] / 100
            return out[["obs_date", "value", "curve_term", "curve_name", "ts_code"]]

        if endpoint == "macro":
            # 优先 GDP 同比，作为估值场景宏观因子；为空则退回 CPI 同比。
            gdp = pd.DataFrame(self._pro.cn_gdp())
            if not gdp.empty:
                gdp["obs_date"] = gdp["quarter"].map(self._quarter_end_date)
                gdp["value"] = pd.to_numeric(gdp.get("gdp_yoy"), errors="coerce") / 100
                gdp = gdp.dropna(subset=["obs_date", "value"])
                if not gdp.empty:
                    gdp["series"] = "gdp_yoy"
                    return gdp[["series", "obs_date", "value"]]
            cpi = pd.DataFrame(self._pro.cn_cpi())
            if cpi.empty:
                return pd.DataFrame(columns=["series", "obs_date", "value"])
            cpi["obs_date"] = pd.to_datetime(cpi["month"], format="%Y%m", errors="coerce")
            cpi["value"] = pd.to_numeric(cpi.get("nt_yoy"), errors="coerce") / 100
            cpi = cpi.dropna(subset=["obs_date", "value"])
            cpi["series"] = "cpi_yoy"
            return cpi[["series", "obs_date", "value"]]

        return None

    def call_api(self, endpoint: str, **params) -> pd.DataFrame:
        start = datetime.now(timezone.utc)
        cache_key = _hash_params(endpoint, params)
        if self._pro is not None:
            try:
                alias_df = self._call_live_alias(endpoint, params)
                if alias_df is not None:
                    df = alias_df
                else:
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


def _safe_call_api(client: TuShareClient, endpoint: str, **params) -> pd.DataFrame:
    """Best-effort endpoint call that returns an empty frame on failure."""
    try:
        return client.call_api(endpoint, **params)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        logger.warning("Endpoint {} unavailable for {}: {}", endpoint, params.get("ts_code", ""), exc)
        return pd.DataFrame()


def _normalize_end_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(series, errors="coerce")


def _coalesce_numeric(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    out = pd.Series(index=frame.index, dtype=float)
    for col in candidates:
        if col in frame.columns:
            out = out.fillna(pd.to_numeric(frame[col], errors="coerce"))
    return out


def _alternate_ts_code(ticker: str) -> Optional[str]:
    upper = ticker.upper()
    if upper.endswith(".SH"):
        return f"{upper[:-3]}.SZ"
    if upper.endswith(".SZ"):
        return f"{upper[:-3]}.SH"
    return None


def _load_financials_from_standard_endpoints(client: TuShareClient, ticker: str) -> pd.DataFrame:
    """Build a normalized financials frame from official TuShare endpoints."""
    income = _safe_call_api(client, "income", ts_code=ticker)
    cashflow = _safe_call_api(client, "cashflow", ts_code=ticker)
    balance = _safe_call_api(client, "balancesheet", ts_code=ticker)

    parts: list[pd.DataFrame] = []

    if not income.empty and "end_date" in income.columns:
        frame = pd.DataFrame(
            {
                "ts_code": income.get("ts_code", ticker),
                "end_date": _normalize_end_date(income["end_date"]),
                "revenue": _coalesce_numeric(income, ["revenue", "total_revenue"]),
                "net_profit": _coalesce_numeric(income, ["n_income_attr_p", "n_income", "net_profit"]),
                "interest_expense": _coalesce_numeric(income, ["fin_exp_int_exp", "int_exp", "fin_exp"]),
            }
        )
        parts.append(frame)

    if not cashflow.empty and "end_date" in cashflow.columns:
        frame = pd.DataFrame(
            {
                "ts_code": cashflow.get("ts_code", ticker),
                "end_date": _normalize_end_date(cashflow["end_date"]),
                "operating_cf": _coalesce_numeric(cashflow, ["n_cashflow_act", "operate_cashflow", "operating_cf"]),
            }
        )
        parts.append(frame)

    if not balance.empty and "end_date" in balance.columns:
        total_liab = _coalesce_numeric(balance, ["total_liab", "total_debt"])
        money_cap = _coalesce_numeric(balance, ["money_cap"])
        total_assets = _coalesce_numeric(balance, ["total_assets"])
        equity = _coalesce_numeric(balance, ["total_hldr_eqy_inc_min_int", "total_hldr_eqy_exc_min_int"])

        invested = total_assets.where(total_assets.notna(), total_liab + equity)
        net_debt = total_liab.where(total_liab.notna(), pd.Series(index=balance.index, dtype=float)) - money_cap.fillna(0.0)

        frame = pd.DataFrame(
            {
                "ts_code": balance.get("ts_code", ticker),
                "end_date": _normalize_end_date(balance["end_date"]),
                "net_debt": net_debt,
                "invested_capital": invested,
                "total_debt": total_liab,
                "shares_outstanding": _coalesce_numeric(balance, ["total_share", "shares_outstanding"]),
            }
        )
        parts.append(frame)

    if not parts:
        return pd.DataFrame()

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["ts_code", "end_date"], how="outer")

    merged["ts_code"] = merged["ts_code"].fillna(ticker)
    merged = merged.dropna(subset=["end_date"]).sort_values("end_date")
    merged = merged.drop_duplicates(subset=["ts_code", "end_date"], keep="last").reset_index(drop=True)
    return merged


def load_financials(client: TuShareClient, ticker: str, as_of: Optional[date] = None) -> pd.DataFrame:
    requested_ticker = ticker.upper()
    resolved_ticker = requested_ticker

    # Live mode: use official TuShare statements first.
    df = _load_financials_from_standard_endpoints(client, requested_ticker) if client.mode == "live" else pd.DataFrame()

    if df.empty:
        alt = _alternate_ts_code(requested_ticker)
        if alt:
            alt_df = _load_financials_from_standard_endpoints(client, alt) if client.mode == "live" else pd.DataFrame()
            if not alt_df.empty:
                logger.warning("No financials for {}; using alternate ticker {}.", requested_ticker, alt)
                df = alt_df
                resolved_ticker = alt

    # Fixture/backward fallback
    if df.empty:
        # Fixture mode fallback only.
        if client.mode != "live":
            df = _safe_call_api(client, "financials", ts_code=requested_ticker)
            if not df.empty:
                resolved_ticker = requested_ticker

    if df.empty and client.mode != "live":
        alt = _alternate_ts_code(requested_ticker)
        if alt:
            alt_df = _safe_call_api(client, "financials", ts_code=alt)
            if not alt_df.empty:
                logger.warning("No financials fixture for {}; using alternate ticker {}.", requested_ticker, alt)
                df = alt_df
                resolved_ticker = alt

    if df.empty:
        return df

    df["end_date"] = _normalize_end_date(df["end_date"])
    df = df.dropna(subset=["end_date"])
    if as_of is not None:
        df = df[df["end_date"] <= pd.Timestamp(as_of)]
    if df.empty:
        return df
    df = df.sort_values("end_date")
    if "ts_code" not in df.columns:
        df["ts_code"] = resolved_ticker
    else:
        df["ts_code"] = df["ts_code"].fillna(resolved_ticker)

    for col in ["revenue", "net_profit", "net_debt", "operating_cf", "invested_capital"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["interest_expense", "total_debt", "shares_outstanding", "total_share", "total_shares", "share_total"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cache_dataframe("raw_financials", df)
    return df


def load_prices(client: TuShareClient, ticker: str, as_of: Optional[date] = None) -> pd.DataFrame:
    requested_ticker = ticker.upper()
    df = _safe_call_api(client, "daily", ts_code=requested_ticker)
    if df.empty:
        alt = _alternate_ts_code(requested_ticker)
        if alt:
            alt_df = _safe_call_api(client, "daily", ts_code=alt)
            if not alt_df.empty:
                logger.warning("No price data for {}; using alternate ticker {}.", requested_ticker, alt)
                df = alt_df

    if df.empty or "trade_date" not in df.columns:
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if as_of is not None:
        df = df[df["trade_date"] <= pd.Timestamp(as_of)]
    if "ts_code" not in df.columns:
        df["ts_code"] = requested_ticker
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
    resolved_ticker = str(fin.iloc[-1].get("ts_code", ticker)).strip() or ticker
    resolved_ticker = resolved_ticker.upper()
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
            daily_basic = client.call_api("daily_basic", ts_code=resolved_ticker)
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
        "requested_ticker": ticker.upper(),
        "resolved_ticker": resolved_ticker,
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
            basic = client.call_api("stock_basic", ts_code=resolved_ticker)
            if not basic.empty:
                metadata["industry"] = str(
                    basic.iloc[0].get("industry", basic.iloc[0].get("industry_name", ""))
                ).strip()
        except (FileNotFoundError, AttributeError, KeyError):
            pass

    return ValidatedInputs(
        ticker=resolved_ticker,
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
