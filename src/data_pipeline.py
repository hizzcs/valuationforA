import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import tushare as ts


@dataclass
class TushareClient:
    token: str

    def pro(self):
        return ts.pro_api(self.token)


def get_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("TUSHARE_TOKEN not set")
    return token


def fetch_financials(ts_code: str) -> dict:
    """Fetch basic financial statements from Tushare."""
    pro = TushareClient(get_token()).pro()
    income = pro.income(ts_code=ts_code, fields="ts_code,ann_date,f_ann_date,end_date,report_type,net_profit,revenue")
    balance = pro.balancesheet(ts_code=ts_code, fields="ts_code,ann_date,end_date,total_assets,total_liab")
    cashflow = pro.cashflow(ts_code=ts_code, fields="ts_code,ann_date,end_date,n_cashflow_act")
    return {"income": income, "balance": balance, "cashflow": cashflow}


def fetch_daily_price(ts_code: str) -> pd.DataFrame:
    pro = TushareClient(get_token()).pro()
    df = pro.daily(ts_code=ts_code, fields="ts_code,trade_date,close")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("trade_date")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    return df


def fetch_basic(ts_code: str) -> Optional[pd.DataFrame]:
    pro = TushareClient(get_token()).pro()
    df = pro.stock_basic(ts_code=ts_code, fields="ts_code,name,industry,area,market")
    return df


def fetch_daily_basic(ts_code: str) -> Optional[pd.DataFrame]:
    pro = TushareClient(get_token()).pro()
    df = pro.daily_basic(ts_code=ts_code, fields="ts_code,trade_date,total_mv,circ_mv")
    if df is None or df.empty:
        return None
    df = df.sort_values("trade_date")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    return df


def fetch_index_weight(index_code: str, limit: int = 50) -> pd.DataFrame:
    """Fetch top N constituents of an index from Tushare."""
    pro = TushareClient(get_token()).pro()
    latest = pro.index_weight(index_code=index_code, limit=1)
    trade_date = latest['trade_date'].iloc[0] if not latest.empty else None
    if not trade_date:
        return pd.DataFrame()
    df = pro.index_weight(index_code=index_code, trade_date=trade_date)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values('weight', ascending=False)
    if limit > 0:
        df = df.head(limit)
    return df


def fetch_all_stock_basic() -> pd.DataFrame:
    """Fetch all stock basic info for filtering."""
    pro = TushareClient(get_token()).pro()
    return pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,area,market')
