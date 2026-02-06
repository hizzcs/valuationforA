"""Industry-specific valuation models for A-share market."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd

from loguru import logger

from .data_pipeline import ValidatedInputs
from .risk_params import RiskProfile
from .valuation_core import revenue_driven_dcf, roic_driven_dcf, two_stage_dcf, ValuationResult
from .scenario_engine import sensitivity_analysis, tornado_analysis, build_distribution_params, run_monte_carlo


class IndustryType(Enum):
    """Industry classification for A-share market."""
    CYCLICAL = "cyclical"  # 周期性行业（钢铁、煤炭、有色、化工）
    TECH_GROWTH = "tech_growth"  # 科技成长行业（半导体、软件服务、新能源）
    FINANCIAL = "financial"  # 金融行业（银行、保险、券商）
    CONSUMER = "consumer"  # 消费行业（食品饮料、家电、医药）
    UTILITIES = "utilities"  # 公用事业（电力、水务、燃气）
    REAL_ESTATE = "real_estate"  # 房地产
    MANUFACTURING = "manufacturing"  # 制造业
    OTHER = "other"  # 其他行业


class ValuationMethod(Enum):
    """Valuation methods for different industries."""
    DCF = "dcf"  # 自由现金流贴现
    TWO_STAGE_DCF = "two_stage_dcf"  # 两阶段DCF
    THREE_STAGE_DCF = "three_stage_dcf"  # 三阶段DCF
    DDM = "ddm"  # 股利贴现
    RELATIVE = "relative"  # 相对估值
    RESIDUAL_INCOME = "residual_income"  # 剩余收益模型
    EMBEDDED_VALUE = "embedded_value"  # 内含价值（保险）
    PB_ROE = "pb_roe"  # PB-ROE（银行）


@dataclass
class IndustryValuationResult:
    """Industry-specific valuation result."""
    ticker: str
    as_of_date: date
    industry_type: str
    base_value: float
    scenarios: Dict[str, float]
    sensitivity: Optional[Dict[str, List[float]]] = None
    tornado: Optional[Dict[str, float]] = None
    risk_factors: Optional[Dict[str, float]] = None
    assumptions: Optional[Dict[str, float]] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class IndustrySpecificParams:
    """Industry-specific parameters."""
    cycle_length: Optional[int] = None
    growth_stages: Optional[List[Tuple[int, float]]] = None
    minimum_margin: float = 0.01
    maximum_margin: float = 0.4
    minimum_growth: float = -0.5
    maximum_growth: float = 0.5
    terminal_growth: float = 0.02
    wacc_range: Tuple[float, float] = (0.03, 0.3)
    specific_risk_premium: float = 0.0
    required_capital_supply_ratio: Optional[float] = None
    regulatory_caps: Optional[Dict[str, float]] = None


def estimate_industry_type(ticker: str, validated: ValidatedInputs) -> IndustryType:
    """
    Estimate industry type based on ticker and financial characteristics.
    
    Args:
        ticker: Stock ticker
        validated: Validated inputs
        
    Returns:
        IndustryType: Estimated industry type
    """
    if any(term in str(validated.metadata.get("industry", "")).lower() for term in ["银行", "保险", "券商", "金融"]):
        return IndustryType.FINANCIAL
    
    if any(term in str(validated.metadata.get("industry", "")).lower() for term in ["半导体", "软件", "科技", "电子"]):
        return IndustryType.TECH_GROWTH
    
    if any(term in str(validated.metadata.get("industry", "")).lower() for term in ["钢铁", "煤炭", "有色", "化工"]):
        return IndustryType.CYCLICAL
    
    if any(term in str(validated.metadata.get("industry", "")).lower() for term in ["食品", "饮料", "家电", "医药", "消费"]):
        return IndustryType.CONSUMER
    
    if any(term in str(validated.metadata.get("industry", "")).lower() for term in ["电力", "水务", "燃气", "公用"]):
        return IndustryType.UTILITIES
    
    if any(term in str(validated.metadata.get("industry", "")).lower() for term in ["房地产", "地产"]):
        return IndustryType.REAL_ESTATE
    
    return IndustryType.OTHER


def get_industry_specific_params(industry_type: IndustryType) -> IndustrySpecificParams:
    """
    Get industry-specific parameters.
    
    Args:
        industry_type: Industry type
        
    Returns:
        IndustrySpecificParams: Industry-specific parameters
    """
    params = {
        IndustryType.CYCLICAL: IndustrySpecificParams(
            cycle_length=3,
            growth_stages=[(2, 0.15), (1, 0.05), (2, 0.02)],
            minimum_margin=0.03,
            maximum_margin=0.15,
            specific_risk_premium=0.03,
            minimum_growth=-0.1,
            maximum_growth=0.3
        ),
        IndustryType.TECH_GROWTH: IndustrySpecificParams(
            cycle_length=5,
            growth_stages=[(3, 0.3), (2, 0.1), (2, 0.03)],
            minimum_margin=0.05,
            maximum_margin=0.3,
            specific_risk_premium=0.05,
            minimum_growth=-0.2,
            maximum_growth=0.5
        ),
        IndustryType.FINANCIAL: IndustrySpecificParams(
            cycle_length=4,
            growth_stages=[(2, 0.1), (2, 0.05), (1, 0.02)],
            minimum_margin=0.08,
            maximum_margin=0.25,
            specific_risk_premium=0.02,
            minimum_growth=-0.05,
            maximum_growth=0.2
        ),
        IndustryType.CONSUMER: IndustrySpecificParams(
            cycle_length=5,
            growth_stages=[(3, 0.1), (2, 0.05), (2, 0.03)],
            minimum_margin=0.08,
            maximum_margin=0.3,
            specific_risk_premium=0.01,
            minimum_growth=-0.03,
            maximum_growth=0.2
        ),
        IndustryType.UTILITIES: IndustrySpecificParams(
            cycle_length=6,
            growth_stages=[(2, 0.05), (3, 0.03), (2, 0.02)],
            minimum_margin=0.05,
            maximum_margin=0.2,
            specific_risk_premium=0.005,
            minimum_growth=-0.02,
            maximum_growth=0.1
        ),
        IndustryType.REAL_ESTATE: IndustrySpecificParams(
            cycle_length=4,
            growth_stages=[(2, 0.08), (2, 0.03), (2, 0.01)],
            minimum_margin=0.05,
            maximum_margin=0.25,
            specific_risk_premium=0.025,
            minimum_growth=-0.1,
            maximum_growth=0.2
        ),
        IndustryType.MANUFACTURING: IndustrySpecificParams(
            cycle_length=4,
            growth_stages=[(2, 0.12), (2, 0.06), (2, 0.02)],
            minimum_margin=0.05,
            maximum_margin=0.2,
            specific_risk_premium=0.02,
            minimum_growth=-0.08,
            maximum_growth=0.25
        ),
        IndustryType.OTHER: IndustrySpecificParams(
            cycle_length=4,
            growth_stages=[(2, 0.1), (2, 0.05), (2, 0.02)],
            minimum_margin=0.05,
            maximum_margin=0.3,
            specific_risk_premium=0.015,
            minimum_growth=-0.05,
            maximum_growth=0.25
        )
    }
    return params.get(industry_type, params[IndustryType.OTHER])


def cyclical_industry_valuation(
    ticker: str,
    as_of_date: date,
    validated: ValidatedInputs,
    risk: RiskProfile,
    commodity_prices: pd.DataFrame,
    cycle_length: int = 3
) -> IndustryValuationResult:
    """
    Valuation for cyclical industries.
    
    Args:
        ticker: Stock ticker
        as_of_date: Valuation date
        validated: Validated inputs
        risk: Risk profile
        commodity_prices: Commodity prices data
        cycle_length: Cycle length in years
        
    Returns:
        IndustryValuationResult: Valuation result with scenarios
    """
    params = get_industry_specific_params(IndustryType.CYCLICAL)
    adjusted_risk = RiskProfile(
        ticker=risk.ticker,
        as_of_date=risk.as_of_date,
        beta=risk.beta * 1.2,  # 周期性行业β调整
        risk_free=risk.risk_free,
        market_risk_premium=risk.market_risk_premium,
        cost_of_equity=risk.cost_of_equity + params.specific_risk_premium,
        cost_of_debt=risk.cost_of_debt,
        wacc=risk.wacc + params.specific_risk_premium,
        window_start=risk.window_start,
        window_end=risk.window_end,
        observations=risk.observations,
        std_err=risk.std_err,
        trace=risk.trace
    )
    
    # 基础估值
    base_result = two_stage_dcf(
        validated,
        adjusted_risk,
        high_growth_years=params.growth_stages[0][0],
        stable_growth_years=params.growth_stages[1][0],
        high_growth_rate=params.growth_stages[0][1],
        stable_growth_rate=params.growth_stages[1][1],
    )
    
    # 情景分析
    scenarios = {
        "基础情景": base_result.intrinsic_value,
        "繁荣情景": two_stage_dcf(
            validated,
            adjusted_risk,
            high_growth_years=3,
            stable_growth_years=2,
            high_growth_rate=0.25,
            stable_growth_rate=0.08,
        ).intrinsic_value,
        "衰退情景": two_stage_dcf(
            validated,
            adjusted_risk,
            high_growth_years=3,
            stable_growth_years=2,
            high_growth_rate=-0.1,
            stable_growth_rate=0.01,
        ).intrinsic_value
    }
    
    # 敏感性分析
    sensitivity = sensitivity_analysis(
        base_result,
        validated,
        adjusted_risk,
        two_stage_dcf,
        ["high_growth_rate", "stable_growth_rate", "wacc"],
        [[0.1, 0.3], [0.02, 0.1], [0.06, 0.15]]
    )
    
    # 龙卷风分析
    tornado = tornado_analysis(
        base_result,
        validated,
        adjusted_risk,
        two_stage_dcf,
        ["high_growth_rate", "stable_growth_rate", "wacc"]
    )
    
    # 风险因素
    risk_factors = {
        "周期阶段": "衰退期" if validated.revenue < validated.statements["revenue"].mean() else "繁荣期",
        "商品价格波动": np.std(commodity_prices["price"]),
        "需求增长率": validated.revenue / validated.statements["revenue"].iloc[0] - 1
    }
    
    return IndustryValuationResult(
        ticker=ticker,
        as_of_date=as_of_date,
        industry_type=IndustryType.CYCLICAL.value,
        base_value=scenarios["基础情景"],
        scenarios=scenarios,
        sensitivity=sensitivity,
        tornado=tornado,
        risk_factors=risk_factors,
        assumptions=base_result.assumptions,
        metadata=base_result.metadata
    )


def tech_growth_industry_valuation(
    ticker: str,
    as_of_date: date,
    validated: ValidatedInputs,
    risk: RiskProfile
) -> IndustryValuationResult:
    """
    Valuation for tech growth industries.
    
    Args:
        ticker: Stock ticker
        as_of_date: Valuation date
        validated: Validated inputs
        risk: Risk profile
        
    Returns:
        IndustryValuationResult: Valuation result with scenarios
    """
    params = get_industry_specific_params(IndustryType.TECH_GROWTH)
    adjusted_risk = RiskProfile(
        ticker=risk.ticker,
        as_of_date=risk.as_of_date,
        beta=risk.beta * 1.3,  # 科技行业β调整
        risk_free=risk.risk_free,
        market_risk_premium=risk.market_risk_premium,
        cost_of_equity=risk.cost_of_equity + params.specific_risk_premium,
        cost_of_debt=risk.cost_of_debt,
        wacc=risk.wacc + params.specific_risk_premium,
        window_start=risk.window_start,
        window_end=risk.window_end,
        observations=risk.observations,
        std_err=risk.std_err,
        trace=risk.trace
    )
    
    # 基础估值
    base_result = two_stage_dcf(
        validated,
        adjusted_risk,
        high_growth_years=params.growth_stages[0][0],
        stable_growth_years=params.growth_stages[1][0],
        high_growth_rate=params.growth_stages[0][1],
        stable_growth_rate=params.growth_stages[1][1],
    )
    
    # 情景分析
    scenarios = {
        "基础情景": base_result.intrinsic_value,
        "成功情景": two_stage_dcf(
            validated,
            adjusted_risk,
            high_growth_years=3,
            stable_growth_years=2,
            high_growth_rate=0.4,
            stable_growth_rate=0.15,
        ).intrinsic_value,
        "失败情景": two_stage_dcf(
            validated,
            adjusted_risk,
            high_growth_years=3,
            stable_growth_years=2,
            high_growth_rate=-0.15,
            stable_growth_rate=0.01,
        ).intrinsic_value
    }
    
    # 敏感性分析
    sensitivity = sensitivity_analysis(
        base_result,
        validated,
        adjusted_risk,
        two_stage_dcf,
        ["high_growth_rate", "stable_growth_rate", "wacc"],
        [[0.2, 0.45], [0.05, 0.2], [0.08, 0.2]]
    )
    
    # 龙卷风分析
    tornado = tornado_analysis(
        base_result,
        validated,
        adjusted_risk,
        two_stage_dcf,
        ["high_growth_rate", "stable_growth_rate", "wacc"]
    )
    
    # 风险因素
    risk_factors = {
        "研发强度": validated.research_expense / validated.revenue if hasattr(validated, "research_expense") else 0.0,
        "市场份额": validated.market_share if hasattr(validated, "market_share") else 0.0,
        "技术生命周期": validated.technology_life_cycle if hasattr(validated, "technology_life_cycle") else 0.0
    }
    
    return IndustryValuationResult(
        ticker=ticker,
        as_of_date=as_of_date,
        industry_type=IndustryType.TECH_GROWTH.value,
        base_value=scenarios["基础情景"],
        scenarios=scenarios,
        sensitivity=sensitivity,
        tornado=tornado,
        risk_factors=risk_factors,
        assumptions=base_result.assumptions,
        metadata=base_result.metadata
    )


def financial_industry_valuation(
    ticker: str,
    as_of_date: date,
    validated: ValidatedInputs,
    risk: RiskProfile,
    regulatory_data: pd.DataFrame
) -> IndustryValuationResult:
    """
    Valuation for financial industries.
    
    Args:
        ticker: Stock ticker
        as_of_date: Valuation date
        validated: Validated inputs
        risk: Risk profile
        regulatory_data: Regulatory data
        
    Returns:
        IndustryValuationResult: Valuation result with scenarios
    """
    params = get_industry_specific_params(IndustryType.FINANCIAL)
    adjusted_risk = RiskProfile(
        ticker=risk.ticker,
        as_of_date=risk.as_of_date,
        beta=risk.beta * 0.9,  # 金融行业β调整（系统性风险高但稳定）
        risk_free=risk.risk_free,
        market_risk_premium=risk.market_risk_premium,
        cost_of_equity=risk.cost_of_equity + params.specific_risk_premium,
        cost_of_debt=risk.cost_of_debt,
        wacc=risk.wacc + params.specific_risk_premium,
        window_start=risk.window_start,
        window_end=risk.window_end,
        observations=risk.observations,
        std_err=risk.std_err,
        trace=risk.trace
    )
    
    # 基础估值（使用两阶段DCF）
    base_result = two_stage_dcf(
        validated,
        adjusted_risk,
        high_growth_years=params.growth_stages[0][0],
        stable_growth_years=params.growth_stages[1][0],
        high_growth_rate=params.growth_stages[0][1],
        stable_growth_rate=params.growth_stages[1][1],
    )
    
    # 情景分析（考虑监管压力）
    scenarios = {
        "基础情景": base_result.intrinsic_value,
        "放松监管": two_stage_dcf(
            validated,
            adjusted_risk,
            high_growth_years=2,
            stable_growth_years=3,
            high_growth_rate=0.15,
            stable_growth_rate=0.07,
        ).intrinsic_value,
        "强化监管": two_stage_dcf(
            validated,
            adjusted_risk,
            high_growth_years=2,
            stable_growth_years=3,
            high_growth_rate=0.03,
            stable_growth_rate=0.01,
        ).intrinsic_value
    }
    
    # 敏感性分析
    sensitivity = sensitivity_analysis(
        base_result,
        validated,
        adjusted_risk,
        two_stage_dcf,
        ["high_growth_rate", "stable_growth_rate", "wacc"],
        [[0.05, 0.2], [0.02, 0.1], [0.05, 0.12]]
    )
    
    # 龙卷风分析
    tornado = tornado_analysis(
        base_result,
        validated,
        adjusted_risk,
        two_stage_dcf,
        ["high_growth_rate", "stable_growth_rate", "wacc"]
    )
    
    # 风险因素
    risk_factors = {
        "资本充足率": validated.capital_ratio if hasattr(validated, "capital_ratio") else 0.0,
        "不良贷款率": validated.npl_ratio if hasattr(validated, "npl_ratio") else 0.0,
        "净息差": validated.net_interest_margin if hasattr(validated, "net_interest_margin") else 0.0
    }
    
    return IndustryValuationResult(
        ticker=ticker,
        as_of_date=as_of_date,
        industry_type=IndustryType.FINANCIAL.value,
        base_value=scenarios["基础情景"],
        scenarios=scenarios,
        sensitivity=sensitivity,
        tornado=tornado,
        risk_factors=risk_factors,
        assumptions=base_result.assumptions,
        metadata=base_result.metadata
    )


def consumer_industry_valuation(
    ticker: str,
    as_of_date: date,
    validated: ValidatedInputs,
    risk: RiskProfile,
    macro_data: pd.DataFrame
) -> IndustryValuationResult:
    """
    Valuation for consumer industries.
    
    Args:
        ticker: Stock ticker
        as_of_date: Valuation date
        validated: Validated inputs
        risk: Risk profile
        macro_data: Macro economic data
        
    Returns:
        IndustryValuationResult: Valuation result with scenarios
    """
    params = get_industry_specific_params(IndustryType.CONSUMER)
    adjusted_risk = RiskProfile(
        ticker=risk.ticker,
        as_of_date=risk.as_of_date,
        beta=risk.beta * 0.8,  # 消费行业β调整（防御性强）
        risk_free=risk.risk_free,
        market_risk_premium=risk.market_risk_premium,
        cost_of_equity=risk.cost_of_equity + params.specific_risk_premium,
        cost_of_debt=risk.cost_of_debt,
        wacc=risk.wacc + params.specific_risk_premium,
        window_start=risk.window_start,
        window_end=risk.window_end,
        observations=risk.observations,
        std_err=risk.std_err,
        trace=risk.trace
    )
    
    # 基础估值
    base_result = two_stage_dcf(
        validated,
        adjusted_risk,
        high_growth_years=params.growth_stages[0][0],
        stable_growth_years=params.growth_stages[1][0],
        high_growth_rate=params.growth_stages[0][1],
        stable_growth_rate=params.growth_stages[1][1],
    )
    
    # 情景分析（考虑经济增长和消费升级）
    scenarios = {
        "基础情景": base_result.intrinsic_value,
        "经济复苏": two_stage_dcf(
            validated,
            adjusted_risk,
            high_growth_years=3,
            stable_growth_years=2,
            high_growth_rate=0.15,
            stable_growth_rate=0.08,
        ).intrinsic_value,
        "经济放缓": two_stage_dcf(
            validated,
            adjusted_risk,
            high_growth_years=3,
            stable_growth_years=2,
            high_growth_rate=0.05,
            stable_growth_rate=0.02
        ).intrinsic_value
    }
    
    # 敏感性分析
    sensitivity = sensitivity_analysis(
        base_result,
        validated,
        adjusted_risk,
        two_stage_dcf,
        ["high_growth_rate", "stable_growth_rate", "wacc"],
        [[0.08, 0.2], [0.03, 0.1], [0.05, 0.1]]
    )
    
    # 龙卷风分析
    tornado = tornado_analysis(
        base_result,
        validated,
        adjusted_risk,
        two_stage_dcf,
        ["high_growth_rate", "stable_growth_rate", "wacc"]
    )
    
    # 风险因素
    risk_factors = {
        "消费升级": validated.consumer_upgrade if hasattr(validated, "consumer_upgrade") else 0.0,
        "通货膨胀": macro_data["cpi"].iloc[-1] if "cpi" in macro_data.columns else 0.025,
        "竞争强度": validated.competition_intensity if hasattr(validated, "competition_intensity") else 0.0
    }
    
    return IndustryValuationResult(
        ticker=ticker,
        as_of_date=as_of_date,
        industry_type=IndustryType.CONSUMER.value,
        base_value=scenarios["基础情景"],
        scenarios=scenarios,
        sensitivity=sensitivity,
        tornado=tornado,
        risk_factors=risk_factors,
        assumptions=base_result.assumptions,
        metadata=base_result.metadata
    )


def estimate_industry_valuation(
    ticker: str,
    as_of_date: date,
    validated: ValidatedInputs,
    risk: RiskProfile,
    commodity_prices: Optional[pd.DataFrame] = None,
    regulatory_data: Optional[pd.DataFrame] = None,
    macro_data: Optional[pd.DataFrame] = None
) -> IndustryValuationResult:
    """
    Estimate industry-specific valuation based on ticker and industry classification.
    
    Args:
        ticker: Stock ticker
        as_of_date: Valuation date
        validated: Validated inputs
        risk: Risk profile
        commodity_prices: Commodity prices data (for cyclical industries)
        regulatory_data: Regulatory data (for financial industries)
        macro_data: Macro economic data (for consumer industries)
        
    Returns:
        IndustryValuationResult: Industry-specific valuation result
    """
    industry_type = estimate_industry_type(ticker, validated)
    
    try:
        if industry_type == IndustryType.CYCLICAL:
            logger.info(f"Using cyclical industry valuation for {ticker}")
            if commodity_prices is None:
                logger.warning("Cyclical industry valuation without commodity prices data")
                commodity_prices = pd.DataFrame({
                    "date": pd.date_range(as_of_date - pd.DateOffset(years=3), as_of_date),
                    "price": np.random.uniform(80, 120, 36)
                })
            return cyclical_industry_valuation(
                ticker,
                as_of_date,
                validated,
                risk,
                commodity_prices
            )
        
        elif industry_type == IndustryType.TECH_GROWTH:
            logger.info(f"Using tech growth industry valuation for {ticker}")
            return tech_growth_industry_valuation(
                ticker,
                as_of_date,
                validated,
                risk
            )
        
        elif industry_type == IndustryType.FINANCIAL:
            logger.info(f"Using financial industry valuation for {ticker}")
            if regulatory_data is None:
                logger.warning("Financial industry valuation without regulatory data")
                regulatory_data = pd.DataFrame({
                    "date": pd.date_range(as_of_date - pd.DateOffset(years=2), as_of_date),
                    "capital_requirement": np.random.uniform(0.08, 0.12, 24)
                })
            return financial_industry_valuation(
                ticker,
                as_of_date,
                validated,
                risk,
                regulatory_data
            )
        
        elif industry_type == IndustryType.CONSUMER:
            logger.info(f"Using consumer industry valuation for {ticker}")
            if macro_data is None:
                logger.warning("Consumer industry valuation without macro data")
                macro_data = pd.DataFrame({
                    "date": pd.date_range(as_of_date - pd.DateOffset(years=2), as_of_date),
                    "gdp_growth": np.random.uniform(0.04, 0.07, 24),
                    "cpi": np.random.uniform(0.02, 0.03, 24)
                })
            return consumer_industry_valuation(
                ticker,
                as_of_date,
                validated,
                risk,
                macro_data
            )
        
        else:
            logger.info(f"Using generic industry valuation for {ticker}")
            params = get_industry_specific_params(industry_type)
            base_result = two_stage_dcf(validated, risk)
            scenarios = {
                "基础情景": base_result.intrinsic_value,
                "乐观情景": base_result.intrinsic_value * 1.3,
                "悲观情景": base_result.intrinsic_value * 0.8
            }
            return IndustryValuationResult(
                ticker=ticker,
                as_of_date=as_of_date,
                industry_type=industry_type.value,
                base_value=scenarios["基础情景"],
                scenarios=scenarios,
                assumptions=base_result.assumptions,
                metadata=base_result.metadata
            )
            
    except Exception as e:
        logger.error(f"Industry valuation failed for {ticker}: {str(e)}")
        base_result = two_stage_dcf(validated, risk)
        scenarios = {
            "基础情景": base_result.intrinsic_value,
            "乐观情景": base_result.intrinsic_value * 1.3,
            "悲观情景": base_result.intrinsic_value * 0.8
        }
        return IndustryValuationResult(
            ticker=ticker,
            as_of_date=as_of_date,
            industry_type=industry_type.value,
            base_value=scenarios["基础情景"],
            scenarios=scenarios,
            assumptions=base_result.assumptions,
            metadata=base_result.metadata
        )


__all__ = [
    "IndustryType",
    "ValuationMethod",
    "IndustryValuationResult",
    "IndustrySpecificParams",
    "estimate_industry_type",
    "get_industry_specific_params",
    "cyclical_industry_valuation",
    "tech_growth_industry_valuation",
    "financial_industry_valuation",
    "consumer_industry_valuation",
    "estimate_industry_valuation"
]
