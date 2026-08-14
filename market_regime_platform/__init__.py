"""Market regime detection, strategy evaluation, and backtesting platform."""

from market_regime_platform.backtesting.vectorized import BacktestResult, run_strategy_backtests
from market_regime_platform.data.loader import DataSummary, load_futures_data
from market_regime_platform.regimes.detection import RegimeResult, detect_regimes
from market_regime_platform.strategies.signals import generate_strategy_signals
from market_regime_platform.stress.simulation import StressResult, run_stress_suite

__all__ = [
    "BacktestResult",
    "DataSummary",
    "RegimeResult",
    "StressResult",
    "detect_regimes",
    "generate_strategy_signals",
    "load_futures_data",
    "run_strategy_backtests",
    "run_stress_suite",
]
