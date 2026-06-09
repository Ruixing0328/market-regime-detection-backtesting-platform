"""Market Regime Strategy Backtester package."""

from regime_strategy_backtester.backtesting.vectorized import BacktestResult, run_strategy_backtests
from regime_strategy_backtester.data.loader import DataSummary, load_futures_data
from regime_strategy_backtester.regimes.detection import RegimeResult, detect_regimes
from regime_strategy_backtester.strategies.signals import generate_strategy_signals
from regime_strategy_backtester.stress.simulation import StressResult, run_stress_suite

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
