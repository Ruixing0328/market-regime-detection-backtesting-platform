from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_regime_platform.config import DEFAULT_BARS_PER_YEAR


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    equity_curves: pd.DataFrame
    metrics: pd.DataFrame
    per_regime_metrics: pd.DataFrame
    trades: pd.DataFrame


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak.replace(0.0, np.nan) - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _metrics_from_returns(returns: pd.Series, equity: pd.Series, bars_per_year: float) -> dict[str, float | int]:
    clean = returns.fillna(0.0)
    std = clean.std(ddof=0)
    downside = clean[clean < 0]
    var_95 = float(np.percentile(clean, 5)) if len(clean) else 0.0
    cvar_95 = float(clean[clean <= var_95].mean()) if (clean <= var_95).any() else var_95
    return {
        "bars": int(len(clean)),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
        "expectancy": float(clean.mean()) if len(clean) else 0.0,
        "sharpe": float(np.sqrt(bars_per_year) * clean.mean() / std) if std > 1e-12 else 0.0,
        "sortino": float(np.sqrt(bars_per_year) * clean.mean() / downside.std(ddof=0)) if len(downside) > 1 and downside.std(ddof=0) > 1e-12 else 0.0,
        "max_drawdown": _max_drawdown(equity),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "bar_win_rate": float((clean > 0).mean()) if len(clean) else 0.0,
    }


def _summarize_trades(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if trades.empty:
        empty = trades.copy()
        empty["r_multiple_proxy"] = pd.Series(dtype=float)
        return empty, {"num_trades": 0, "profit_factor": 0.0, "expected_r": 0.0, "win_rate": 0.0}

    enriched = trades.copy()
    pnl = enriched["pnl"].astype(float).fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.abs().sum()) if len(losses) else 0.0
    risk_unit = float(losses.abs().mean()) if len(losses) else 0.0
    enriched["r_multiple_proxy"] = (pnl / risk_unit) if risk_unit > 1e-12 else 0.0
    trade_metrics = {
        "num_trades": int(len(enriched)),
        "win_rate": float((pnl > 0.0).mean()) if len(enriched) else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 1e-12 else (float("inf") if gross_profit > 0 else 0.0),
        "expected_r": float(enriched["r_multiple_proxy"].mean()) if len(enriched) else 0.0,
    }
    return enriched, trade_metrics


def _extract_trades(data: pd.DataFrame, strategy: str, position: pd.Series, strategy_returns: pd.Series, initial_capital: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active_exposure = 0.0
    entry_idx = None
    tolerance = 1e-12
    for idx, exposure in position.astype(float).items():
        exposure = float(exposure)
        if abs(active_exposure) <= tolerance and abs(exposure) > tolerance:
            active_exposure = exposure
            entry_idx = idx
        elif abs(active_exposure) > tolerance and abs(exposure - active_exposure) > tolerance:
            exit_idx = idx
            segment = strategy_returns.loc[entry_idx:exit_idx].fillna(0.0)
            total_return = float((1.0 + segment).prod() - 1.0)
            rows.append(
                {
                    "strategy": strategy,
                    "entry_date": data.loc[entry_idx, "date"],
                    "exit_date": data.loc[exit_idx, "date"],
                    "side": "LONG" if active_exposure > 0 else "SHORT",
                    "entry_exposure": float(active_exposure),
                    "exit_exposure": float(exposure),
                    "entry_price": float(data.loc[entry_idx, "close"]),
                    "exit_price": float(data.loc[exit_idx, "close"]),
                    "return": total_return,
                    "pnl": float(initial_capital * total_return),
                    "bars_held": int(exit_idx - entry_idx + 1),
                    "exit_reason": "signal_flip" if abs(exposure) > tolerance else "signal_exit",
                }
            )
            active_exposure = exposure
            entry_idx = idx if abs(exposure) > tolerance else None

    if abs(active_exposure) > tolerance and entry_idx is not None:
        exit_idx = int(position.index[-1])
        segment = strategy_returns.loc[entry_idx:exit_idx].fillna(0.0)
        total_return = float((1.0 + segment).prod() - 1.0)
        rows.append(
            {
                "strategy": strategy,
                "entry_date": data.loc[entry_idx, "date"],
                "exit_date": data.loc[exit_idx, "date"],
                "side": "LONG" if active_exposure > 0 else "SHORT",
                "entry_exposure": float(active_exposure),
                "exit_exposure": 0.0,
                "entry_price": float(data.loc[entry_idx, "close"]),
                "exit_price": float(data.loc[exit_idx, "close"]),
                "return": total_return,
                "pnl": float(initial_capital * total_return),
                "bars_held": int(exit_idx - entry_idx + 1),
                "exit_reason": "end_of_sample",
            }
        )
    return pd.DataFrame(rows)


def run_strategy_backtests(
    data: pd.DataFrame,
    signals: dict[str, pd.Series],
    regimes: pd.DataFrame | None = None,
    initial_capital: float = 100_000.0,
    bars_per_year: float = DEFAULT_BARS_PER_YEAR,
    transaction_cost_bps: float = 0.25,
) -> BacktestResult:
    """Run no-lookahead vectorized backtests for a dictionary of signals."""

    close_returns = data["close"].astype(float).pct_change().fillna(0.0)
    returns_payload = pd.DataFrame({"date": data["date"]})
    equity_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []

    regime_aligned = None
    if regimes is not None and not regimes.empty:
        regime_aligned = pd.DataFrame(index=data.index)
        regime_aligned["regime_id"] = np.nan
        regime_aligned["regime_name"] = pd.Series(index=data.index, dtype=object)
        regime_aligned.loc[regimes["source_index"].astype(int), "regime_id"] = regimes["regime_id"].to_numpy()
        regime_aligned.loc[regimes["source_index"].astype(int), "regime_name"] = regimes["regime_name"].to_numpy()
        regime_aligned = regime_aligned.ffill()

    for strategy, raw_signal in signals.items():
        signal = raw_signal.reindex(data.index).fillna(0.0).astype(float).clip(-1.0, 1.0)
        position = signal.shift(1).fillna(0.0).astype(float).clip(-1.0, 1.0)
        turnover = position.diff().abs().fillna(position.abs())
        cost = turnover * (transaction_cost_bps / 10_000.0)
        strategy_return = (position * close_returns - cost).fillna(0.0)
        equity = initial_capital * (1.0 + strategy_return).cumprod()

        returns_payload[strategy] = strategy_return
        eq_frame = pd.DataFrame({"date": data["date"], "strategy": strategy, "equity": equity, "return": strategy_return})
        equity_rows.append(eq_frame)

        metrics = _metrics_from_returns(strategy_return, equity, bars_per_year)
        metrics["strategy"] = strategy
        metrics["turnover"] = float(turnover.sum())
        trades = _extract_trades(data.reset_index(drop=True), strategy, position.reset_index(drop=True), strategy_return.reset_index(drop=True), initial_capital)
        trades, trade_metrics = _summarize_trades(trades)
        metrics.update(trade_metrics)
        metric_rows.append(metrics)

        if regime_aligned is not None:
            frame = pd.DataFrame(
                {
                    "return": strategy_return,
                    "equity": equity,
                    "regime_id": regime_aligned["regime_id"],
                    "regime_name": regime_aligned["regime_name"],
                }
            ).dropna(subset=["regime_id"])
            for (regime_id, regime_name), group in frame.groupby(["regime_id", "regime_name"]):
                local_equity = (1.0 + group["return"].fillna(0.0)).cumprod() * initial_capital
                local_metrics = _metrics_from_returns(group["return"], local_equity, bars_per_year)
                local_metrics["win_rate"] = local_metrics.get("bar_win_rate", 0.0)
                local_metrics.update(
                    {
                        "strategy": strategy,
                        "regime_id": int(regime_id),
                        "regime_name": str(regime_name),
                    }
                )
                regime_rows.append(local_metrics)

        trade_frames.append(trades)

    return BacktestResult(
        returns=returns_payload,
        equity_curves=pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame(),
        metrics=pd.DataFrame(metric_rows).sort_values("strategy").reset_index(drop=True),
        per_regime_metrics=pd.DataFrame(regime_rows).sort_values(["strategy", "regime_id"]).reset_index(drop=True) if regime_rows else pd.DataFrame(),
        trades=pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
    )
