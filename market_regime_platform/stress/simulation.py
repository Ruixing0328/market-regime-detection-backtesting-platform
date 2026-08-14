from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


STRESS_MODELS = ["gbm", "regime_mixture_gbm", "block_bootstrap"]


@dataclass
class StressResult:
    summary: pd.DataFrame
    paths: pd.DataFrame


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / np.maximum(peak, 1e-12) - 1.0
    return float(drawdown.min()) if drawdown.size else 0.0


def _path_frame(
    strategy: str,
    model: str,
    regime_filter: str,
    paths: np.ndarray,
    initial_capital: float,
    max_saved_paths: int,
) -> pd.DataFrame:
    if paths.size == 0 or max_saved_paths <= 0:
        return pd.DataFrame()
    saved = paths[: min(max_saved_paths, len(paths))]
    rows = []
    for sim, returns in enumerate(saved):
        equity = initial_capital * np.cumprod(1.0 + returns)
        rows.append(
            pd.DataFrame(
                {
                    "strategy": strategy,
                    "stress_model": model,
                    "regime_filter": regime_filter,
                    "simulation": sim,
                    "step": np.arange(1, len(returns) + 1),
                    "return": returns,
                    "equity": equity,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _summarize(
    strategy: str,
    model: str,
    regime_filter: str,
    paths: np.ndarray,
    initial_capital: float,
    drawdown_threshold: float,
) -> dict[str, object]:
    if paths.size == 0:
        terminal = np.array([], dtype=float)
        terminal_equity = np.array([], dtype=float)
        max_dd = np.array([], dtype=float)
    else:
        terminal = np.prod(1.0 + paths, axis=1) - 1.0
        terminal_equity = initial_capital * (1.0 + terminal)
        max_dd = np.array([_max_drawdown_from_returns(path) for path in paths])
    var_95 = float(np.percentile(terminal, 5)) if terminal.size else 0.0
    equity_p5 = float(np.percentile(terminal_equity, 5)) if terminal_equity.size else initial_capital
    return {
        "strategy": strategy,
        "stress_model": model,
        "regime_filter": regime_filter,
        "simulations": int(len(terminal)),
        "horizon": int(paths.shape[1]) if paths.ndim == 2 else 0,
        "terminal_mean": float(np.mean(terminal)) if terminal.size else 0.0,
        "terminal_std": float(np.std(terminal)) if terminal.size else 0.0,
        "terminal_p5": var_95,
        "terminal_p50": float(np.percentile(terminal, 50)) if terminal.size else 0.0,
        "terminal_p95": float(np.percentile(terminal, 95)) if terminal.size else 0.0,
        "terminal_cvar_95": float(np.mean(terminal[terminal <= var_95])) if (terminal <= var_95).any() else var_95,
        "terminal_equity_p5": equity_p5,
        "terminal_equity_p50": float(np.percentile(terminal_equity, 50)) if terminal_equity.size else initial_capital,
        "terminal_equity_p95": float(np.percentile(terminal_equity, 95)) if terminal_equity.size else initial_capital,
        "max_drawdown_p50": float(np.percentile(max_dd, 50)) if max_dd.size else 0.0,
        "max_drawdown_p95": float(np.percentile(max_dd, 5)) if max_dd.size else 0.0,
        "probability_of_loss": float((terminal < 0.0).mean()) if terminal.size else 0.0,
        "prob_drawdown_exceed": float((max_dd <= -abs(drawdown_threshold)).mean()) if max_dd.size else 0.0,
        "risk_of_ruin": float((terminal_equity <= initial_capital * 0.5).mean()) if terminal_equity.size else 0.0,
    }


def _simulate_gbm(returns: np.ndarray, simulations: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    clean = returns[np.isfinite(returns)]
    if clean.size == 0:
        clean = np.array([0.0])
    mu = float(np.nanmean(clean))
    sigma = float(np.nanstd(clean))
    return rng.normal(mu, sigma, size=(simulations, horizon))


def _simulate_regime_mixture(
    returns: pd.Series,
    regimes: pd.Series | None,
    simulations: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if regimes is None or regimes.dropna().empty:
        return _simulate_gbm(returns.to_numpy(dtype=float), simulations, horizon, rng)

    frame = pd.DataFrame({"return": returns, "regime": regimes}).dropna()
    if frame.empty:
        return _simulate_gbm(returns.to_numpy(dtype=float), simulations, horizon, rng)

    grouped = {str(regime): group["return"].to_numpy(dtype=float) for regime, group in frame.groupby("regime")}
    regime_values = np.array(list(grouped.keys()))
    probabilities = frame["regime"].astype(str).value_counts(normalize=True).reindex(regime_values).to_numpy(dtype=float)
    probabilities = probabilities / probabilities.sum()

    paths = np.empty((simulations, horizon), dtype=float)
    for sim in range(simulations):
        sampled_regimes = rng.choice(regime_values, size=horizon, p=probabilities)
        for step, regime in enumerate(sampled_regimes):
            pool = grouped[str(regime)]
            paths[sim, step] = rng.choice(pool) if len(pool) else 0.0
    return paths


def _simulate_block_bootstrap(returns: np.ndarray, simulations: int, horizon: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    clean = returns[np.isfinite(returns)]
    if len(clean) == 0:
        clean = np.array([0.0])
    block_size = max(1, min(block_size, len(clean)))
    max_start = max(len(clean) - block_size, 0)
    paths = np.empty((simulations, horizon), dtype=float)
    for sim in range(simulations):
        chunks = []
        while sum(len(chunk) for chunk in chunks) < horizon:
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            chunks.append(clean[start : start + block_size])
        paths[sim] = np.concatenate(chunks)[:horizon]
    return paths


def _aligned_regime_names(strategy_returns: pd.DataFrame, regimes: pd.DataFrame | None) -> pd.Series | None:
    if regimes is None or regimes.empty:
        return None
    regime_series = pd.Series(index=strategy_returns.index, dtype=object)
    regime_series.loc[regimes["source_index"].astype(int)] = regimes["regime_name"].to_numpy()
    return regime_series.ffill()


def _expand_regime_filters(regime_series: pd.Series | None, requested: list[str] | None) -> list[str]:
    if not requested:
        requested = ["all"]
    out: list[str] = []
    available = sorted(regime_series.dropna().astype(str).unique().tolist()) if regime_series is not None else []
    for item in requested:
        if item == "each":
            out.extend(available)
        elif item == "all":
            out.append("all")
        elif item in available:
            out.append(item)
    deduped: list[str] = []
    for item in out:
        if item not in deduped:
            deduped.append(item)
    return deduped or ["all"]


def run_stress_suite(
    strategy_returns: pd.DataFrame,
    regimes: pd.DataFrame | None = None,
    simulations: int = 500,
    horizon: int = 390,
    block_size: int = 30,
    seed: int = 181,
    stress_models: list[str] | None = None,
    regime_filters: list[str] | None = None,
    initial_capital: float = 100_000.0,
    max_saved_paths: int = 50,
    drawdown_threshold: float = 0.20,
) -> StressResult:
    """Run GBM, regime-mixture, and block-bootstrap stress tests by strategy."""

    rng = np.random.default_rng(seed)
    strategy_columns = [column for column in strategy_returns.columns if column != "date"]
    model_list = stress_models or STRESS_MODELS
    unknown_models = set(model_list) - set(STRESS_MODELS)
    if unknown_models:
        raise ValueError(f"Unknown stress models requested: {sorted(unknown_models)}")

    regime_series = _aligned_regime_names(strategy_returns, regimes)
    filter_list = _expand_regime_filters(regime_series, regime_filters)
    summary_rows: list[dict[str, object]] = []
    path_rows: list[pd.DataFrame] = []

    for strategy in strategy_columns:
        base_returns = strategy_returns[strategy].fillna(0.0)
        for regime_filter in filter_list:
            if regime_filter == "all" or regime_series is None:
                returns = base_returns
                local_regimes = regime_series
            else:
                mask = regime_series == regime_filter
                returns = base_returns.loc[mask]
                local_regimes = regime_series.loc[mask]
            if returns.dropna().empty:
                continue

            for model in model_list:
                if model == "gbm":
                    paths = _simulate_gbm(returns.to_numpy(dtype=float), simulations, horizon, rng)
                elif model == "regime_mixture_gbm":
                    paths = _simulate_regime_mixture(returns, local_regimes, simulations, horizon, rng)
                elif model == "block_bootstrap":
                    paths = _simulate_block_bootstrap(returns.to_numpy(dtype=float), simulations, horizon, block_size, rng)
                else:
                    raise ValueError(f"Unknown stress model {model!r}")

                summary_rows.append(_summarize(strategy, model, regime_filter, paths, initial_capital, drawdown_threshold))
                path_frame = _path_frame(strategy, model, regime_filter, paths, initial_capital, max_saved_paths)
                if not path_frame.empty:
                    path_rows.append(path_frame)

    return StressResult(
        summary=pd.DataFrame(summary_rows).sort_values(["strategy", "regime_filter", "stress_model"]).reset_index(drop=True)
        if summary_rows
        else pd.DataFrame(),
        paths=pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame(),
    )
