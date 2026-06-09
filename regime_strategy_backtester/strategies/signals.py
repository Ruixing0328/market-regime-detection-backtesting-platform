from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_STRATEGIES = ["sma_cross", "donchian", "rsi_mr", "zscore_mr", "ml_rf", "ml_logreg", "ml_gbrt"]
RISK_REGIME_NAMES = {"Trend + High Volatility", "Range + High Volatility"}


def _clip_signal(series: pd.Series) -> pd.Series:
    finite = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return finite.clip(-1.0, 1.0)


def _discrete(series: pd.Series) -> pd.Series:
    return pd.Series(np.sign(_clip_signal(series)).astype(int), index=series.index).clip(-1, 1)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _sma_cross(close: pd.Series, short_window: int, long_window: int) -> pd.Series:
    short_ma = close.rolling(short_window, min_periods=max(5, short_window // 2)).mean()
    long_ma = close.rolling(long_window, min_periods=max(10, long_window // 2)).mean()
    return _discrete(short_ma - long_ma)


def _donchian(data: pd.DataFrame, window: int) -> pd.Series:
    prior_high = data["high"].rolling(window, min_periods=max(10, window // 2)).max().shift(1)
    prior_low = data["low"].rolling(window, min_periods=max(10, window // 2)).min().shift(1)
    signal = pd.Series(0, index=data.index, dtype=float)
    signal.loc[data["close"] > prior_high] = 1.0
    signal.loc[data["close"] < prior_low] = -1.0
    return signal.fillna(0.0)


def _rsi_mean_reversion(close: pd.Series, window: int = 14) -> pd.Series:
    rsi = _rsi(close, window)
    signal = pd.Series(0, index=close.index, dtype=float)
    signal.loc[rsi < 30.0] = 1.0
    signal.loc[rsi > 70.0] = -1.0
    return signal.fillna(0.0)


def _zscore_mean_reversion(close: pd.Series, window: int, threshold: float = 1.25) -> pd.Series:
    mean = close.rolling(window, min_periods=max(10, window // 2)).mean()
    std = close.rolling(window, min_periods=max(10, window // 2)).std(ddof=0)
    zscore = (close - mean) / std.replace(0.0, np.nan)
    signal = pd.Series(0, index=close.index, dtype=float)
    signal.loc[zscore < -threshold] = 1.0
    signal.loc[zscore > threshold] = -1.0
    return signal.fillna(0.0)


def _ml_features(data: pd.DataFrame) -> pd.DataFrame:
    close = data["close"].astype(float)
    log_return = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
    high_low = (data["high"].astype(float) - data["low"].astype(float)) / close.replace(0.0, np.nan)
    volume = data["volume"].astype(float)
    features = pd.DataFrame(
        {
            "ret_1": log_return,
            "ret_5": log_return.rolling(5, min_periods=3).sum(),
            "ret_20": log_return.rolling(20, min_periods=10).sum(),
            "vol_20": log_return.rolling(20, min_periods=10).std(ddof=0),
            "vol_80": log_return.rolling(80, min_periods=20).std(ddof=0),
            "range_pct": high_low,
            "rsi": _rsi(close, 14),
            "volume_z": (volume - volume.rolling(80, min_periods=20).mean()) / volume.rolling(80, min_periods=20).std(ddof=0).replace(0.0, np.nan),
        },
        index=data.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)


def _fit_ml_signal(
    data: pd.DataFrame,
    model_name: str,
    train_fraction: float = 0.60,
    max_train_rows: int = 50_000,
    random_state: int = 181,
) -> pd.Series:
    features = _ml_features(data)
    close = data["close"].astype(float)
    future_return = close.pct_change().shift(-1)
    target = (future_return > 0.0).astype(int)
    valid = features.dropna().index.intersection(target.dropna().index)
    split_pos = int(len(data) * train_fraction)
    train_idx = valid[valid < split_pos]
    test_idx = valid[valid >= split_pos]
    signal = pd.Series(0.0, index=data.index)
    if len(train_idx) < 200 or len(test_idx) < 10 or target.loc[train_idx].nunique() < 2:
        return signal

    if len(train_idx) > max_train_rows:
        take = np.linspace(0, len(train_idx) - 1, max_train_rows).astype(int)
        train_idx = train_idx[take]

    if model_name == "ml_rf":
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_name == "ml_logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state))
    elif model_name == "ml_gbrt":
        model = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=random_state)
    else:
        raise ValueError(f"Unknown ML strategy {model_name!r}")

    model.fit(features.loc[train_idx], target.loc[train_idx])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features.loc[test_idx])[:, 1]
        predicted = np.where(proba > 0.55, 1.0, np.where(proba < 0.45, -1.0, 0.0))
    else:
        predicted = np.where(model.predict(features.loc[test_idx]) > 0, 1.0, -1.0)
    signal.loc[test_idx] = predicted
    return signal.fillna(0.0)


def _liquidity_filter(data: pd.DataFrame, window: int) -> pd.Series:
    volume = data["volume"].astype(float)
    bar_range = (data["high"].astype(float) - data["low"].astype(float)).abs()
    median_volume = volume.rolling(window, min_periods=max(10, window // 2)).median()
    median_range = bar_range.rolling(window, min_periods=max(10, window // 2)).median()
    liquid = (volume >= median_volume) & (bar_range >= median_range)
    return liquid.fillna(False).astype(float)


def _risk_regime_multiplier(data: pd.DataFrame, regimes: pd.DataFrame | None) -> pd.Series:
    multiplier = pd.Series(1.0, index=data.index)
    if regimes is None or regimes.empty:
        return multiplier
    aligned = pd.Series(index=data.index, dtype=object)
    source_index = regimes["source_index"].astype(int)
    aligned.loc[source_index] = regimes["regime_name"].to_numpy()
    aligned = aligned.ffill()
    multiplier.loc[aligned.isin(RISK_REGIME_NAMES)] = 0.5
    return multiplier


def _base_signals(
    data: pd.DataFrame,
    short_window: int,
    long_window: int,
    breakout_window: int,
    mean_reversion_window: int,
) -> dict[str, pd.Series]:
    close = data["close"].astype(float)
    return {
        "sma_cross": _sma_cross(close, short_window, long_window),
        "donchian": _donchian(data, breakout_window),
        "rsi_mr": _rsi_mean_reversion(close, 14),
        "zscore_mr": _zscore_mean_reversion(close, mean_reversion_window),
        "ml_rf": _fit_ml_signal(data, "ml_rf"),
        "ml_logreg": _fit_ml_signal(data, "ml_logreg"),
        "ml_gbrt": _fit_ml_signal(data, "ml_gbrt"),
    }


def generate_strategy_signals(
    data: pd.DataFrame,
    regimes: pd.DataFrame | None = None,
    short_window: int = 20,
    long_window: int = 80,
    breakout_window: int = 60,
    mean_reversion_window: int = 80,
    selected_strategies: list[str] | None = None,
) -> dict[str, pd.Series]:
    """Generate baseline, regime-aware, and liquidity-filtered strategy variants."""

    requested = set(selected_strategies or BASE_STRATEGIES)
    unknown = requested - set(BASE_STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies requested: {sorted(unknown)}")

    base = _base_signals(data, short_window, long_window, breakout_window, mean_reversion_window)
    risk_multiplier = _risk_regime_multiplier(data, regimes)
    liquid_multiplier = _liquidity_filter(data, breakout_window)
    signals: dict[str, pd.Series] = {}

    for name in BASE_STRATEGIES:
        if name not in requested:
            continue
        baseline = _clip_signal(base[name]).reindex(data.index).fillna(0.0)
        signals[f"{name}__baseline"] = baseline
        signals[f"{name}__regime_aware"] = _clip_signal(baseline * risk_multiplier)
        signals[f"{name}__liquidity_filtered"] = _clip_signal(baseline * liquid_multiplier)

    return {name: series.reindex(data.index).fillna(0.0).clip(-1.0, 1.0) for name, series in signals.items()}
