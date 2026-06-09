from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from regime_strategy_backtester.regimes.detection import REGIME_NAMES, RegimeResult


@dataclass
class WalkForwardConfig:
    train_bars: int = 120_000
    test_bars: int = 20_000
    step_bars: int | None = None
    min_train_bars: int = 50_000


@dataclass
class _FitResult:
    candidate_name: str
    model_family: str
    feature_cols: list[str]
    scaler: StandardScaler
    model: object
    quadrant_mapping: dict[int, int]
    axis_cluster_stats: pd.DataFrame
    train_rows: int


def _base_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.reset_index(drop=True).copy()
    frame["source_index"] = frame.index.astype(int)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame


def _safe_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _generated_features(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    frame = _base_frame(data)
    close = frame["close"].astype(float)
    log_return = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
    rolling_vol_short = log_return.rolling(short_window, min_periods=max(10, short_window // 4)).std(ddof=0)
    rolling_vol_long = log_return.rolling(long_window, min_periods=max(20, long_window // 4)).std(ddof=0)
    abs_return = log_return.abs().rolling(short_window, min_periods=max(10, short_window // 4)).mean()
    trend_strength = _safe_zscore(close, long_window).abs()
    momentum = close.pct_change(long_window)
    range_pct = (frame["high"].astype(float) - frame["low"].astype(float)) / close.replace(0.0, np.nan)
    features = pd.DataFrame(
        {
            "source_index": frame["source_index"],
            "date": frame["date"],
            "log_return": log_return,
            "rolling_vol_short": rolling_vol_short,
            "rolling_vol_long": rolling_vol_long,
            "abs_return": abs_return,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "range_pct": range_pct,
        }
    )
    features["vol_axis"] = features[["rolling_vol_long", "abs_return", "range_pct"]].mean(axis=1)
    features["structure_axis"] = features[["trend_strength", "momentum"]].abs().mean(axis=1)
    return features


def _scripted_features(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    frame = _base_frame(data)
    close = frame["close"].astype(float)
    log_return = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
    fast = max(short_window * 2, 60)
    slow = max(long_window * 2, 240)
    vol_fast = log_return.rolling(fast, min_periods=max(20, fast // 4)).std(ddof=0)
    vol_slow = log_return.rolling(slow, min_periods=max(30, slow // 4)).std(ddof=0)
    absret_mean = log_return.abs().rolling(fast, min_periods=max(20, fast // 4)).mean()
    trend_strength = close.pct_change(slow).abs()
    directional_persistence = np.sign(log_return).rolling(fast, min_periods=max(20, fast // 4)).mean().abs()
    range_pct = (frame["high"].astype(float) - frame["low"].astype(float)) / close.replace(0.0, np.nan)
    features = pd.DataFrame(
        {
            "source_index": frame["source_index"],
            "date": frame["date"],
            "log_return": log_return,
            "script_vol_fast": vol_fast,
            "script_vol_slow": vol_slow,
            "script_absret_mean": absret_mean,
            "script_trend_strength": trend_strength,
            "script_directional_persistence": directional_persistence,
            "script_range_pct": range_pct,
        }
    )
    features["vol_axis"] = features[["script_vol_slow", "script_absret_mean", "script_range_pct"]].mean(axis=1)
    features["structure_axis"] = features[["script_trend_strength", "script_directional_persistence"]].abs().mean(axis=1)
    return features


def _resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    frame = _base_frame(data).set_index("date")
    ohlcv = frame[["open", "high", "low", "close", "volume"]].astype(float)
    out = ohlcv.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def _tf_feature_table(ohlcv: pd.DataFrame, prefix: str, windows: tuple[int, int]) -> pd.DataFrame:
    close = ohlcv["close"].astype(float)
    log_return = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
    short, long = windows
    min_short = max(2, short // 3)
    min_long = max(3, long // 3)
    range_pct = (ohlcv["high"].astype(float) - ohlcv["low"].astype(float)) / close.replace(0.0, np.nan)
    feats = pd.DataFrame(
        {
            "date": ohlcv.index,
            f"{prefix}_vol_short": log_return.rolling(short, min_periods=min_short).std(ddof=0),
            f"{prefix}_vol_long": log_return.rolling(long, min_periods=min_long).std(ddof=0),
            f"{prefix}_trend_short": log_return.rolling(short, min_periods=min_short).sum(),
            f"{prefix}_trend_long": log_return.rolling(long, min_periods=min_long).sum(),
            f"{prefix}_range_pct": range_pct,
        }
    )
    value_cols = [col for col in feats.columns if col != "date"]
    feats[value_cols] = feats[value_cols].shift(1)
    return feats.reset_index(drop=True)


def _merge_asof_features(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return base
    left = base.sort_values("date")
    right = features.sort_values("date")
    return pd.merge_asof(left, right, on="date", direction="backward")


def _notebook_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = _base_frame(data)
    base = frame[["source_index", "date"]].copy()
    five_min = _tf_feature_table(_resample_ohlcv(data, "5min"), "i5", (12, 288))
    daily = _tf_feature_table(_resample_ohlcv(data, "1D"), "daily", (5, 60))
    weekly = _tf_feature_table(_resample_ohlcv(data, "W-FRI"), "weekly", (4, 26))
    features = _merge_asof_features(base, five_min)
    features = _merge_asof_features(features, daily)
    features = _merge_asof_features(features, weekly)

    vol_cols = [col for col in features.columns if col.endswith(("vol_short", "vol_long", "range_pct"))]
    trend_cols = [col for col in features.columns if "trend_" in col]
    features["vol_axis"] = features[vol_cols].mean(axis=1) if vol_cols else np.nan
    features["structure_axis"] = features[trend_cols].abs().mean(axis=1) if trend_cols else np.nan
    return features


def _candidate_frames(data: pd.DataFrame, short_window: int, long_window: int) -> dict[str, pd.DataFrame]:
    generated = _generated_features(data, short_window, long_window)
    scripted = _scripted_features(data, short_window, long_window)
    notebook = _notebook_features(data)
    hybrid = generated.merge(
        notebook.drop(columns=["vol_axis", "structure_axis"], errors="ignore"),
        on=["source_index", "date"],
        how="left",
    )
    hybrid["vol_axis"] = hybrid[[col for col in hybrid.columns if "vol" in col or col.endswith("range_pct")]].mean(axis=1)
    hybrid["structure_axis"] = hybrid[
        [col for col in hybrid.columns if "trend" in col or "momentum" in col or "directional" in col]
    ].abs().mean(axis=1)
    return {
        "minute_volatility_trend_kmeans": generated,
        "rolling_return_volatility_kmeans": scripted,
        "multi_timeframe_volatility_trend_kmeans": notebook,
        "hybrid_minute_multi_timeframe_kmeans": hybrid,
    }


def _model_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"date", "source_index", "log_return"}
    numeric_cols = [col for col in features.columns if col not in excluded and pd.api.types.is_numeric_dtype(features[col])]
    usable: list[str] = []
    for col in numeric_cols:
        series = features[col].replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() >= 20 and float(series.std(ddof=0) or 0.0) > 1e-12:
            usable.append(col)
    for required in ("vol_axis", "structure_axis"):
        if required in usable:
            usable.remove(required)
        usable.insert(0, required)
    return [col for col in usable if col in features.columns]


def _assign_quadrant_labels(raw_labels: np.ndarray, features: pd.DataFrame) -> tuple[np.ndarray, dict[int, int], pd.DataFrame]:
    stats = (
        pd.DataFrame(
            {
                "raw_label": raw_labels,
                "vol_axis": features["vol_axis"].to_numpy(dtype=float),
                "structure_axis": features["structure_axis"].to_numpy(dtype=float),
            }
        )
        .groupby("raw_label", as_index=False)
        .agg(bars=("raw_label", "size"), vol_axis=("vol_axis", "mean"), structure_axis=("structure_axis", "mean"))
        .sort_values(["vol_axis", "structure_axis", "raw_label"])
        .reset_index(drop=True)
    )
    if len(stats) != 4:
        raise ValueError("Regime model must produce exactly four non-empty clusters")

    low_vol = stats.iloc[:2].sort_values(["structure_axis", "raw_label"])
    high_vol = stats.iloc[2:].sort_values(["structure_axis", "raw_label"])
    mapping = {
        int(low_vol.iloc[0]["raw_label"]): 0,
        int(low_vol.iloc[1]["raw_label"]): 1,
        int(high_vol.iloc[1]["raw_label"]): 2,
        int(high_vol.iloc[0]["raw_label"]): 3,
    }
    labels = np.array([mapping[int(label)] for label in raw_labels], dtype=int)
    stats["regime_id"] = stats["raw_label"].map(mapping).astype(int)
    stats["regime_name"] = stats["regime_id"].map(dict(enumerate(REGIME_NAMES)))
    return labels, mapping, stats.sort_values("regime_id").reset_index(drop=True)


def _transition_matrix(label_series: pd.Series) -> pd.DataFrame:
    transitions = pd.DataFrame({"from": label_series.shift(1), "to": label_series}).dropna()
    if transitions.empty:
        return pd.DataFrame()
    table = pd.crosstab(transitions["from"].astype(int), transitions["to"].astype(int), normalize="index")
    table.index.name = "from_regime"
    table.columns.name = "to_regime"
    return table.reset_index()


def _stability(labels: pd.DataFrame, window: int) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    chunks: list[dict[str, object]] = []
    ordered = labels.sort_values("date").reset_index(drop=True)
    for start in range(0, len(ordered), window):
        chunk = ordered.iloc[start : start + window]
        if chunk.empty:
            continue
        counts = chunk["regime_id"].value_counts(normalize=True).reindex(range(4), fill_value=0.0).sort_index()
        row: dict[str, object] = {
            "window_start": chunk["date"].iloc[0],
            "window_end": chunk["date"].iloc[-1],
            "bars": int(len(chunk)),
        }
        for regime, value in counts.items():
            row[f"regime_{int(regime)}_share"] = float(value)
        chunks.append(row)
    return pd.DataFrame(chunks)


def _build_stats(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    numeric_cols = [
        col for col in features.columns if col not in {"source_index", "date"} and pd.api.types.is_numeric_dtype(features[col])
    ]
    grouped = features.assign(regime_id=labels).groupby("regime_id")
    stats = grouped[numeric_cols].mean().add_suffix("_mean")
    stats.insert(0, "bars", grouped.size())
    stats = stats.reset_index().sort_values("regime_id")
    stats["regime_name"] = stats["regime_id"].map(dict(enumerate(REGIME_NAMES)))
    if "rolling_vol_long_mean" not in stats.columns and "vol_axis_mean" in stats.columns:
        stats["rolling_vol_long_mean"] = stats["vol_axis_mean"]
    if "trend_strength_mean" not in stats.columns and "structure_axis_mean" in stats.columns:
        stats["trend_strength_mean"] = stats["structure_axis_mean"]
    return stats


def _silhouette(scaled: np.ndarray, labels: np.ndarray) -> float:
    sample_rows = min(len(scaled), 10_000)
    if len(np.unique(labels)) <= 1 or sample_rows < 4:
        return 0.0
    sample_idx = np.linspace(0, len(scaled) - 1, sample_rows).astype(int)
    return float(silhouette_score(scaled[sample_idx], labels[sample_idx]))


def _candidate_internal_score(scaled: np.ndarray, labels: np.ndarray, features: pd.DataFrame) -> dict[str, float]:
    proportions = pd.Series(labels).value_counts(normalize=True).reindex(range(4), fill_value=0.0).to_numpy(dtype=float)
    balance = float(max(0.0, min(1.0, proportions.min() / 0.10)))
    persistence = float((pd.Series(labels).shift(1) == pd.Series(labels)).mean())
    axis = (
        features.assign(regime_id=labels)
        .groupby("regime_id")
        .agg(vol_axis=("vol_axis", "mean"), structure_axis=("structure_axis", "mean"))
        .reindex(range(4))
    )
    low_vol = axis.loc[[0, 1], "vol_axis"].mean()
    high_vol = axis.loc[[2, 3], "vol_axis"].mean()
    range_structure = axis.loc[[0, 3], "structure_axis"].mean()
    trend_structure = axis.loc[[1, 2], "structure_axis"].mean()
    vol_sep = float((high_vol - low_vol) / (axis["vol_axis"].std(ddof=0) + 1e-12))
    structure_sep = float((trend_structure - range_structure) / (axis["structure_axis"].std(ddof=0) + 1e-12))
    interpretability = float(max(0.0, min(2.0, vol_sep + structure_sep)) / 2.0)
    return {
        "silhouette_score": _silhouette(scaled, labels),
        "cluster_balance_score": balance,
        "transition_persistence": persistence,
        "interpretability_score": interpretability,
    }


def _hmm_available() -> bool:
    return importlib.util.find_spec("hmmlearn") is not None


def _fit_kmeans_quadrant(features: pd.DataFrame, feature_cols: list[str], n_regimes: int, random_state: int) -> _FitResult:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[feature_cols])
    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
    raw = km.fit_predict(scaled)
    labels, mapping, axis_stats = _assign_quadrant_labels(raw, features)
    return _FitResult(
        candidate_name="",
        model_family="kmeans_quadrant",
        feature_cols=feature_cols,
        scaler=scaler,
        model=km,
        quadrant_mapping=mapping,
        axis_cluster_stats=axis_stats,
        train_rows=int(len(features)),
    )


def _fit_gaussian_hmm_quadrant(features: pd.DataFrame, feature_cols: list[str], n_regimes: int, random_state: int) -> _FitResult:
    if not _hmm_available():
        raise RuntimeError("hmmlearn not installed")
    from hmmlearn.hmm import GaussianHMM

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[feature_cols])
    hmm = GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=200,
        random_state=random_state,
    )
    hmm.fit(scaled)
    raw = hmm.predict(scaled)
    labels, mapping, axis_stats = _assign_quadrant_labels(raw, features)
    return _FitResult(
        candidate_name="",
        model_family="gaussian_hmm_quadrant",
        feature_cols=feature_cols,
        scaler=scaler,
        model=hmm,
        quadrant_mapping=mapping,
        axis_cluster_stats=axis_stats,
        train_rows=int(len(features)),
    )


def _prepare_features(raw_features: pd.DataFrame, n_regimes: int) -> tuple[pd.DataFrame, list[str]] | None:
    features = raw_features.replace([np.inf, -np.inf], np.nan).copy()
    feature_cols = _model_columns(features)
    if n_regimes != 4 or len(feature_cols) < 3:
        return None
    features = features.dropna(subset=feature_cols + ["vol_axis", "structure_axis"]).reset_index(drop=True)
    if len(features) < max(n_regimes * 30, 120):
        return None
    return features, feature_cols


def _predict_with_fit(fit: _FitResult, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    scaled = fit.scaler.transform(features[fit.feature_cols])
    if fit.model_family == "kmeans_quadrant":
        raw = fit.model.predict(scaled)  # type: ignore[union-attr]
        probs = None
    elif fit.model_family == "gaussian_hmm_quadrant":
        raw = fit.model.predict(scaled)  # type: ignore[union-attr]
        try:
            probs = fit.model.predict_proba(scaled)  # type: ignore[union-attr]
        except Exception:
            probs = None
    else:
        raise ValueError(f"Unknown model family {fit.model_family!r}")
    labels = np.array([fit.quadrant_mapping[int(state)] for state in raw], dtype=int)
    return labels, probs


def _make_labels_frame(
    features: pd.DataFrame,
    labels: np.ndarray,
    model_name: str,
    probs: np.ndarray | None,
    raw_state_to_regime_id: dict[int, int] | None = None,
) -> pd.DataFrame:
    names = dict(enumerate(REGIME_NAMES))
    out = pd.DataFrame(
        {
            "source_index": features["source_index"].astype(int),
            "date": features["date"],
            "regime_id": labels.astype(int),
            "regime_name": pd.Series(labels).map(names).to_numpy(),
            "regime_model": model_name,
        }
    )
    if probs is None:
        for regime_id in range(4):
            out[f"prob_regime_{regime_id}"] = (out["regime_id"] == regime_id).astype(float)
        return out

    if raw_state_to_regime_id is None:
        for regime_id in range(4):
            out[f"prob_regime_{regime_id}"] = (out["regime_id"] == regime_id).astype(float)
        return out

    # probs are by hidden-state id; map to regime_id then aggregate.
    mapped = np.zeros((len(out), 4), dtype=float)
    for raw_state, regime_id in raw_state_to_regime_id.items():
        mapped[:, regime_id] += probs[:, raw_state]
    for regime_id in range(4):
        out[f"prob_regime_{regime_id}"] = mapped[:, regime_id]
    return out


def _fit_one(
    candidate_name: str,
    model_family: Literal["kmeans_quadrant", "gaussian_hmm_quadrant"],
    raw_features: pd.DataFrame,
    n_regimes: int,
    random_state: int,
) -> tuple[_FitResult, pd.DataFrame, np.ndarray, dict[str, float]]:
    prepared = _prepare_features(raw_features, n_regimes)
    if prepared is None:
        raise ValueError("insufficient usable feature rows/cols")
    features, feature_cols = prepared
    if model_family == "kmeans_quadrant":
        fit = _fit_kmeans_quadrant(features, feature_cols, n_regimes, random_state)
    else:
        fit = _fit_gaussian_hmm_quadrant(features, feature_cols, n_regimes, random_state)
    fit.candidate_name = candidate_name
    labels, _ = _predict_with_fit(fit, features)
    scaled = fit.scaler.transform(features[fit.feature_cols])
    diagnostics = _candidate_internal_score(scaled, labels, features)
    return fit, features, labels, diagnostics


def _tournament_in_sample(
    data: pd.DataFrame,
    n_regimes: int,
    short_window: int,
    long_window: int,
    random_state: int,
    include_hmm: bool,
) -> tuple[_FitResult, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    candidates = _candidate_frames(data, short_window, long_window)
    evaluated: list[dict[str, object]] = []
    best: tuple[float, _FitResult, pd.DataFrame, np.ndarray, dict[str, float]] | None = None
    skipped: dict[str, str] = {}

    model_families: list[str] = ["kmeans_quadrant"]
    if include_hmm and _hmm_available():
        model_families.append("gaussian_hmm_quadrant")

    for candidate_name, raw_feats in candidates.items():
        for family in model_families:
            key = f"{candidate_name}::{family}"
            try:
                fit, features, labels, diag = _fit_one(candidate_name, family, raw_feats, n_regimes, random_state)
            except Exception as exc:
                skipped[key] = str(exc)
                continue
            # Use the same basic heuristic score as baseline for quick iteration;
            # OOS scoring is done in `regime_strategy_backtester.regimes.validation`.
            score = (
                float(diag["silhouette_score"]) * 0.45
                + float(diag["cluster_balance_score"]) * 0.25
                + float(diag["transition_persistence"]) * 0.10
                + float(diag["interpretability_score"]) * 0.20
            )
            evaluated.append(
                {
                    "candidate": candidate_name,
                    "model_family": family,
                    "score": float(score),
                    **{k: float(v) for k, v in diag.items()},
                    "train_rows": int(fit.train_rows),
                }
            )
            if best is None or score > best[0]:
                best = (score, fit, features, labels, diag)

    if best is None:
        raise ValueError("No candidate produced enough usable data")

    _, fit, features, labels, _ = best
    model_name = f"{fit.candidate_name}::{fit.model_family}"
    diagnostics: dict[str, object] = {
        "model_name": model_name,
        "selected_model": model_name,
        "n_regimes": int(n_regimes),
        "regime_names": REGIME_NAMES,
        "short_window": int(short_window),
        "long_window": int(long_window),
        "selection_method": "dev in-sample tournament (use validation.py for OOS selection)",
        "candidate_scores": evaluated,
        "skipped_candidates": skipped,
        "hmm_status": "available" if (include_hmm and _hmm_available()) else ("not_requested" if not include_hmm else "unavailable"),
    }
    labels_frame = _make_labels_frame(features, labels, model_name, probs=None)
    return fit, features, labels_frame, diagnostics


def _walk_forward_slices(total_bars: int, config: WalkForwardConfig) -> list[tuple[int, int, int, int]]:
    step = int(config.step_bars or config.test_bars)
    train = int(config.train_bars)
    test = int(config.test_bars)
    slices: list[tuple[int, int, int, int]] = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train
        test_start = train_end
        test_end = test_start + test
        if test_end > total_bars:
            break
        slices.append((train_start, train_end, test_start, test_end))
        start += step
    return slices


def detect_regimes_dev(
    data: pd.DataFrame,
    n_regimes: int = 4,
    short_window: int = 30,
    long_window: int = 120,
    random_state: int = 181,
    include_hmm: bool = True,
    stability_window: int = 2_000,
    mode: Literal["in_sample", "walk_forward"] = "in_sample",
    walk_forward: WalkForwardConfig | None = None,
) -> RegimeResult:
    """
    Dev regime detector.

    - `mode="in_sample"`: fast tournament on the full history (like production, but includes optional HMM competitor)
    - `mode="walk_forward"`: produces OOS labels by repeated fit-on-train → label-test.
      The model/candidate selection for walk-forward is delegated to `regime_strategy_backtester.regimes.validation`.
    """
    if n_regimes != 4:
        raise ValueError("Regime detection requires exactly four regimes")
    if len(data) < max(long_window + 5, n_regimes * 30):
        raise ValueError("Not enough rows to detect regimes with the requested windows")

    if mode == "in_sample":
        _, features, labels, diagnostics = _tournament_in_sample(
            data=data,
            n_regimes=n_regimes,
            short_window=short_window,
            long_window=long_window,
            random_state=random_state,
            include_hmm=include_hmm,
        )
        stats = _build_stats(features, labels["regime_id"].to_numpy(dtype=int))
        return RegimeResult(
            labels=labels.sort_values("date").reset_index(drop=True),
            features=features.sort_values("date").reset_index(drop=True),
            regime_stats=stats,
            transition_matrix=_transition_matrix(labels.sort_values("date")["regime_id"]),
            stability=_stability(labels, stability_window),
            diagnostics=diagnostics,
            model_name=str(diagnostics["model_name"]),
            hmm_status=str(diagnostics.get("hmm_status", "not_requested")),
        )

    if walk_forward is None:
        walk_forward = WalkForwardConfig()

    # Walk-forward mode is implemented in validation.py so it can score OOS and select a winner.
    from regime_strategy_backtester.regimes.validation import walk_forward_detect_regimes

    return walk_forward_detect_regimes(
        data=data,
        n_regimes=n_regimes,
        short_window=short_window,
        long_window=long_window,
        random_state=random_state,
        include_hmm=include_hmm,
        stability_window=stability_window,
        config=walk_forward,
    )
