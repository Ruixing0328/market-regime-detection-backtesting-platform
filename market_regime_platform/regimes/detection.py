from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get("LOKY_MAX_CPU_COUNT") or "1"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


REGIME_NAMES = [
    "Range + Low Volatility",
    "Trend + Low Volatility",
    "Trend + High Volatility",
    "Range + High Volatility",
]

@dataclass
class RegimeResult:
    labels: pd.DataFrame
    features: pd.DataFrame
    regime_stats: pd.DataFrame
    transition_matrix: pd.DataFrame
    stability: pd.DataFrame
    diagnostics: dict[str, object]
    model_name: str
    hmm_status: str = "not_requested"


@dataclass
class _CandidateResult:
    name: str
    features: pd.DataFrame
    feature_cols: list[str]
    labels: np.ndarray
    raw_labels: np.ndarray
    score: float
    diagnostics: dict[str, object]
    kmeans_inertia: float


def _safe_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _base_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.reset_index(drop=True).copy()
    frame["source_index"] = frame.index.astype(int)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame


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
    # Shift completed higher-timeframe context so minute-level labels never use
    # the bar that is still forming.
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
    numeric_cols = [
        col
        for col in features.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(features[col])
    ]
    usable = []
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
    chunks = []
    ordered = labels.sort_values("date").reset_index(drop=True)
    for start in range(0, len(ordered), window):
        chunk = ordered.iloc[start : start + window]
        if chunk.empty:
            continue
        counts = chunk["regime_id"].value_counts(normalize=True).reindex(range(4), fill_value=0.0).sort_index()
        row = {
            "window_start": chunk["date"].iloc[0],
            "window_end": chunk["date"].iloc[-1],
            "bars": int(len(chunk)),
        }
        for regime, value in counts.items():
            row[f"regime_{int(regime)}_share"] = float(value)
        chunks.append(row)
    return pd.DataFrame(chunks)


def _stability_score(labels: np.ndarray, window: int) -> float:
    if len(labels) < max(window, 20):
        return 0.5
    shares = []
    for start in range(0, len(labels), window):
        chunk = labels[start : start + window]
        if len(chunk) < max(20, window // 4):
            continue
        shares.append(pd.Series(chunk).value_counts(normalize=True).reindex(range(4), fill_value=0.0).to_numpy(dtype=float))
    if len(shares) < 2:
        return 0.5
    matrix = np.vstack(shares)
    return float(max(0.0, 1.0 - matrix.std(axis=0).mean() * 4.0))


def _candidate_score(
    scaled: np.ndarray,
    labels: np.ndarray,
    features: pd.DataFrame,
    inertia: float,
    stability_window: int,
) -> dict[str, float]:
    sample_rows = min(len(scaled), 10_000)
    if len(np.unique(labels)) > 1 and sample_rows >= 4:
        sample_idx = np.linspace(0, len(scaled) - 1, sample_rows).astype(int)
        silhouette = float(silhouette_score(scaled[sample_idx], labels[sample_idx]))
    else:
        silhouette = 0.0

    proportions = pd.Series(labels).value_counts(normalize=True).reindex(range(4), fill_value=0.0).to_numpy(dtype=float)
    balance = float(max(0.0, min(1.0, proportions.min() / 0.10)))
    persistence = float((pd.Series(labels).shift(1) == pd.Series(labels)).mean())
    stability = _stability_score(labels, stability_window)

    axis = features.assign(regime_id=labels).groupby("regime_id").agg(vol_axis=("vol_axis", "mean"), structure_axis=("structure_axis", "mean"))
    low_vol = axis.loc[[0, 1], "vol_axis"].mean()
    high_vol = axis.loc[[2, 3], "vol_axis"].mean()
    range_structure = axis.loc[[0, 3], "structure_axis"].mean()
    trend_structure = axis.loc[[1, 2], "structure_axis"].mean()
    vol_sep = float((high_vol - low_vol) / (axis["vol_axis"].std(ddof=0) + 1e-12))
    structure_sep = float((trend_structure - range_structure) / (axis["structure_axis"].std(ddof=0) + 1e-12))
    interpretability = float(max(0.0, min(2.0, vol_sep + structure_sep)) / 2.0)

    score = (
        silhouette * 0.35
        + balance * 0.20
        + stability * 0.15
        + persistence * 0.10
        + interpretability * 0.20
    )
    return {
        "score": float(score),
        "silhouette_score": float(silhouette),
        "cluster_balance_score": balance,
        "transition_persistence": persistence,
        "rolling_stability_score": stability,
        "interpretability_score": interpretability,
        "kmeans_inertia": float(inertia),
    }


def _evaluate_candidate(
    name: str,
    raw_features: pd.DataFrame,
    n_regimes: int,
    random_state: int,
    stability_window: int,
) -> _CandidateResult | None:
    features = raw_features.replace([np.inf, -np.inf], np.nan).copy()
    feature_cols = _model_columns(features)
    if n_regimes != 4 or len(feature_cols) < 3:
        return None
    features = features.dropna(subset=feature_cols + ["vol_axis", "structure_axis"]).reset_index(drop=True)
    if len(features) < max(n_regimes * 30, 120):
        return None

    scaled = StandardScaler().fit_transform(features[feature_cols])
    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
    raw = km.fit_predict(scaled)
    if len(np.unique(raw)) != n_regimes:
        return None
    labels, mapping, axis_stats = _assign_quadrant_labels(raw, features)
    score_parts = _candidate_score(scaled, labels, features, km.inertia_, stability_window)
    diagnostics = {
        "candidate": name,
        "feature_columns": feature_cols,
        "feature_rows": int(len(features)),
        "raw_to_regime_id": {str(key): int(value) for key, value in mapping.items()},
        "axis_cluster_stats": axis_stats.to_dict(orient="records"),
        **score_parts,
    }
    return _CandidateResult(
        name=name,
        features=features,
        feature_cols=feature_cols,
        labels=labels,
        raw_labels=raw,
        score=float(score_parts["score"]),
        diagnostics=diagnostics,
        kmeans_inertia=float(km.inertia_),
    )


def _build_stats(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in features.columns
        if col not in {"source_index", "date"} and pd.api.types.is_numeric_dtype(features[col])
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


def _hmm_status(include_hmm: bool, features: pd.DataFrame, feature_cols: list[str], n_regimes: int, random_state: int) -> str:
    if not include_hmm:
        return "not_requested"
    if importlib.util.find_spec("hmmlearn") is None:
        return "unavailable: install hmmlearn to enable Gaussian HMM comparison"
    try:
        from hmmlearn.hmm import GaussianHMM

        scaled = StandardScaler().fit_transform(features[feature_cols])
        model = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=100, random_state=random_state)
        model.fit(scaled)
        score = float(model.score(scaled))
        return f"available_fit_success log_likelihood={score:.2f}"
    except Exception as exc:  # pragma: no cover - depends on optional dependency behavior.
        return f"available_fit_failed: {exc}"


def detect_regimes(
    data: pd.DataFrame,
    n_regimes: int = 4,
    short_window: int = 30,
    long_window: int = 120,
    random_state: int = 181,
    include_hmm: bool = False,
    stability_window: int = 2_000,
) -> RegimeResult:
    """Select the most robust four-regime KMeans model for ES/NQ minute data."""

    if n_regimes != 4:
        raise ValueError("Production regime detection requires exactly four regimes")
    if len(data) < max(long_window + 5, n_regimes * 30):
        raise ValueError("Not enough rows to detect regimes with the requested windows")

    evaluated: list[_CandidateResult] = []
    skipped: dict[str, str] = {}
    for name, frame in _candidate_frames(data, short_window, long_window).items():
        try:
            result = _evaluate_candidate(name, frame, n_regimes, random_state, stability_window)
        except Exception as exc:
            skipped[name] = str(exc)
            result = None
        if result is None:
            skipped.setdefault(name, "insufficient usable feature rows or columns")
            continue
        evaluated.append(result)

    if not evaluated:
        raise ValueError("No regime candidate produced enough usable data")

    selected = max(evaluated, key=lambda candidate: candidate.score)
    names = dict(enumerate(REGIME_NAMES))
    labels = pd.DataFrame(
        {
            "source_index": selected.features["source_index"].astype(int),
            "date": selected.features["date"],
            "regime_id": selected.labels,
            "regime_name": pd.Series(selected.labels).map(names).to_numpy(),
            "regime_model": selected.name,
        }
    )
    for regime_id in range(n_regimes):
        labels[f"prob_regime_{regime_id}"] = (labels["regime_id"] == regime_id).astype(float)

    stats = _build_stats(selected.features, selected.labels)
    hmm_status = _hmm_status(include_hmm, selected.features, selected.feature_cols, n_regimes, random_state)
    candidate_scores = {candidate.name: candidate.diagnostics for candidate in evaluated}
    diagnostics = {
        "model_name": selected.name,
        "selected_model": selected.name,
        "n_regimes": int(n_regimes),
        "regime_names": REGIME_NAMES,
        "short_window": int(short_window),
        "long_window": int(long_window),
        "feature_rows": int(len(selected.features)),
        "feature_columns": selected.feature_cols,
        "selection_method": "highest robustness score from regime candidate tournament",
        "candidate_scores": candidate_scores,
        "skipped_candidates": skipped,
        "silhouette_score": float(selected.diagnostics["silhouette_score"]),
        "kmeans_inertia": float(selected.kmeans_inertia),
        "cluster_balance_score": float(selected.diagnostics["cluster_balance_score"]),
        "rolling_stability_score": float(selected.diagnostics["rolling_stability_score"]),
        "transition_persistence": float(selected.diagnostics["transition_persistence"]),
        "interpretability_score": float(selected.diagnostics["interpretability_score"]),
        "label_ordering": "fixed quadrant mapping: range/trend crossed with low/high volatility",
        "hmm_status": hmm_status,
    }

    return RegimeResult(
        labels=labels.sort_values("date").reset_index(drop=True),
        features=selected.features.sort_values("date").reset_index(drop=True),
        regime_stats=stats,
        transition_matrix=_transition_matrix(labels.sort_values("date")["regime_id"]),
        stability=_stability(labels, stability_window),
        diagnostics=diagnostics,
        model_name=selected.name,
        hmm_status=hmm_status,
    )
