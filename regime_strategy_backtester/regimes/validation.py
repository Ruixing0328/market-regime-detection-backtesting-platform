from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from regime_strategy_backtester.backtesting.vectorized import (
    BacktestResult,
    _metrics_from_returns,
    _summarize_trades,
    run_strategy_backtests,
)
from regime_strategy_backtester.config import DEFAULT_BARS_PER_YEAR
from regime_strategy_backtester.regimes.detection import REGIME_NAMES, RegimeResult
from regime_strategy_backtester.regimes.detection_dev import (
    WalkForwardConfig,
    _FitResult,
    _build_stats,
    _candidate_frames,
    _candidate_internal_score,
    _hmm_available,
    _make_labels_frame,
    _predict_with_fit,
    _prepare_features,
    _stability,
    _transition_matrix,
    _fit_gaussian_hmm_quadrant,
    _fit_kmeans_quadrant,
    _fit_one,
)


@dataclass
class WalkForwardStrategyArtifacts:
    signals: pd.DataFrame
    backtests: BacktestResult
    overall_oos: pd.DataFrame
    per_regime_oos: pd.DataFrame
    ranking_oos: pd.DataFrame


def _model_id(candidate_name: str, model_family: str) -> str:
    return f"{candidate_name}::{model_family}"


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


def _realized_vol(log_return: pd.Series, window: int = 60) -> pd.Series:
    return log_return.astype(float).rolling(window, min_periods=max(10, window // 3)).std(ddof=0)


def _oos_score_block(features: pd.DataFrame, labels: np.ndarray, scaled: np.ndarray) -> dict[str, float]:
    base = _candidate_internal_score(scaled, labels, features)

    proportions = (
        pd.Series(labels).value_counts(normalize=True).reindex(range(4), fill_value=0.0).to_numpy(dtype=float)
    )
    min_share = float(proportions.min())
    coverage = float(max(0.0, min(1.0, min_share / 0.05)))  # want at least ~5% in each regime

    churn = float((pd.Series(labels).shift(1) != pd.Series(labels)).mean())
    churn_score = float(max(0.0, 1.0 - churn * 5.0))

    # semantic checks in the same OOS block
    axis = (
        features.assign(regime_id=labels)
        .groupby("regime_id")
        .agg(vol_axis=("vol_axis", "mean"), structure_axis=("structure_axis", "mean"))
        .reindex(range(4))
    )
    low_vol = float(axis.loc[[0, 1], "vol_axis"].mean())
    high_vol = float(axis.loc[[2, 3], "vol_axis"].mean())
    vol_semantic = float(1.0 if high_vol > low_vol else 0.0)
    range_structure = float(axis.loc[[0, 3], "structure_axis"].mean())
    trend_structure = float(axis.loc[[1, 2], "structure_axis"].mean())
    structure_semantic = float(1.0 if trend_structure > range_structure else 0.0)
    semantics = 0.5 * vol_semantic + 0.5 * structure_semantic

    score = (
        float(base["silhouette_score"]) * 0.30
        + float(base["interpretability_score"]) * 0.20
        + float(base["transition_persistence"]) * 0.10
        + churn_score * 0.15
        + coverage * 0.15
        + semantics * 0.10
    )
    return {
        "oos_score": float(score),
        "oos_silhouette": float(base["silhouette_score"]),
        "oos_interpretability": float(base["interpretability_score"]),
        "oos_persistence": float(base["transition_persistence"]),
        "oos_churn": churn,
        "oos_churn_score": churn_score,
        "oos_min_share": min_share,
        "oos_coverage_score": coverage,
        "oos_semantics_score": float(semantics),
    }


def _fit_family(
    family: Literal["kmeans_quadrant", "gaussian_hmm_quadrant"],
    features: pd.DataFrame,
    feature_cols: list[str],
    n_regimes: int,
    random_state: int,
) -> _FitResult:
    if family == "kmeans_quadrant":
        return _fit_kmeans_quadrant(features, feature_cols, n_regimes, random_state)
    return _fit_gaussian_hmm_quadrant(features, feature_cols, n_regimes, random_state)


def materialize_dev_model_result(
    data: pd.DataFrame,
    n_regimes: int,
    short_window: int,
    long_window: int,
    random_state: int,
    include_hmm: bool,
    stability_window: int,
    candidate_name: str,
    model_family: Literal["kmeans_quadrant", "gaussian_hmm_quadrant"],
    walk_forward: WalkForwardConfig | None = None,
) -> RegimeResult:
    frame = data.reset_index(drop=True).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    model_id = _model_id(candidate_name, model_family)

    if walk_forward is None:
        candidates = _candidate_frames(frame, short_window, long_window)
        raw_feats = candidates[candidate_name]
        fit, features, labels, diag = _fit_one(candidate_name, model_family, raw_feats, n_regimes, random_state)
        score = (
            float(diag["silhouette_score"]) * 0.45
            + float(diag["cluster_balance_score"]) * 0.25
            + float(diag["transition_persistence"]) * 0.10
            + float(diag["interpretability_score"]) * 0.20
        )
        labels_frame = _make_labels_frame(features, labels, model_id, probs=None)
        stats = _build_stats(features, labels_frame["regime_id"].to_numpy(dtype=int))
        diagnostics: dict[str, object] = {
            "model_name": model_id,
            "selected_model": model_id,
            "candidate_name": candidate_name,
            "model_family": model_family,
            "n_regimes": int(n_regimes),
            "regime_names": REGIME_NAMES,
            "short_window": int(short_window),
            "long_window": int(long_window),
            "random_state": int(random_state),
            "selection_method": "dev in-sample candidate materialization",
            "feature_columns": fit.feature_cols,
            "train_rows": int(fit.train_rows),
            "model_score": float(score),
            **{k: float(v) for k, v in diag.items()},
            "hmm_status": "available" if (include_hmm and _hmm_available()) else ("not_requested" if not include_hmm else "unavailable"),
        }
        return RegimeResult(
            labels=labels_frame.sort_values("date").reset_index(drop=True),
            features=features.sort_values("date").reset_index(drop=True),
            regime_stats=stats,
            transition_matrix=_transition_matrix(labels_frame.sort_values("date")["regime_id"]),
            stability=_stability(labels_frame, stability_window),
            diagnostics=diagnostics,
            model_name=str(diagnostics["model_name"]),
            hmm_status=str(diagnostics["hmm_status"]),
        )

    candidates = _candidate_frames(frame, short_window, long_window)
    raw_feats = candidates[candidate_name]
    prepared = _prepare_features(raw_feats, n_regimes)
    if prepared is None:
        raise ValueError(f"Model {model_id!r} did not produce usable features")
    features_all, feature_cols = prepared

    slices = _walk_forward_slices(len(frame), walk_forward)
    labels_chunks: list[pd.DataFrame] = []
    feature_chunks: list[pd.DataFrame] = []
    block_rows: list[dict[str, object]] = []
    model_scores: list[float] = []

    for block_id, (tr0, tr1, te0, te1) in enumerate(slices):
        train = features_all[(features_all["source_index"] >= tr0) & (features_all["source_index"] < tr1)]
        test = features_all[(features_all["source_index"] >= te0) & (features_all["source_index"] < te1)]
        if len(train) < walk_forward.min_train_bars or len(test) < max(200, n_regimes * 50):
            continue
        fit = _fit_family(model_family, train, feature_cols, n_regimes, random_state + block_id)
        labels, probs = _predict_with_fit(fit, test)

        lbl = _make_labels_frame(
            test,
            labels,
            f"{model_id}::wf",
            probs,
            raw_state_to_regime_id=fit.quadrant_mapping if probs is not None else None,
        )
        lbl["walk_forward_block"] = int(block_id)
        lbl["train_start_source_index"] = int(tr0)
        lbl["train_end_source_index"] = int(tr1 - 1)
        lbl["test_start_source_index"] = int(te0)
        lbl["test_end_source_index"] = int(te1 - 1)
        labels_chunks.append(lbl)
        feature_chunks.append(test.assign(walk_forward_block=int(block_id)))
        scaled = fit.scaler.transform(test[fit.feature_cols])
        scored = _oos_score_block(test, labels, scaled)
        model_scores.append(float(scored["oos_score"]))
        block_rows.append(
            {
                "walk_forward_block": int(block_id),
                "candidate": candidate_name,
                "model_family": model_family,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                **{k: float(v) for k, v in scored.items()},
            }
        )

    if not labels_chunks:
        raise ValueError(f"Model {model_id!r} produced no walk-forward labels")

    labels_all = pd.concat(labels_chunks, ignore_index=True).sort_values("date").reset_index(drop=True)
    features_used = pd.concat(feature_chunks, ignore_index=True).sort_values("date").reset_index(drop=True)
    stats = _build_stats(features_used, labels_all["regime_id"].to_numpy(dtype=int))
    diagnostics = {
        "model_name": f"{model_id}::wf",
        "selected_model": model_id,
        "candidate_name": candidate_name,
        "model_family": model_family,
        "selection_method": "walk-forward candidate materialization",
        "feature_columns": feature_cols,
        "model_score": float(np.mean(model_scores)) if model_scores else 0.0,
        "walk_forward": asdict(walk_forward),
        "walk_forward_blocks": block_rows,
        "short_window": int(short_window),
        "long_window": int(long_window),
        "random_state": int(random_state),
        "n_regimes": int(n_regimes),
        "regime_names": REGIME_NAMES,
        "hmm_status": "available" if (include_hmm and _hmm_available()) else ("not_requested" if not include_hmm else "unavailable"),
    }
    return RegimeResult(
        labels=labels_all,
        features=features_used,
        regime_stats=stats,
        transition_matrix=_transition_matrix(labels_all["regime_id"]),
        stability=_stability(labels_all, stability_window),
        diagnostics=diagnostics,
        model_name=str(diagnostics["model_name"]),
        hmm_status=str(diagnostics["hmm_status"]),
    )


def rank_regime_candidates_oos(
    data: pd.DataFrame,
    n_regimes: int,
    short_window: int,
    long_window: int,
    random_state: int,
    include_hmm: bool,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    frame = data.reset_index(drop=True)
    slices = _walk_forward_slices(len(frame), config)
    if not slices:
        raise ValueError("Walk-forward config produced no train/test splits")

    candidates = _candidate_frames(frame, short_window, long_window)
    families: list[str] = ["kmeans_quadrant"]
    if include_hmm and _hmm_available():
        families.append("gaussian_hmm_quadrant")

    rows: list[dict[str, object]] = []
    for candidate_name, raw_feats in candidates.items():
        prepared = _prepare_features(raw_feats, n_regimes)
        if prepared is None:
            for fam in families:
                rows.append(
                    {
                        "candidate": candidate_name,
                        "model_family": fam,
                        "status": "skipped",
                        "reason": "insufficient usable feature rows/cols",
                    }
                )
            continue
        features_all, feature_cols = prepared

        for fam in families:
            oos_scores: list[float] = []
            parts: list[dict[str, float]] = []
            blocks = 0
            failed_blocks = 0
            for block_id, (tr0, tr1, te0, te1) in enumerate(slices):
                train = features_all[(features_all["source_index"] >= tr0) & (features_all["source_index"] < tr1)]
                test = features_all[(features_all["source_index"] >= te0) & (features_all["source_index"] < te1)]
                if len(train) < config.min_train_bars or len(test) < max(200, n_regimes * 50):
                    continue
                blocks += 1
                try:
                    fit = _fit_family(fam, train, feature_cols, n_regimes, random_state + block_id)
                    labels, _ = _predict_with_fit(fit, test)
                    scaled = fit.scaler.transform(test[fit.feature_cols])
                    scored = _oos_score_block(test, labels, scaled)
                except Exception:
                    failed_blocks += 1
                    continue
                oos_scores.append(float(scored["oos_score"]))
                parts.append(scored)

            if blocks == 0 or not oos_scores:
                rows.append(
                    {
                        "candidate": candidate_name,
                        "model_family": fam,
                        "status": "failed",
                        "reason": "no valid OOS blocks",
                        "feature_cols": ",".join(feature_cols),
                    }
                )
                continue

            agg = pd.DataFrame(parts).mean(numeric_only=True).to_dict()
            rows.append(
                {
                    "candidate": candidate_name,
                    "model_family": fam,
                    "status": "ok",
                    "blocks": int(blocks),
                    "failed_blocks": int(failed_blocks),
                    "feature_cols": ",".join(feature_cols),
                    "oos_score_mean": float(np.mean(oos_scores)),
                    "oos_score_std": float(np.std(oos_scores)),
                    **{k: float(v) for k, v in agg.items()},
                }
            )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise ValueError("No OOS ranking rows produced")
    ranking["status_rank"] = (ranking["status"] == "ok").astype(int)
    ranking = ranking.sort_values(["status_rank", "oos_score_mean"], ascending=[False, False]).reset_index(drop=True)
    ranking = ranking.drop(columns=["status_rank"])
    return ranking


def walk_forward_detect_regimes(
    data: pd.DataFrame,
    n_regimes: int,
    short_window: int,
    long_window: int,
    random_state: int,
    include_hmm: bool,
    stability_window: int,
    config: WalkForwardConfig,
) -> RegimeResult:
    frame = data.reset_index(drop=True).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    ranking = rank_regime_candidates_oos(
        frame,
        n_regimes=n_regimes,
        short_window=short_window,
        long_window=long_window,
        random_state=random_state,
        include_hmm=include_hmm,
        config=config,
    )
    winners = ranking[ranking["status"] == "ok"]
    if winners.empty:
        raise ValueError("No regime candidates succeeded under OOS ranking")
    winner = winners.iloc[0]
    candidate_name = str(winner["candidate"])
    model_family = str(winner["model_family"])

    candidates = _candidate_frames(frame, short_window, long_window)
    raw_feats = candidates[candidate_name]
    prepared = _prepare_features(raw_feats, n_regimes)
    if prepared is None:
        raise ValueError("Winning candidate did not produce usable features")
    features_all, feature_cols = prepared

    slices = _walk_forward_slices(len(frame), config)
    labels_chunks: list[pd.DataFrame] = []
    feature_chunks: list[pd.DataFrame] = []
    block_rows: list[dict[str, object]] = []

    for block_id, (tr0, tr1, te0, te1) in enumerate(slices):
        train = features_all[(features_all["source_index"] >= tr0) & (features_all["source_index"] < tr1)]
        test = features_all[(features_all["source_index"] >= te0) & (features_all["source_index"] < te1)]
        if len(train) < config.min_train_bars or len(test) < max(200, n_regimes * 50):
            continue
        fit = _fit_family(model_family, train, feature_cols, n_regimes, random_state + block_id)
        labels, probs = _predict_with_fit(fit, test)

        model_name = f"{candidate_name}::{model_family}::wf"
        lbl = _make_labels_frame(
            test,
            labels,
            model_name,
            probs,
            raw_state_to_regime_id=fit.quadrant_mapping if probs is not None else None,
        )
        lbl["walk_forward_block"] = int(block_id)
        lbl["train_start_source_index"] = int(tr0)
        lbl["train_end_source_index"] = int(tr1 - 1)
        lbl["test_start_source_index"] = int(te0)
        lbl["test_end_source_index"] = int(te1 - 1)
        labels_chunks.append(lbl)
        feature_chunks.append(test.assign(walk_forward_block=int(block_id)))
        scaled = fit.scaler.transform(test[fit.feature_cols])
        scored = _oos_score_block(test, labels, scaled)
        block_rows.append(
            {
                "walk_forward_block": int(block_id),
                "candidate": candidate_name,
                "model_family": model_family,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                **{k: float(v) for k, v in scored.items()},
            }
        )

    if not labels_chunks:
        raise ValueError("Walk-forward labeling produced no labels")

    labels_all = pd.concat(labels_chunks, ignore_index=True).sort_values("date").reset_index(drop=True)
    features_used = pd.concat(feature_chunks, ignore_index=True).sort_values("date").reset_index(drop=True)
    stats = _build_stats(features_used, labels_all["regime_id"].to_numpy(dtype=int))

    diagnostics: dict[str, object] = {
        "model_name": f"{candidate_name}::{model_family}::wf",
        "selected_model": f"{candidate_name}::{model_family}",
        "selection_method": "walk-forward OOS ranking; winner refit each block",
        "candidate_rankings": ranking.to_dict(orient="records"),
        "walk_forward": asdict(config),
        "walk_forward_blocks": block_rows,
        "feature_columns": feature_cols,
        "short_window": int(short_window),
        "long_window": int(long_window),
        "random_state": int(random_state),
        "n_regimes": int(n_regimes),
        "hmm_status": "available" if (include_hmm and _hmm_available()) else ("not_requested" if not include_hmm else "unavailable"),
        "regime_names": REGIME_NAMES,
    }

    return RegimeResult(
        labels=labels_all,
        features=features_used,
        regime_stats=stats,
        transition_matrix=_transition_matrix(labels_all["regime_id"]),
        stability=_stability(labels_all, stability_window),
        diagnostics=diagnostics,
        model_name=str(diagnostics["model_name"]),
        hmm_status=str(diagnostics["hmm_status"]),
    )


def strategy_per_regime_oos_metrics(
    data: pd.DataFrame,
    regimes_oos: pd.DataFrame,
    signals: dict[str, pd.Series],
    initial_capital: float,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    backtests = run_strategy_backtests(
        data=data,
        signals=signals,
        regimes=regimes_oos,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
    )

    per_regime = backtests.per_regime_metrics.copy()
    overall = backtests.metrics.copy()
    return overall, per_regime


def _equity_curves_from_returns(returns: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if returns.empty or "date" not in returns.columns:
        return pd.DataFrame()
    strategy_cols = [column for column in returns.columns if column != "date"]
    pieces: list[pd.DataFrame] = []
    for strategy in strategy_cols:
        series = pd.to_numeric(returns[strategy], errors="coerce").fillna(0.0)
        equity = initial_capital * (1.0 + series).cumprod()
        pieces.append(pd.DataFrame({"date": returns["date"], "strategy": strategy, "equity": equity, "return": series}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _metrics_from_stitched_walk_forward(
    returns: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if returns.empty or "date" not in returns.columns:
        return pd.DataFrame(), pd.DataFrame()
    strategy_cols = [column for column in returns.columns if column != "date"]
    equity_curves = _equity_curves_from_returns(returns, initial_capital)
    metric_rows: list[dict[str, object]] = []
    for strategy in strategy_cols:
        series = pd.to_numeric(returns[strategy], errors="coerce").fillna(0.0)
        equity = equity_curves[equity_curves["strategy"] == strategy]["equity"].reset_index(drop=True)
        metrics = _metrics_from_returns(series.reset_index(drop=True), equity, bars_per_year=DEFAULT_BARS_PER_YEAR)
        strategy_trades = trades[trades["strategy"].astype(str) == strategy] if not trades.empty else pd.DataFrame()
        _, trade_metrics = _summarize_trades(strategy_trades)
        metrics.update(trade_metrics)
        metrics["strategy"] = strategy
        metric_rows.append(metrics)
    metrics_frame = pd.DataFrame(metric_rows).sort_values("strategy").reset_index(drop=True) if metric_rows else pd.DataFrame()
    return equity_curves, metrics_frame


def _per_regime_metrics_from_stitched_walk_forward(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    if returns.empty or regimes.empty or "date" not in returns.columns:
        return pd.DataFrame()
    strategy_cols = [column for column in returns.columns if column != "date"]
    regime_lookup = regimes[["date", "regime_id", "regime_name"]].drop_duplicates(subset=["date"]).copy()
    frame = returns.merge(regime_lookup, on="date", how="left").dropna(subset=["regime_id"])
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for strategy in strategy_cols:
        strategy_frame = frame[["date", strategy, "regime_id", "regime_name"]].copy()
        strategy_frame["return"] = pd.to_numeric(strategy_frame[strategy], errors="coerce").fillna(0.0)
        for (regime_id, regime_name), group in strategy_frame.groupby(["regime_id", "regime_name"]):
            local_equity = (1.0 + group["return"].fillna(0.0)).cumprod() * initial_capital
            metrics = _metrics_from_returns(group["return"], local_equity, bars_per_year=DEFAULT_BARS_PER_YEAR)
            metrics["win_rate"] = metrics.get("bar_win_rate", 0.0)
            metrics["strategy"] = strategy
            metrics["regime_id"] = int(regime_id)
            metrics["regime_name"] = str(regime_name)
            rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["strategy", "regime_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def build_walk_forward_strategy_artifacts(
    data: pd.DataFrame,
    regimes: RegimeResult,
    signals: dict[str, pd.Series],
    initial_capital: float,
    transaction_cost_bps: float,
    apply_train_regime_filter: bool = False,
    filter_metric: Literal["sharpe", "expectancy"] = "sharpe",
    filter_threshold: float = 0.0,
) -> WalkForwardStrategyArtifacts:
    if regimes.labels.empty or "walk_forward_block" not in regimes.labels.columns:
        raise ValueError("RegimeResult must come from walk-forward detection (missing walk_forward_block)")

    selected = str(regimes.diagnostics.get("selected_model", ""))
    if "::" not in selected:
        raise ValueError("Could not infer selected model from regime diagnostics")
    candidate_name, model_family = selected.split("::", 1)
    model_family = model_family.strip()
    candidate_name = candidate_name.strip()

    n_regimes = int(regimes.diagnostics.get("n_regimes", 4))
    short_window = int(regimes.diagnostics.get("short_window", 30))
    long_window = int(regimes.diagnostics.get("long_window", 120))

    candidates = _candidate_frames(data, short_window, long_window)
    if candidate_name not in candidates:
        raise ValueError(f"Selected candidate {candidate_name!r} not available")
    raw_feats = candidates[candidate_name]
    prepared = _prepare_features(raw_feats, n_regimes)
    if prepared is None:
        raise ValueError("Selected candidate did not produce usable features")
    features_all, feature_cols = prepared

    overall_rows: list[pd.DataFrame] = []
    per_regime_rows: list[pd.DataFrame] = []
    filter_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    return_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    blocks = sorted(regimes.labels["walk_forward_block"].dropna().astype(int).unique().tolist())
    for block_id in blocks:
        lbl_block = regimes.labels[regimes.labels["walk_forward_block"].astype(int) == block_id].copy()
        if lbl_block.empty:
            continue
        te0 = int(lbl_block["test_start_source_index"].iloc[0])
        te1 = int(lbl_block["test_end_source_index"].iloc[0]) + 1
        tr0 = int(lbl_block["train_start_source_index"].iloc[0])
        tr1 = int(lbl_block["train_end_source_index"].iloc[0]) + 1

        data_test = data.iloc[te0:te1].reset_index(drop=True)
        regimes_test = lbl_block.copy()
        regimes_test["source_index"] = regimes_test["source_index"].astype(int) - te0

        signals_test = {name: series.iloc[te0:te1].reset_index(drop=True) for name, series in signals.items()}

        if apply_train_regime_filter:
            train = features_all[(features_all["source_index"] >= tr0) & (features_all["source_index"] < tr1)]
            if len(train) >= max(5_000, n_regimes * 200):
                fit = _fit_family(model_family, train, feature_cols, n_regimes, int(regimes.diagnostics.get("random_state", 181)) + block_id)
                train_labels, _ = _predict_with_fit(fit, train)
                train_lbl = _make_labels_frame(
                    train,
                    train_labels,
                    model_name=f"{candidate_name}::{model_family}::train",
                    probs=None,
                )
                train_lbl["source_index"] = train_lbl["source_index"].astype(int) - tr0
                data_train = data.iloc[tr0:tr1].reset_index(drop=True)
                signals_train = {name: series.iloc[tr0:tr1].reset_index(drop=True) for name, series in signals.items()}
                overall_train, per_regime_train = strategy_per_regime_oos_metrics(
                    data=data_train,
                    regimes_oos=train_lbl,
                    signals=signals_train,
                    initial_capital=initial_capital,
                    transaction_cost_bps=transaction_cost_bps,
                )
                allow_map: dict[str, set[int]] = {}
                for strategy in overall_train["strategy"].astype(str).unique():
                    local = per_regime_train[per_regime_train["strategy"].astype(str) == strategy]
                    allowed = set(local.loc[local[filter_metric].astype(float) > filter_threshold, "regime_id"].astype(int).tolist())
                    allow_map[strategy] = allowed
                test_regime = regimes_test.set_index("source_index")["regime_id"].astype(int).reindex(range(len(data_test))).ffill()
                for name in list(signals_test.keys()):
                    allowed = allow_map.get(str(name), set(range(4)))
                    mask = test_regime.isin(list(allowed)).fillna(False).astype(float)
                    signals_test[name] = signals_test[name].astype(float) * mask.to_numpy(dtype=float)
                    filter_rows.append(
                        {
                            "walk_forward_block": int(block_id),
                            "strategy": str(name),
                            "allowed_regimes": ",".join(str(x) for x in sorted(allowed)),
                            "filter_metric": filter_metric,
                            "filter_threshold": float(filter_threshold),
                        }
                    )

        backtests = run_strategy_backtests(
            data=data_test,
            signals=signals_test,
            regimes=regimes_test,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
        )
        overall = backtests.metrics.copy()
        per_regime = backtests.per_regime_metrics.copy()
        overall["walk_forward_block"] = int(block_id)
        per_regime["walk_forward_block"] = int(block_id)
        overall_rows.append(overall)
        per_regime_rows.append(per_regime)

        signal_frame = pd.DataFrame({"date": data_test["date"]})
        for name, series in signals_test.items():
            signal_frame[name] = series.astype(float).to_numpy()
        signal_frames.append(signal_frame)
        return_frames.append(backtests.returns.copy())
        if not backtests.trades.empty:
            trades = backtests.trades.copy()
            trades["walk_forward_block"] = int(block_id)
            trade_frames.append(trades)

    overall_all = pd.concat(overall_rows, ignore_index=True) if overall_rows else pd.DataFrame()
    per_regime_all = pd.concat(per_regime_rows, ignore_index=True) if per_regime_rows else pd.DataFrame()
    filters = pd.DataFrame(filter_rows)

    if per_regime_all.empty:
        ranking = filters
    else:
        ranking = (
            per_regime_all.groupby(["strategy", "regime_id", "regime_name"], as_index=False)
            .agg(
                blocks=("walk_forward_block", "nunique"),
                sharpe=("sharpe", "mean"),
                expectancy=("expectancy", "mean"),
                max_drawdown=("max_drawdown", "mean"),
                win_rate=("win_rate", "mean"),
                bars=("bars", "sum"),
            )
            .sort_values(["regime_id", "expectancy", "sharpe", "bars"], ascending=[True, False, False, False])
            .reset_index(drop=True)
        )

    signals_all = pd.concat(signal_frames, ignore_index=True).sort_values("date").reset_index(drop=True) if signal_frames else pd.DataFrame()
    returns_all = pd.concat(return_frames, ignore_index=True).sort_values("date").reset_index(drop=True) if return_frames else pd.DataFrame()
    trades_all = pd.concat(trade_frames, ignore_index=True).sort_values(["strategy", "entry_date", "exit_date"]).reset_index(drop=True) if trade_frames else pd.DataFrame()
    equity_all, metrics_all = _metrics_from_stitched_walk_forward(returns_all, trades_all, initial_capital)
    per_regime_metrics_all = _per_regime_metrics_from_stitched_walk_forward(returns_all, regimes.labels, initial_capital)

    return WalkForwardStrategyArtifacts(
        signals=signals_all,
        backtests=BacktestResult(
            returns=returns_all,
            equity_curves=equity_all,
            metrics=metrics_all,
            per_regime_metrics=per_regime_metrics_all,
            trades=trades_all,
        ),
        overall_oos=overall_all,
        per_regime_oos=per_regime_all,
        ranking_oos=ranking,
    )


def evaluate_strategies_by_regime_walk_forward(
    data: pd.DataFrame,
    regimes: RegimeResult,
    signals: dict[str, pd.Series],
    initial_capital: float,
    transaction_cost_bps: float,
    apply_train_regime_filter: bool = False,
    filter_metric: Literal["sharpe", "expectancy"] = "sharpe",
    filter_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts = build_walk_forward_strategy_artifacts(
        data=data,
        regimes=regimes,
        signals=signals,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        apply_train_regime_filter=apply_train_regime_filter,
        filter_metric=filter_metric,
        filter_threshold=filter_threshold,
    )
    return artifacts.overall_oos, artifacts.per_regime_oos, artifacts.ranking_oos
