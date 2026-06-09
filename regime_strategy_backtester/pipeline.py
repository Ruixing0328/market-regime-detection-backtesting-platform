from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy_backtester.backtesting import run_strategy_backtests
from regime_strategy_backtester.config import OUTPUT_ROOT, NY_TZ, resolve_data_path
from regime_strategy_backtester.data import load_futures_data
from regime_strategy_backtester.regimes import WalkForwardConfig, detect_regimes, detect_regimes_dev
from regime_strategy_backtester.regimes.validation import build_walk_forward_strategy_artifacts, materialize_dev_model_result
from regime_strategy_backtester.strategies import generate_strategy_signals
from regime_strategy_backtester.stress import run_stress_suite


OUTRIGHT_PATTERNS = {
    "ES": re.compile(r"^ES[HMUZ]\d{1,2}$"),
    "NQ": re.compile(r"^NQ[HMUZ]\d{1,2}$"),
}
MODEL_BUNDLE_DIRNAME = "dashboard_models"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _signals_to_frame(data: pd.DataFrame, signals: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame({"date": data["date"]})
    for name, series in signals.items():
        frame[name] = series.reindex(data.index).fillna(0.0).astype(float).to_numpy()
    return frame


def _price_regime_frame(data: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels_for_merge = labels[["source_index", "regime_id", "regime_name"]].copy()
    price_regimes = data.reset_index(drop=True).reset_index(names="source_index").merge(labels_for_merge, on="source_index", how="left")
    price_regimes[["regime_id", "regime_name"]] = price_regimes[["regime_id", "regime_name"]].ffill()
    return price_regimes


def _stress_regime_frame(strategy_returns: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if strategy_returns.empty or labels.empty or "date" not in strategy_returns.columns or "date" not in labels.columns:
        return labels
    lookup = labels[["date", "regime_id", "regime_name"]].drop_duplicates(subset=["date"]).sort_values("date")
    frame = strategy_returns[["date"]].drop_duplicates(subset=["date"]).sort_values("date").merge(lookup, on="date", how="left")
    frame[["regime_id", "regime_name"]] = frame[["regime_id", "regime_name"]].ffill()
    return frame.reset_index(drop=True).reset_index(names="source_index")


def _downsample_equity_curves(equity: pd.DataFrame, max_points_per_strategy: int) -> pd.DataFrame:
    if equity.empty or max_points_per_strategy <= 0 or "strategy" not in equity.columns:
        return equity
    pieces = []
    for _, group in equity.groupby("strategy", sort=False):
        if len(group) <= max_points_per_strategy:
            pieces.append(group)
            continue
        take = np.linspace(0, len(group) - 1, max_points_per_strategy).astype(int)
        pieces.append(group.iloc[np.unique(take)])
    return pd.concat(pieces, ignore_index=True) if pieces else equity


def _latest_raw_date(symbol: str, data_path: str | Path | None) -> pd.Timestamp:
    path = resolve_data_path(symbol, data_path)
    pattern = OUTRIGHT_PATTERNS[symbol]
    latest: pd.Timestamp | None = None
    for chunk in pd.read_csv(path, usecols=["ts_event", "symbol"], chunksize=500_000):
        rows = chunk[chunk["symbol"].astype(str).str.match(pattern)]
        if rows.empty:
            continue
        ts = pd.to_datetime(rows["ts_event"], utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        local_latest = ts.max().tz_convert(NY_TZ).tz_localize(None)
        latest = local_latest if latest is None or local_latest > latest else latest
    if latest is None:
        raise ValueError(f"Could not infer latest available date for {symbol}")
    return latest


def _resolve_date_window(args: argparse.Namespace, symbol: str) -> tuple[str | None, str | None]:
    if args.full_history:
        return None, None
    if args.last_years is None:
        return args.start_date, args.end_date
    end = pd.Timestamp(args.end_date) if args.end_date else _latest_raw_date(symbol, args.data_path)
    start = pd.Timestamp(args.start_date) if args.start_date else end - pd.DateOffset(years=int(args.last_years))
    return str(start.date()), str(end.date())


def _strategy_list(value: list[str] | None) -> list[str] | None:
    if not value:
        return None
    out: list[str] = []
    for item in value:
        out.extend(part.strip() for part in item.split(",") if part.strip())
    return out or None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _model_slug(model_id: str) -> str:
    slug = model_id.replace("::", "__")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", slug).strip("._").lower()
    return slug or "model"


def _model_display_label(model_id: str) -> str:
    def _pretty(part: str) -> str:
        words: list[str] = []
        for token in part.split("_"):
            lowered = token.lower()
            if lowered == "kmeans":
                words.append("KMeans")
            elif lowered == "hmm":
                words.append("HMM")
            else:
                words.append(token.capitalize())
        return " ".join(words)

    if "::" in model_id:
        candidate, family = model_id.split("::", 1)
        return f"{_pretty(candidate)} / {_pretty(family)}"
    return _pretty(model_id)


def _implicit_model_variant(model_id: str, recommendation_source: str, artifact_subdir: str | None = None) -> dict[str, object]:
    return {
        "model_id": model_id,
        "display_label": _model_display_label(model_id),
        "candidate": model_id.split("::", 1)[0],
        "model_family": model_id.split("::", 1)[1] if "::" in model_id else "",
        "artifact_subdir": artifact_subdir,
        "recommendation_source": recommendation_source,
        "rank_metric": "selected",
        "rank_value": 1.0,
        "rank": 1,
        "is_best": True,
    }


def _prod_dashboard_model_variants(regimes, include_hmm: bool) -> list[dict[str, object]]:
    selected_candidate = str(regimes.diagnostics.get("selected_model", regimes.model_name)).split("::", 1)[0].strip()
    if not selected_candidate:
        selected_candidate = str(regimes.model_name).split("::", 1)[0].strip() or "current_model"

    candidate_scores = regimes.diagnostics.get("candidate_scores", {})
    selected_score = candidate_scores.get(selected_candidate, {}) if isinstance(candidate_scores, dict) else {}
    variants: list[dict[str, object]] = []

    kmeans_model_id = f"{selected_candidate}::kmeans_quadrant"
    variants.append(
        {
            "model_id": kmeans_model_id,
            "display_label": _model_display_label(kmeans_model_id),
            "candidate": selected_candidate,
            "model_family": "kmeans_quadrant",
            "artifact_subdir": None,
            "recommendation_source": "production_default",
            "rank_metric": "rank_value",
            "rank_value": _safe_float(selected_score.get("score"), 1.0),
            "score": _safe_float(selected_score.get("score"), 1.0),
            "silhouette_score": _safe_float(selected_score.get("silhouette_score"), 0.0),
            "cluster_balance_score": _safe_float(selected_score.get("cluster_balance_score"), 0.0),
            "rolling_stability_score": _safe_float(selected_score.get("rolling_stability_score"), 0.0),
            "transition_persistence": _safe_float(selected_score.get("transition_persistence"), 0.0),
            "interpretability_score": _safe_float(selected_score.get("interpretability_score"), 0.0),
            "rank": 1,
            "is_best": True,
        }
    )

    if include_hmm and importlib.util.find_spec("hmmlearn") is not None:
        hmm_model_id = f"{selected_candidate}::gaussian_hmm_quadrant"
        variants.append(
            {
                "model_id": hmm_model_id,
                "display_label": _model_display_label(hmm_model_id),
                "candidate": selected_candidate,
                "model_family": "gaussian_hmm_quadrant",
                "artifact_subdir": f"{MODEL_BUNDLE_DIRNAME}/{_model_slug(hmm_model_id)}",
                "recommendation_source": "alternate_family",
                "rank_metric": "model_score",
                "rank_value": 0.0,
                "model_score": 0.0,
                "rank": 2,
                "is_best": False,
            }
        )

    return variants


def _normalize_dashboard_model_variants(args: argparse.Namespace, regimes) -> list[dict[str, object]]:
    selected_model = str(regimes.diagnostics.get("selected_model", regimes.model_name))
    variants: list[dict[str, object]] = []

    if args.regime_detector == "dev" and args.walk_forward:
        for row in regimes.diagnostics.get("candidate_rankings", []):
            if str(row.get("status", "")) != "ok":
                continue
            candidate = str(row.get("candidate", "")).strip()
            family = str(row.get("model_family", "")).strip()
            if not candidate or not family:
                continue
            model_id = f"{candidate}::{family}"
            variants.append(
                {
                    "model_id": model_id,
                    "display_label": _model_display_label(model_id),
                    "candidate": candidate,
                    "model_family": family,
                    "artifact_subdir": f"{MODEL_BUNDLE_DIRNAME}/{_model_slug(model_id)}",
                    "recommendation_source": "oos",
                    "rank_metric": "oos_score_mean",
                    "rank_value": _safe_float(row.get("oos_score_mean"), 0.0),
                    "oos_score_mean": _safe_float(row.get("oos_score_mean"), 0.0),
                    "oos_score_std": _safe_float(row.get("oos_score_std"), 0.0),
                    "oos_silhouette": _safe_float(row.get("oos_silhouette"), 0.0),
                    "oos_interpretability": _safe_float(row.get("oos_interpretability"), 0.0),
                    "oos_persistence": _safe_float(row.get("oos_persistence"), 0.0),
                    "oos_churn": _safe_float(row.get("oos_churn"), 0.0),
                    "oos_min_share": _safe_float(row.get("oos_min_share"), 0.0),
                    "blocks": _safe_int(row.get("blocks"), 0),
                    "failed_blocks": _safe_int(row.get("failed_blocks"), 0),
                    "status": "ok",
                }
            )
    elif args.regime_detector == "dev":
        for row in regimes.diagnostics.get("candidate_scores", []):
            candidate = str(row.get("candidate", "")).strip()
            family = str(row.get("model_family", "")).strip()
            if not candidate or not family:
                continue
            model_id = f"{candidate}::{family}"
            variants.append(
                {
                    "model_id": model_id,
                    "display_label": _model_display_label(model_id),
                    "candidate": candidate,
                    "model_family": family,
                    "artifact_subdir": f"{MODEL_BUNDLE_DIRNAME}/{_model_slug(model_id)}",
                    "recommendation_source": "in_sample",
                    "rank_metric": "score",
                    "rank_value": _safe_float(row.get("score"), 0.0),
                    "score": _safe_float(row.get("score"), 0.0),
                    "silhouette_score": _safe_float(row.get("silhouette_score"), 0.0),
                    "cluster_balance_score": _safe_float(row.get("cluster_balance_score"), 0.0),
                    "transition_persistence": _safe_float(row.get("transition_persistence"), 0.0),
                    "interpretability_score": _safe_float(row.get("interpretability_score"), 0.0),
                    "train_rows": _safe_int(row.get("train_rows"), 0),
                }
            )
    else:
        return _prod_dashboard_model_variants(regimes, include_hmm=bool(args.include_hmm))

    if not variants:
        recommendation_source = "oos" if args.walk_forward else "in_sample"
        return [_implicit_model_variant(selected_model, recommendation_source)]

    variants = sorted(variants, key=lambda item: _safe_float(item.get("rank_value"), 0.0), reverse=True)
    selected_found = False
    for rank, variant in enumerate(variants, start=1):
        variant["rank"] = int(rank)
        is_best = str(variant.get("model_id", "")) == selected_model
        variant["is_best"] = is_best
        selected_found = selected_found or is_best
    if not selected_found and variants:
        variants[0]["is_best"] = True
    return variants


def _write_dashboard_bundle(
    bundle_dir: Path,
    *,
    run_id: str,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    full_history: bool,
    data_summary: object,
    args: argparse.Namespace,
    variant: dict[str, object],
    regimes,
    price_regimes: pd.DataFrame,
    strategy_names: list[str],
    signals: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    equity: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    per_regime_metrics: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    stress_summary: pd.DataFrame | None = None,
    stress_paths: pd.DataFrame | None = None,
    oos_regime_model_rankings: pd.DataFrame | None = None,
    oos_regime_blocks: pd.DataFrame | None = None,
    strategy_metrics_oos: pd.DataFrame | None = None,
    strategy_per_regime_metrics_oos: pd.DataFrame | None = None,
    strategy_regime_rankings_oos: pd.DataFrame | None = None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary_dict = data_summary.to_dict() if hasattr(data_summary, "to_dict") else dict(data_summary)

    _write_json(bundle_dir / "data_summary.json", summary_dict)
    _write_json(bundle_dir / "regime_diagnostics.json", regimes.diagnostics)
    regimes.labels.to_csv(bundle_dir / "regime_labels.csv", index=False)
    regimes.features.to_csv(bundle_dir / "regime_features.csv", index=False)
    regimes.regime_stats.to_csv(bundle_dir / "regime_stats.csv", index=False)
    regimes.transition_matrix.to_csv(bundle_dir / "regime_transitions.csv", index=False)
    regimes.stability.to_csv(bundle_dir / "regime_stability.csv", index=False)
    price_regimes.to_csv(bundle_dir / "price_regimes.csv", index=False)

    if args.walk_forward:
        regimes.labels.to_csv(bundle_dir / "oos_regime_labels.csv", index=False)
        if oos_regime_model_rankings is not None:
            oos_regime_model_rankings.to_csv(bundle_dir / "oos_regime_model_rankings.csv", index=False)
        if oos_regime_blocks is not None:
            oos_regime_blocks.to_csv(bundle_dir / "oos_regime_blocks.csv", index=False)

    if signals is not None:
        signals.to_csv(bundle_dir / "strategy_signals.csv", index=False)
    if returns is not None:
        returns.to_csv(bundle_dir / "strategy_returns.csv", index=False)
    if equity is not None:
        equity.to_csv(bundle_dir / "equity_curves.csv", index=False)
    if metrics is not None:
        metrics.to_csv(bundle_dir / "metrics.csv", index=False)
    if per_regime_metrics is not None:
        per_regime_metrics.to_csv(bundle_dir / "per_regime_metrics.csv", index=False)
    if trades is not None:
        trades.to_csv(bundle_dir / "trades.csv", index=False)
    if stress_summary is not None:
        stress_summary.to_csv(bundle_dir / "stress_summary.csv", index=False)
    if stress_paths is not None:
        stress_paths.to_csv(bundle_dir / "monte_carlo_paths.csv", index=False)
    if strategy_metrics_oos is not None:
        strategy_metrics_oos.to_csv(bundle_dir / "strategy_metrics_oos.csv", index=False)
    if strategy_per_regime_metrics_oos is not None:
        strategy_per_regime_metrics_oos.to_csv(bundle_dir / "strategy_per_regime_metrics_oos.csv", index=False)
    if strategy_regime_rankings_oos is not None:
        strategy_regime_rankings_oos.to_csv(bundle_dir / "strategy_regime_rankings_oos.csv", index=False)

    bundle_manifest = {
        "run_id": f"{run_id}::{variant.get('model_id', '')}",
        "parent_run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol.upper(),
        "project": "Market Regime Strategy Backtester",
        "mode": "regime_only" if args.regime_only else "full",
        "date_window": {"start_date": start_date, "end_date": end_date, "full_history": bool(full_history)},
        "data_summary": summary_dict,
        "regime_model": regimes.model_name,
        "regime_diagnostics": regimes.diagnostics,
        "hmm_status": regimes.hmm_status,
        "strategies": strategy_names,
        "costs": {"initial_capital": args.initial_capital, "transaction_cost_bps": args.transaction_cost_bps},
        "dashboard_model_variant": variant,
    }
    bundle_manifest["artifacts"] = sorted({path.name for path in bundle_dir.iterdir() if path.is_file()} | {"manifest.json"})
    _write_json(bundle_dir / "manifest.json", bundle_manifest)


def _run_one_symbol(args: argparse.Namespace, symbol: str, run_id: str, output_root: Path) -> Path:
    start_date, end_date = _resolve_date_window(args, symbol)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data, data_summary = load_futures_data(
        symbol=symbol,
        data_path=args.data_path,
        start_date=start_date,
        end_date=end_date,
        sample_rows=args.sample_rows,
        session_start=args.session_start,
        session_end=args.session_end,
        back_adjust=not args.no_back_adjust,
        use_cache=not args.rebuild_cache,
    )
    wf_config: WalkForwardConfig | None = None
    if args.regime_detector == "dev":
        wf_config = WalkForwardConfig(
            train_bars=int(args.wf_train_bars),
            test_bars=int(args.wf_test_bars),
            step_bars=int(args.wf_step_bars) if args.wf_step_bars is not None else None,
            min_train_bars=int(args.wf_min_train_bars),
        )
        regimes = detect_regimes_dev(
            data,
            n_regimes=args.regimes,
            short_window=args.short_window,
            long_window=args.long_window,
            random_state=args.seed,
            include_hmm=args.include_hmm,
            stability_window=args.stability_window,
            mode="walk_forward" if args.walk_forward else "in_sample",
            walk_forward=wf_config,
        )
    else:
        regimes = detect_regimes(
            data,
            n_regimes=args.regimes,
            short_window=args.short_window,
            long_window=args.long_window,
            random_state=args.seed,
            include_hmm=args.include_hmm,
            stability_window=args.stability_window,
        )
    price_regimes = _price_regime_frame(data, regimes.labels)

    _write_json(run_dir / "data_summary.json", data_summary.to_dict())
    regimes.labels.to_csv(run_dir / "regime_labels.csv", index=False)
    regimes.features.to_csv(run_dir / "regime_features.csv", index=False)
    regimes.regime_stats.to_csv(run_dir / "regime_stats.csv", index=False)
    regimes.transition_matrix.to_csv(run_dir / "regime_transitions.csv", index=False)
    regimes.stability.to_csv(run_dir / "regime_stability.csv", index=False)
    price_regimes.to_csv(run_dir / "price_regimes.csv", index=False)

    oos_regime_model_rankings = pd.DataFrame()
    oos_regime_blocks = pd.DataFrame()
    if args.regime_detector == "dev" and args.walk_forward:
        # Additional OOS artifacts for walk-forward evaluation.
        regimes.labels.to_csv(run_dir / "oos_regime_labels.csv", index=False)
        oos_regime_model_rankings = pd.DataFrame(regimes.diagnostics.get("candidate_rankings", []))
        oos_regime_blocks = pd.DataFrame(regimes.diagnostics.get("walk_forward_blocks", []))
        oos_regime_model_rankings.to_csv(run_dir / "oos_regime_model_rankings.csv", index=False)
        oos_regime_blocks.to_csv(run_dir / "oos_regime_blocks.csv", index=False)

    selected_strategies = _strategy_list(args.strategies)
    strategy_metrics_oos = pd.DataFrame()
    strategy_per_regime_metrics_oos = pd.DataFrame()
    strategy_regime_rankings_oos = pd.DataFrame()
    strategy_names: list[str] = []
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol.upper(),
        "project": "Market Regime Strategy Backtester",
        "mode": "regime_only" if args.regime_only else "full",
        "date_window": {"start_date": start_date, "end_date": end_date, "full_history": bool(args.full_history)},
        "data_summary": data_summary.to_dict(),
        "regime_model": regimes.model_name,
        "regime_diagnostics": regimes.diagnostics,
        "hmm_status": regimes.hmm_status,
        "strategies": [],
        "costs": {"initial_capital": args.initial_capital, "transaction_cost_bps": args.transaction_cost_bps},
        "monte_carlo": {
            "stress_models": args.stress_models,
            "regime_filters": args.stress_regimes,
            "simulations": args.simulations,
            "horizon": args.stress_horizon,
            "block_size": args.block_size,
            "max_saved_paths": args.max_saved_paths,
            "max_equity_points_per_strategy": args.max_equity_points_per_strategy,
            "drawdown_threshold": args.drawdown_threshold,
        },
    }

    if not args.regime_only:
        signals = generate_strategy_signals(data, regimes.labels, selected_strategies=selected_strategies)
        strategy_names = list(signals.keys())
        backtests = run_strategy_backtests(
            data,
            signals,
            regimes.labels,
            initial_capital=args.initial_capital,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        stress = run_stress_suite(
            backtests.returns,
            _stress_regime_frame(backtests.returns, regimes.labels),
            simulations=args.simulations,
            horizon=args.stress_horizon,
            block_size=args.block_size,
            seed=args.seed,
            stress_models=args.stress_models,
            regime_filters=args.stress_regimes,
            initial_capital=args.initial_capital,
            max_saved_paths=args.max_saved_paths,
            drawdown_threshold=args.drawdown_threshold,
        )

        _signals_to_frame(data, signals).to_csv(run_dir / "strategy_signals.csv", index=False)
        backtests.returns.to_csv(run_dir / "strategy_returns.csv", index=False)
        dashboard_equity = _downsample_equity_curves(backtests.equity_curves, args.max_equity_points_per_strategy)
        dashboard_equity.to_csv(run_dir / "equity_curves.csv", index=False)
        backtests.metrics.to_csv(run_dir / "metrics.csv", index=False)
        backtests.per_regime_metrics.to_csv(run_dir / "per_regime_metrics.csv", index=False)
        backtests.trades.to_csv(run_dir / "trades.csv", index=False)
        stress.summary.to_csv(run_dir / "stress_summary.csv", index=False)
        stress.paths.to_csv(run_dir / "monte_carlo_paths.csv", index=False)
        manifest["strategies"] = strategy_names

        if args.regime_detector == "dev" and args.walk_forward:
            strategy_artifacts = build_walk_forward_strategy_artifacts(
                data=data,
                regimes=regimes,
                signals=signals,
                initial_capital=args.initial_capital,
                transaction_cost_bps=args.transaction_cost_bps,
                apply_train_regime_filter=args.regime_filter_train,
                filter_metric=args.regime_filter_metric,
                filter_threshold=args.regime_filter_threshold,
            )
            strategy_metrics_oos = strategy_artifacts.overall_oos
            strategy_per_regime_metrics_oos = strategy_artifacts.per_regime_oos
            strategy_regime_rankings_oos = strategy_artifacts.ranking_oos
            strategy_metrics_oos.to_csv(run_dir / "strategy_metrics_oos.csv", index=False)
            strategy_per_regime_metrics_oos.to_csv(run_dir / "strategy_per_regime_metrics_oos.csv", index=False)
            strategy_regime_rankings_oos.to_csv(run_dir / "strategy_regime_rankings_oos.csv", index=False)

    dashboard_variants = _normalize_dashboard_model_variants(args, regimes)
    bundle_errors: dict[str, str] = {}
    written_variants: list[dict[str, object]] = []

    if dashboard_variants:
        for variant in dashboard_variants:
            artifact_subdir = str(variant.get("artifact_subdir", "") or "")
            if not artifact_subdir:
                written_variants.append(variant)
                continue
            try:
                bundle_regimes = materialize_dev_model_result(
                    data=data,
                    n_regimes=args.regimes,
                    short_window=args.short_window,
                    long_window=args.long_window,
                    random_state=args.seed,
                    include_hmm=args.include_hmm,
                    stability_window=args.stability_window,
                    candidate_name=str(variant.get("candidate", "")),
                    model_family=str(variant.get("model_family", "")),
                    walk_forward=wf_config if args.walk_forward else None,
                )
                variant["rank_value"] = _safe_float(bundle_regimes.diagnostics.get("model_score"), _safe_float(variant.get("rank_value"), 0.0))
                if "model_score" in bundle_regimes.diagnostics:
                    variant["model_score"] = _safe_float(bundle_regimes.diagnostics.get("model_score"), 0.0)
                if "silhouette_score" in bundle_regimes.diagnostics:
                    variant["silhouette_score"] = _safe_float(bundle_regimes.diagnostics.get("silhouette_score"), 0.0)
                if "transition_persistence" in bundle_regimes.diagnostics:
                    variant["transition_persistence"] = _safe_float(bundle_regimes.diagnostics.get("transition_persistence"), 0.0)
                if "interpretability_score" in bundle_regimes.diagnostics:
                    variant["interpretability_score"] = _safe_float(bundle_regimes.diagnostics.get("interpretability_score"), 0.0)
                bundle_price_regimes = _price_regime_frame(data, bundle_regimes.labels)

                bundle_signals_frame: pd.DataFrame | None = None
                bundle_returns: pd.DataFrame | None = None
                bundle_equity: pd.DataFrame | None = None
                bundle_metrics: pd.DataFrame | None = None
                bundle_per_regime_metrics: pd.DataFrame | None = None
                bundle_trades: pd.DataFrame | None = None
                bundle_stress_summary: pd.DataFrame | None = None
                bundle_stress_paths: pd.DataFrame | None = None
                bundle_strategy_metrics_oos: pd.DataFrame | None = None
                bundle_strategy_per_regime_metrics_oos: pd.DataFrame | None = None
                bundle_strategy_regime_rankings_oos: pd.DataFrame | None = None

                if not args.regime_only:
                    bundle_signals = generate_strategy_signals(
                        data,
                        bundle_regimes.labels,
                        selected_strategies=selected_strategies,
                    )
                    bundle_signals_frame = _signals_to_frame(data, bundle_signals)
                    if args.walk_forward:
                        bundle_strategy_artifacts = build_walk_forward_strategy_artifacts(
                            data=data,
                            regimes=bundle_regimes,
                            signals=bundle_signals,
                            initial_capital=args.initial_capital,
                            transaction_cost_bps=args.transaction_cost_bps,
                            apply_train_regime_filter=args.regime_filter_train,
                            filter_metric=args.regime_filter_metric,
                            filter_threshold=args.regime_filter_threshold,
                        )
                        bundle_backtests = bundle_strategy_artifacts.backtests
                        bundle_strategy_metrics_oos = bundle_strategy_artifacts.overall_oos
                        bundle_strategy_per_regime_metrics_oos = bundle_strategy_artifacts.per_regime_oos
                        bundle_strategy_regime_rankings_oos = bundle_strategy_artifacts.ranking_oos
                    else:
                        bundle_backtests = run_strategy_backtests(
                            data,
                            bundle_signals,
                            bundle_regimes.labels,
                            initial_capital=args.initial_capital,
                            transaction_cost_bps=args.transaction_cost_bps,
                        )
                    bundle_returns = bundle_backtests.returns
                    bundle_equity = _downsample_equity_curves(
                        bundle_backtests.equity_curves,
                        args.max_equity_points_per_strategy,
                    )
                    bundle_metrics = bundle_backtests.metrics
                    bundle_per_regime_metrics = bundle_backtests.per_regime_metrics
                    bundle_trades = bundle_backtests.trades
                    bundle_stress = run_stress_suite(
                        bundle_backtests.returns,
                        _stress_regime_frame(bundle_backtests.returns, bundle_regimes.labels),
                        simulations=args.simulations,
                        horizon=args.stress_horizon,
                        block_size=args.block_size,
                        seed=args.seed,
                        stress_models=args.stress_models,
                        regime_filters=args.stress_regimes,
                        initial_capital=args.initial_capital,
                        max_saved_paths=args.max_saved_paths,
                        drawdown_threshold=args.drawdown_threshold,
                    )
                    bundle_stress_summary = bundle_stress.summary
                    bundle_stress_paths = bundle_stress.paths

                _write_dashboard_bundle(
                    run_id=run_id,
                    bundle_dir=run_dir / artifact_subdir,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    full_history=bool(args.full_history),
                    data_summary=data_summary,
                    args=args,
                    variant=variant,
                    regimes=bundle_regimes,
                    price_regimes=bundle_price_regimes,
                    strategy_names=list(bundle_signals.keys()) if not args.regime_only else [],
                    signals=bundle_signals_frame,
                    returns=bundle_returns,
                    equity=bundle_equity,
                    metrics=bundle_metrics,
                    per_regime_metrics=bundle_per_regime_metrics,
                    trades=bundle_trades,
                    stress_summary=bundle_stress_summary,
                    stress_paths=bundle_stress_paths,
                    oos_regime_model_rankings=oos_regime_model_rankings if args.walk_forward else None,
                    oos_regime_blocks=pd.DataFrame(bundle_regimes.diagnostics.get("walk_forward_blocks", [])) if args.walk_forward else None,
                    strategy_metrics_oos=bundle_strategy_metrics_oos,
                    strategy_per_regime_metrics_oos=bundle_strategy_per_regime_metrics_oos,
                    strategy_regime_rankings_oos=bundle_strategy_regime_rankings_oos,
                )
                written_variants.append(variant)
            except Exception as exc:
                bundle_errors[str(variant.get("model_id", ""))] = str(exc)
                if bool(variant.get("is_best", False)):
                    raise

    if written_variants:
        written_ids = {str(item.get("model_id", "")) for item in written_variants}
        dashboard_variants = [item for item in dashboard_variants if str(item.get("model_id", "")) in written_ids]
        selected_id = str(regimes.diagnostics.get("selected_model", regimes.model_name))
        selected_present = False
        for rank, variant in enumerate(dashboard_variants, start=1):
            variant["rank"] = int(rank)
            variant["is_best"] = str(variant.get("model_id", "")) == selected_id
            selected_present = selected_present or bool(variant["is_best"])
        if not selected_present and dashboard_variants:
            dashboard_variants[0]["is_best"] = True

    regime_diagnostics = dict(regimes.diagnostics)
    if dashboard_variants:
        regime_diagnostics["dashboard_model_variants"] = dashboard_variants
    if bundle_errors:
        regime_diagnostics["dashboard_model_bundle_errors"] = bundle_errors
    manifest["regime_diagnostics"] = regime_diagnostics
    manifest["dashboard_model_variants"] = dashboard_variants
    manifest["strategies"] = strategy_names
    _write_json(run_dir / "regime_diagnostics.json", regime_diagnostics)

    manifest["artifacts"] = sorted({path.name for path in run_dir.iterdir() if path.is_file()} | {"manifest.json"})
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def run_pipeline(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir).expanduser() if args.output_dir else OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    symbols = args.symbols or [args.symbol]
    symbols = [symbol.upper() for symbol in symbols]
    base_run_id = args.run_id or f"market_regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dirs = []
    for symbol in symbols:
        if symbol not in {"ES", "NQ"}:
            raise ValueError(f"Unsupported symbol {symbol!r}. Use ES or NQ.")
        run_id = f"{base_run_id}_{symbol.lower()}" if len(symbols) > 1 else base_run_id
        run_dirs.append(_run_one_symbol(args, symbol, run_id, output_root))
    return run_dirs[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Market Regime Strategy Backtester pipeline")
    parser.add_argument("--symbol", choices=["ES", "NQ"], default="NQ")
    parser.add_argument("--symbols", nargs="+", choices=["ES", "NQ"], default=None)
    parser.add_argument("--data-path", default=None, help="Path to a raw 1-minute futures CSV. Required for ES because only an NQ demo file is bundled.")
    parser.add_argument("--last-years", type=int, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--full-history", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--session-start", default=None)
    parser.add_argument("--session-end", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--regimes", type=int, default=4)
    parser.add_argument("--short-window", type=int, default=30)
    parser.add_argument("--long-window", type=int, default=120)
    parser.add_argument("--stability-window", type=int, default=2000)
    parser.add_argument("--include-hmm", action="store_true")
    parser.add_argument("--regime-detector", choices=["prod", "dev"], default="prod")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward (OOS) regime labeling (dev detector only)")
    parser.add_argument("--wf-train-bars", type=int, default=120_000)
    parser.add_argument("--wf-test-bars", type=int, default=20_000)
    parser.add_argument("--wf-step-bars", type=int, default=None)
    parser.add_argument("--wf-min-train-bars", type=int, default=50_000)
    parser.add_argument("--regime-filter-train", action="store_true", help="Learn per-strategy allowed regimes on train, apply to test (dev walk-forward only)")
    parser.add_argument("--regime-filter-metric", choices=["sharpe", "expectancy"], default="sharpe")
    parser.add_argument("--regime-filter-threshold", type=float, default=0.0)
    parser.add_argument("--regime-only", action="store_true")
    parser.add_argument("--no-back-adjust", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--strategies", nargs="*", default=None)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.25)
    parser.add_argument("--simulations", type=int, default=500)
    parser.add_argument("--stress-horizon", type=int, default=390)
    parser.add_argument("--stress-models", nargs="+", default=["gbm", "regime_mixture_gbm", "block_bootstrap"])
    parser.add_argument("--stress-regimes", nargs="+", default=["all", "each"])
    parser.add_argument("--block-size", type=int, default=30)
    parser.add_argument("--max-saved-paths", type=int, default=50)
    parser.add_argument("--max-equity-points-per-strategy", type=int, default=10000)
    parser.add_argument("--drawdown-threshold", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=181)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = run_pipeline(args)
    print(f"Market Regime Strategy Backtester run complete: {run_dir}")
    print("Open the dashboard with: streamlit run regime_strategy_backtester/app.py")


if __name__ == "__main__":
    main()
