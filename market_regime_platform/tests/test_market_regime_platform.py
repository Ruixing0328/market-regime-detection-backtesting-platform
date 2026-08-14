from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from market_regime_platform.app import _mc_band_frame, _mc_local_focus_options, _sort_metric_frame, _strategy_preset_selection
from market_regime_platform.backtesting import run_strategy_backtests
from market_regime_platform.config import resolve_data_path
from market_regime_platform.data import load_futures_data
from market_regime_platform.regimes import detect_regimes
from market_regime_platform.regimes.detection import REGIME_NAMES
from market_regime_platform.pipeline import _normalize_dashboard_model_variants
from market_regime_platform.strategies import generate_strategy_signals
from market_regime_platform.stress import run_stress_suite
from market_regime_platform.viz import load_dashboard_run


def _synthetic_ohlcv(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(181)
    range_low = rng.normal(0.0, 0.00020, rows // 4)
    trend_low = rng.normal(0.00007, 0.00022, rows // 4)
    trend_high = rng.normal(0.00012, 0.00120, rows // 4)
    range_high = rng.normal(0.0, 0.00160, rows - 3 * (rows // 4))
    returns = np.concatenate([range_low, trend_low, trend_high, range_high])
    close = 100.0 * np.cumprod(1.0 + returns)
    date = pd.date_range("2024-01-01 09:30", periods=rows, freq="min")
    spread = np.maximum(0.05, close * np.abs(returns) * 4.0)
    frame = pd.DataFrame(
        {
            "date": date,
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.arange(rows) + 100,
            "symbol": "NQ",
        }
    )
    return frame


def _write_dashboard_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_top5"
    run_dir.mkdir()
    dates = pd.date_range("2024-01-01 09:30", periods=12, freq="min")
    strategies = [f"strat_{idx}" for idx in range(1, 7)]

    manifest = {
        "run_id": "run_top5",
        "symbol": "NQ",
        "mode": "full",
        "regime_model": "rolling_return_volatility_kmeans::kmeans_quadrant",
        "regime_diagnostics": {"selected_model": "rolling_return_volatility_kmeans::kmeans_quadrant"},
        "dashboard_model_variants": [
            {
                "model_id": "rolling_return_volatility_kmeans::kmeans_quadrant",
                "display_label": "Rolling Return Volatility KMeans / KMeans Quadrant",
                "is_best": True,
                "rank_metric": "oos_score_mean",
                "rank_value": 0.65,
            },
            {
                "model_id": "rolling_return_volatility_kmeans::gaussian_hmm_quadrant",
                "display_label": "Rolling Return Volatility KMeans / Gaussian HMM Quadrant",
                "artifact_subdir": "dashboard_models/model_hmm",
                "is_best": False,
                "rank_metric": "model_score",
                "rank_value": 0.54,
            },
        ],
        "data_summary": {
            "rows": 100000,
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
        },
        "costs": {"initial_capital": 100000.0, "transaction_cost_bps": 0.25},
        "created_at": "2026-04-21T12:00:00",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    price = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(100, 111, len(dates)),
            "high": np.linspace(101, 112, len(dates)),
            "low": np.linspace(99, 110, len(dates)),
            "close": np.linspace(100, 111, len(dates)),
            "volume": np.linspace(1000, 1500, len(dates)),
            "symbol": ["NQ"] * len(dates),
            "regime_id": [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],
            "regime_name": [
                "Range + Low Volatility",
                "Range + Low Volatility",
                "Trend + Low Volatility",
                "Trend + Low Volatility",
                "Trend + High Volatility",
                "Trend + High Volatility",
                "Range + High Volatility",
                "Range + High Volatility",
                "Range + Low Volatility",
                "Trend + Low Volatility",
                "Trend + High Volatility",
                "Range + High Volatility",
            ],
        }
    )
    price.to_csv(run_dir / "price_regimes.csv", index=False)

    metrics_rows = []
    per_regime_rows = []
    returns_frame = pd.DataFrame({"date": dates})
    signals_frame = pd.DataFrame({"date": dates})
    trades_rows = []
    equity_rows = []
    stress_summary_rows = []
    mc_path_rows = []
    for idx, strategy in enumerate(strategies, start=1):
        expectancy = 0.12 - idx * 0.01
        sharpe = 2.0 - idx * 0.1
        total_return = 0.15 - idx * 0.01
        win_rate = 0.60 - idx * 0.02
        strength = 7 - idx
        returns = np.linspace(0.0002 * strength, 0.00025 * strength, len(dates))
        returns_frame[strategy] = returns
        signals_frame[strategy] = np.where(np.arange(len(dates)) % 2 == 0, 1.0, 0.0)
        equity = 100000.0 * (1.0 + pd.Series(returns)).cumprod()
        equity_rows.append(pd.DataFrame({"date": dates, "strategy": strategy, "equity": equity, "return": returns}))
        metrics_rows.append(
            {
                "strategy": strategy,
                "bars": len(dates),
                "total_return": total_return,
                "expectancy": expectancy,
                "sharpe": sharpe,
                "sortino": sharpe + 0.2,
                "max_drawdown": -0.01 * idx,
                "var_95": -0.002 * idx,
                "cvar_95": -0.003 * idx,
                "bar_win_rate": win_rate,
                "turnover": idx,
                "num_trades": 5 + idx,
                "win_rate": win_rate,
                "profit_factor": 1.1 + idx * 0.1,
                "expected_r": 0.2 + idx * 0.02,
            }
        )
        for regime_id, regime_name in enumerate(price["regime_name"].drop_duplicates().tolist()[:4]):
            per_regime_rows.append(
                {
                    "strategy": strategy,
                    "regime_id": regime_id,
                    "regime_name": regime_name,
                    "bars": 3,
                    "total_return": total_return / 4,
                    "expectancy": expectancy,
                    "sharpe": sharpe,
                    "sortino": sharpe + 0.1,
                    "max_drawdown": -0.01 * idx,
                    "var_95": -0.002 * idx,
                    "cvar_95": -0.003 * idx,
                    "bar_win_rate": win_rate,
                    "win_rate": win_rate,
                }
            )
        trades_rows.append(
            {
                "strategy": strategy,
                "entry_date": dates[0],
                "exit_date": dates[-1],
                "side": "long",
                "entry_exposure": 1.0,
                "exit_exposure": 0.0,
                "entry_price": 100.0,
                "exit_price": 101.0,
                "return": 0.01,
                "pnl": 100.0 * idx,
                "bars_held": len(dates),
                "exit_reason": "time",
                "r_multiple_proxy": 0.5,
            }
        )
        for regime_filter in ["all", *price["regime_name"].drop_duplicates().tolist()]:
            stress_summary_rows.append(
                {
                    "strategy": strategy,
                    "stress_model": "gbm",
                    "regime_filter": regime_filter,
                    "simulations": 10,
                    "horizon": 20,
                    "terminal_mean": 0.01,
                    "terminal_std": 0.02,
                    "terminal_p5": -0.03,
                    "terminal_p50": 0.01,
                    "terminal_p95": 0.04,
                    "terminal_cvar_95": -0.04,
                    "terminal_equity_p5": 97000.0,
                    "terminal_equity_p50": 101000.0,
                    "terminal_equity_p95": 104000.0,
                    "max_drawdown_p50": -0.02,
                    "max_drawdown_p95": -0.05,
                    "probability_of_loss": 0.35,
                    "prob_drawdown_exceed": 0.10,
                    "risk_of_ruin": 0.01,
                }
            )
            for simulation in range(3):
                for step in range(1, 6):
                    mc_path_rows.append(
                        {
                            "strategy": strategy,
                            "stress_model": "gbm",
                            "regime_filter": regime_filter,
                            "simulation": simulation,
                            "step": step,
                            "return": 0.001 * step,
                            "equity": 100000.0 + simulation * 50 + step * 100 + idx * 10,
                        }
                    )

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(per_regime_rows).to_csv(run_dir / "per_regime_metrics.csv", index=False)
    returns_frame.to_csv(run_dir / "strategy_returns.csv", index=False)
    signals_frame.to_csv(run_dir / "strategy_signals.csv", index=False)
    pd.concat(equity_rows, ignore_index=True).to_csv(run_dir / "equity_curves.csv", index=False)
    pd.DataFrame(trades_rows).to_csv(run_dir / "trades.csv", index=False)
    pd.DataFrame(stress_summary_rows).to_csv(run_dir / "stress_summary.csv", index=False)
    pd.DataFrame(mc_path_rows).to_csv(run_dir / "monte_carlo_paths.csv", index=False)
    pd.DataFrame({"regime_id": [0], "regime_name": ["Range + Low Volatility"], "bars": [len(dates)]}).to_csv(run_dir / "regime_stats.csv", index=False)
    pd.DataFrame({"window_start": [dates[0]], "window_end": [dates[-1]], "bars": [len(dates)], "regime_0_share": [0.25], "regime_1_share": [0.25], "regime_2_share": [0.25], "regime_3_share": [0.25]}).to_csv(run_dir / "regime_stability.csv", index=False)
    pd.DataFrame({"from_regime": [0], 0: [0.8], 1: [0.2]}).to_csv(run_dir / "regime_transitions.csv", index=False)
    pd.DataFrame({"date": dates, "vol_axis": np.linspace(0.1, 0.2, len(dates)), "structure_axis": np.linspace(0.2, 0.4, len(dates))}).to_csv(run_dir / "regime_features.csv", index=False)
    bundle_dir = run_dir / "dashboard_models" / "model_hmm"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_top5::rolling_return_volatility_kmeans::gaussian_hmm_quadrant",
                "regime_model": "rolling_return_volatility_kmeans::gaussian_hmm_quadrant",
                "dashboard_model_variant": {
                    "model_id": "rolling_return_volatility_kmeans::gaussian_hmm_quadrant",
                    "display_label": "Rolling Return Volatility KMeans / Gaussian HMM Quadrant",
                },
            }
        )
    )
    pd.DataFrame(metrics_rows).assign(expectancy=lambda frame: frame["expectancy"] * 0.95).to_csv(bundle_dir / "metrics.csv", index=False)
    returns_frame.to_csv(bundle_dir / "strategy_returns.csv", index=False)
    pd.concat(equity_rows, ignore_index=True).to_csv(bundle_dir / "equity_curves.csv", index=False)
    price.to_csv(bundle_dir / "price_regimes.csv", index=False)
    return run_dir


def test_load_futures_data_parses_raw_sample(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "ts_event": pd.date_range("2024-01-01 14:30", periods=4, freq="min", tz="UTC").astype(str),
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10, 20, 30, 40],
            "symbol": ["NQH6", "NQH6", "NQH6", "NQH6"],
        }
    )
    path = tmp_path / "nq.csv"
    raw.to_csv(path, index=False)

    data, summary = load_futures_data("NQ", data_path=path, sample_rows=3)

    assert len(data) == 3
    assert summary.symbol == "NQ"
    assert summary.rows == 3
    assert set(["date", "open", "high", "low", "close", "volume", "symbol"]).issubset(data.columns)


def test_default_nq_demo_data_is_bundled_inside_package() -> None:
    path = resolve_data_path("NQ")

    assert path.exists()
    assert path.name == "NQ_1M_demo.csv"
    assert "market_regime_platform/data/demo" in path.as_posix()


def test_es_requires_explicit_data_path_when_not_bundled() -> None:
    with pytest.raises(FileNotFoundError, match="No bundled dataset"):
        load_futures_data("ES")


def test_regime_model_uses_required_names_and_tournament_diagnostics() -> None:
    data = _synthetic_ohlcv(1000)
    result = detect_regimes(data, short_window=20, long_window=80, stability_window=200)

    assert sorted(result.labels["regime_name"].dropna().unique().tolist()) == sorted(REGIME_NAMES)
    assert result.diagnostics["selected_model"] in result.diagnostics["candidate_scores"]
    assert "interpretability_score" in result.diagnostics
    assert result.hmm_status == "not_requested"


def test_strategy_registry_generates_all_variants_and_half_size_regime_filter() -> None:
    data = _synthetic_ohlcv(900)
    regimes = detect_regimes(data, short_window=20, long_window=80)
    signals = generate_strategy_signals(data, regimes.labels)

    for base in ["sma_cross", "donchian", "rsi_mr", "zscore_mr", "ml_rf", "ml_logreg", "ml_gbrt"]:
        assert f"{base}__baseline" in signals
        assert f"{base}__regime_aware" in signals
        assert f"{base}__liquidity_filtered" in signals
    for signal in signals.values():
        assert set(signal.dropna().unique()).issubset({-1.0, -0.5, 0.0, 0.5, 1.0})
    risk_regimes = regimes.labels.copy()
    risk_regimes["regime_name"] = "Trend + High Volatility"
    risk_signals = generate_strategy_signals(data, risk_regimes, selected_strategies=["sma_cross"])
    first_regime_idx = int(risk_regimes["source_index"].min())
    active = (risk_signals["sma_cross__baseline"].abs() > 0) & (risk_signals["sma_cross__baseline"].index >= first_regime_idx)
    assert (risk_signals["sma_cross__regime_aware"].loc[active].abs() == 0.5).all()


def test_backtest_uses_next_bar_execution_and_reports_win_rate() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="min"),
            "open": [100, 110, 121],
            "high": [100, 110, 121],
            "low": [100, 110, 121],
            "close": [100, 110, 121],
            "volume": [1, 1, 1],
            "symbol": ["NQ", "NQ", "NQ"],
        }
    )
    signal = pd.Series([1.0, 0.0, 0.0], index=data.index)
    result = run_strategy_backtests(data, {"test": signal}, transaction_cost_bps=0.0)

    returns = result.returns["test"].round(10).to_list()
    assert returns == [0.0, 0.1, 0.0]
    assert {"num_trades", "profit_factor", "expected_r", "win_rate"}.issubset(result.metrics.columns)
    assert int(result.metrics.iloc[0]["num_trades"]) == 1
    assert float(result.metrics.iloc[0]["win_rate"]) == 1.0


def test_per_regime_bar_counts_match_global_rows() -> None:
    data = _synthetic_ohlcv(900)
    regimes = detect_regimes(data, short_window=20, long_window=80)
    signals = generate_strategy_signals(data, regimes.labels, selected_strategies=["sma_cross"])
    result = run_strategy_backtests(data, {"sma_cross__baseline": signals["sma_cross__baseline"]}, regimes.labels)

    assert int(result.per_regime_metrics["bars"].sum()) == len(regimes.labels)
    assert "win_rate" in result.per_regime_metrics.columns


def test_stress_suite_outputs_path_level_monte_carlo() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=80, freq="min"),
            "strategy_a": np.linspace(-0.001, 0.001, 80),
        }
    )
    regimes = pd.DataFrame(
        {
            "source_index": np.arange(80),
            "regime_name": np.where(np.arange(80) < 40, "Range + Low Volatility", "Trend + High Volatility"),
        }
    )
    stress = run_stress_suite(returns, regimes, simulations=25, horizon=20, block_size=5, seed=7, regime_filters=["all", "each"], max_saved_paths=3)

    assert set(stress.summary["stress_model"]) == {"gbm", "regime_mixture_gbm", "block_bootstrap"}
    assert {"strategy", "regime_filter", "stress_model", "simulation", "step", "return", "equity"}.issubset(stress.paths.columns)
    assert stress.paths["simulation"].nunique() == 3
    assert (stress.summary["terminal_p5"] <= stress.summary["terminal_p50"]).all()
    assert (stress.summary["terminal_p50"] <= stress.summary["terminal_p95"]).all()


def test_dashboard_loader_reads_manifest_and_monte_carlo_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"run_id":"run","symbol":"NQ"}')
    pd.DataFrame({"strategy": ["a"], "sharpe": [1.0]}).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "strategy": ["a"],
            "stress_model": ["gbm"],
            "regime_filter": ["all"],
            "simulation": [0],
            "step": [1],
            "return": [0.01],
            "equity": [101000.0],
        }
    ).to_csv(run_dir / "monte_carlo_paths.csv", index=False)

    payload = load_dashboard_run(run_dir)

    assert payload["manifest"]["symbol"] == "NQ"
    assert payload["metrics"].iloc[0]["strategy"] == "a"
    assert payload["monte_carlo_paths"].iloc[0]["equity"] == 101000.0


def test_dashboard_loader_defaults_to_recommended_model_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bundle_dir = run_dir / "dashboard_models" / "model_b"
    bundle_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        """
        {
          "run_id": "run",
          "symbol": "NQ",
          "dashboard_model_variants": [
            {
              "model_id": "candidate_a::kmeans_quadrant",
              "display_label": "Candidate A / KMeans Quadrant",
              "artifact_subdir": "dashboard_models/model_a",
              "is_best": false,
              "recommendation_source": "oos",
              "rank_metric": "oos_score_mean",
              "rank_value": 0.4
            },
            {
              "model_id": "candidate_b::gaussian_hmm_quadrant",
              "display_label": "Candidate B / Gaussian HMM Quadrant",
              "artifact_subdir": "dashboard_models/model_b",
              "is_best": true,
              "recommendation_source": "oos",
              "rank_metric": "oos_score_mean",
              "rank_value": 0.7
            }
          ]
        }
        """
    )
    pd.DataFrame(
        {
            "candidate": ["candidate_a", "candidate_b"],
            "model_family": ["kmeans_quadrant", "gaussian_hmm_quadrant"],
            "status": ["ok", "ok"],
            "oos_score_mean": [0.4, 0.7],
        }
    ).to_csv(run_dir / "oos_regime_model_rankings.csv", index=False)
    (bundle_dir / "manifest.json").write_text('{"regime_model":"candidate_b::gaussian_hmm_quadrant::wf"}')
    pd.DataFrame({"strategy": ["winner"], "expectancy": [0.12], "sharpe": [1.8]}).to_csv(bundle_dir / "metrics.csv", index=False)
    pd.DataFrame({"date": ["2024-01-01 09:30:00"], "winner": [0.001]}).to_csv(bundle_dir / "strategy_returns.csv", index=False)
    pd.DataFrame({"date": ["2024-01-01 09:30:00"], "strategy": ["winner"], "equity": [100100.0], "return": [0.001]}).to_csv(bundle_dir / "equity_curves.csv", index=False)
    pd.DataFrame({"date": ["2024-01-01 09:30:00"], "close": [100.0], "regime_id": [1], "regime_name": ["Trend + Low Volatility"]}).to_csv(bundle_dir / "price_regimes.csv", index=False)

    payload = load_dashboard_run(run_dir)

    assert payload["selected_model_id"] == "candidate_b::gaussian_hmm_quadrant"
    assert payload["metrics"].iloc[0]["strategy"] == "winner"
    assert payload["oos_regime_model_rankings"].iloc[1]["candidate"] == "candidate_b"


def test_prod_dashboard_variants_expose_explicit_model_families() -> None:
    data = _synthetic_ohlcv(1000)
    regimes = detect_regimes(data, short_window=20, long_window=80, stability_window=200)
    args = SimpleNamespace(regime_detector="prod", walk_forward=False, include_hmm=True)

    variants = _normalize_dashboard_model_variants(args, regimes)
    model_ids = {str(item["model_id"]) for item in variants}
    selected_candidate = str(regimes.diagnostics["selected_model"])

    assert f"{selected_candidate}::kmeans_quadrant" in model_ids
    if importlib.util.find_spec("hmmlearn") is not None:
        assert f"{selected_candidate}::gaussian_hmm_quadrant" in model_ids


def test_dashboard_loader_can_select_nondefault_model_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    bundle_a = run_dir / "dashboard_models" / "model_a"
    bundle_b = run_dir / "dashboard_models" / "model_b"
    bundle_a.mkdir(parents=True)
    bundle_b.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        """
        {
          "run_id": "run",
          "symbol": "NQ",
          "dashboard_model_variants": [
            {"model_id": "candidate_a::kmeans_quadrant", "display_label": "Candidate A", "artifact_subdir": "dashboard_models/model_a", "is_best": false},
            {"model_id": "candidate_b::gaussian_hmm_quadrant", "display_label": "Candidate B", "artifact_subdir": "dashboard_models/model_b", "is_best": true}
          ]
        }
        """
    )
    (bundle_a / "manifest.json").write_text("{}")
    (bundle_b / "manifest.json").write_text("{}")
    pd.DataFrame({"strategy": ["alpha"], "expectancy": [0.02]}).to_csv(bundle_a / "metrics.csv", index=False)
    pd.DataFrame({"strategy": ["beta"], "expectancy": [0.08]}).to_csv(bundle_b / "metrics.csv", index=False)

    payload = load_dashboard_run(run_dir, model_id="candidate_a::kmeans_quadrant")

    assert payload["selected_model_id"] == "candidate_a::kmeans_quadrant"
    assert payload["metrics"].iloc[0]["strategy"] == "alpha"


def test_sort_metric_frame_uses_expectancy_then_sharpe_then_bars() -> None:
    frame = pd.DataFrame(
        {
            "strategy": ["a", "b", "c"],
            "expectancy": [0.10, 0.10, 0.05],
            "sharpe": [1.2, 1.6, 2.0],
            "bars": [100, 90, 200],
        }
    )

    ranked = _sort_metric_frame(frame, "expectancy")

    assert ranked["strategy"].tolist() == ["b", "a", "c"]


def test_strategy_preset_selection_supports_top5_top3_and_all() -> None:
    ranked = [f"s{idx}" for idx in range(1, 7)]

    assert _strategy_preset_selection(ranked, "top_5") == ["s1", "s2", "s3", "s4", "s5"]
    assert _strategy_preset_selection(ranked, "top_3") == ["s1", "s2", "s3"]
    assert _strategy_preset_selection(ranked, "all") == ranked
    assert _strategy_preset_selection(ranked, "champion_plus_challengers") == ["s1", "s2", "s3", "s4", "s5"]


def test_mc_band_frame_builds_percentile_columns() -> None:
    paths = pd.DataFrame(
        {
            "step": [1, 1, 1, 2, 2, 2],
            "equity": [100000.0, 101000.0, 102000.0, 103000.0, 104000.0, 105000.0],
        }
    )

    bands = _mc_band_frame(paths)

    assert {"step", "p05", "p25", "p50", "p75", "p95", "mean"}.issubset(bands.columns)
    assert len(bands) == 2
    assert float(bands.loc[bands["step"] == 1, "p50"].iloc[0]) == 101000.0


def test_mc_local_focus_options_respect_global_scope() -> None:
    summary = pd.DataFrame(
        {
            "strategy": ["alpha", "alpha", "beta", "beta"],
            "stress_model": ["gbm", "gbm", "block_bootstrap", "block_bootstrap"],
            "regime_filter": ["all", "Trend + High Volatility", "all", "Range + Low Volatility"],
        }
    )

    options = _mc_local_focus_options(summary, ["beta", "alpha"], ["Range + Low Volatility"])

    assert options["strategies"] == ["beta", "alpha"]
    assert options["regimes"] == ["Range + Low Volatility"]
    assert options["models"] == ["block_bootstrap", "gbm"]
    assert options["default_strategy"] == "beta"
    assert options["default_regime"] == "Range + Low Volatility"


def test_dashboard_app_renders_command_bar_and_top5_defaults(tmp_path: Path) -> None:
    _write_dashboard_run(tmp_path)
    at = AppTest.from_file("market_regime_platform/app.py").run(timeout=120)
    at.text_input[0].input(str(tmp_path)).run(timeout=120)

    assert not at.exception
    multiselect_labels = {widget.label for widget in at.multiselect}
    radio_labels = {widget.label for widget in at.radio}
    selectbox_labels = {widget.label for widget in at.selectbox}
    button_labels = {widget.label for widget in at.button}
    assert {"Regimes", "Strategies in view"}.issubset(multiselect_labels)
    assert {"Ranking lens", "Strategy preset"}.issubset(radio_labels)
    assert {"Monte Carlo strategy", "Monte Carlo regime", "Stress model", "Path density", "Distribution metric"}.issubset(selectbox_labels)
    assert "Reset Monte Carlo view" in button_labels
    strategies_widget = next(widget for widget in at.multiselect if widget.label == "Strategies in view")
    assert len(strategies_widget.value) == 5
    assert strategies_widget.value == ["strat_1", "strat_2", "strat_3", "strat_4", "strat_5"]


def test_dashboard_app_explains_single_model_or_stress_scope_limits(tmp_path: Path) -> None:
    run_dir = tmp_path / "single_scope_run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "single_scope_run",
                "symbol": "NQ",
                "mode": "full",
                "regime_model": "rolling_return_volatility_kmeans::kmeans_quadrant",
                "dashboard_model_variants": [
                    {
                        "model_id": "rolling_return_volatility_kmeans::kmeans_quadrant",
                        "display_label": "Rolling Return Volatility KMeans / KMeans Quadrant",
                        "is_best": True,
                    }
                ],
                "data_summary": {"rows": 100000, "start_date": "2024-01-01", "end_date": "2024-01-02"},
                "costs": {"initial_capital": 100000.0, "transaction_cost_bps": 0.25},
                "created_at": "2026-04-21T12:00:00",
            }
        )
    )
    dates = pd.date_range("2024-01-01 09:30", periods=4, freq="min")
    pd.DataFrame(
        {
            "date": dates,
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100, 101, 102, 103],
            "volume": [10, 20, 30, 40],
            "symbol": ["NQ"] * len(dates),
            "regime_id": [0, 1, 2, 3],
            "regime_name": REGIME_NAMES,
        }
    ).to_csv(run_dir / "price_regimes.csv", index=False)
    pd.DataFrame({"strategy": ["alpha"], "expectancy": [0.1], "sharpe": [1.2], "total_return": [0.05], "win_rate": [0.6], "expected_r": [0.2], "max_drawdown": [-0.05], "num_trades": [5]}).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame({"date": dates, "alpha": [0.0, 0.01, -0.005, 0.02]}).to_csv(run_dir / "strategy_returns.csv", index=False)
    pd.DataFrame({"date": dates, "strategy": ["alpha"] * len(dates), "equity": [100000, 101000, 100500, 102500], "return": [0.0, 0.01, -0.005, 0.02]}).to_csv(run_dir / "equity_curves.csv", index=False)
    pd.DataFrame(
        {
            "strategy": ["alpha"],
            "stress_model": ["gbm"],
            "regime_filter": ["all"],
            "simulations": [5],
            "horizon": [20],
            "terminal_mean": [0.01],
            "terminal_std": [0.02],
            "terminal_p5": [-0.02],
            "terminal_p50": [0.01],
            "terminal_p95": [0.03],
            "terminal_cvar_95": [-0.03],
            "terminal_equity_p5": [98000.0],
            "terminal_equity_p50": [101000.0],
            "terminal_equity_p95": [103000.0],
            "max_drawdown_p50": [-0.02],
            "max_drawdown_p95": [-0.04],
            "probability_of_loss": [0.4],
            "prob_drawdown_exceed": [0.1],
            "risk_of_ruin": [0.01],
        }
    ).to_csv(run_dir / "stress_summary.csv", index=False)
    pd.DataFrame(
        {
            "strategy": ["alpha"],
            "stress_model": ["gbm"],
            "regime_filter": ["all"],
            "simulation": [0],
            "step": [1],
            "return": [0.01],
            "equity": [101000.0],
        }
    ).to_csv(run_dir / "monte_carlo_paths.csv", index=False)
    pd.DataFrame({"regime_id": [0], "regime_name": ["Range + Low Volatility"], "bars": [1]}).to_csv(run_dir / "regime_stats.csv", index=False)
    pd.DataFrame({"window_start": [dates[0]], "window_end": [dates[-1]], "bars": [len(dates)], "regime_0_share": [0.25], "regime_1_share": [0.25], "regime_2_share": [0.25], "regime_3_share": [0.25]}).to_csv(run_dir / "regime_stability.csv", index=False)
    pd.DataFrame({"from_regime": [0], 0: [1.0]}).to_csv(run_dir / "regime_transitions.csv", index=False)

    at = AppTest.from_file("market_regime_platform/app.py").run(timeout=120)
    at.text_input[0].input(str(tmp_path)).run(timeout=120)

    info_values = [widget.value for widget in at.info]
    assert any("only contains one saved regime-model variant" in value for value in info_values)
    assert any("only includes saved Monte Carlo artifacts" in value for value in info_values)


def test_market_regime_platform_has_no_external_repo_dependencies() -> None:
    root = Path("market_regime_platform")
    substring_checks = [
        str(Path.home()),
        str(Path("Desktop") / "trading" / "quant algo"),
        "backtest.intraday_futures_data",
        "data/intraday/NQ 1M historical data.csv",
        "data/intraday/ES 1M historical data.csv",
        "reports/state",
    ]
    regex_checks = [
        r"\bfrom backtest\b",
        r"\bimport backtest\b",
    ]

    checked_files = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", "output", "tests"} for part in path.parts):
            continue
        if path.suffix not in {".py", ".md", ".json", ".txt"}:
            continue

        checked_files += 1
        text = path.read_text(errors="ignore")
        for needle in substring_checks:
            assert needle not in text, f"{path} still references external path or repo-only artifact: {needle}"
        for pattern in regex_checks:
            assert re.search(pattern, text) is None, f"{path} still imports repo-level backtest code: {pattern}"

    assert checked_files > 0
