from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ACTIVE_TABLE_FILES = {
    "price_regimes": "price_regimes.csv",
    "regime_stats": "regime_stats.csv",
    "regime_stability": "regime_stability.csv",
    "regime_transitions": "regime_transitions.csv",
    "signals": "strategy_signals.csv",
    "equity": "equity_curves.csv",
    "metrics": "metrics.csv",
    "per_regime_metrics": "per_regime_metrics.csv",
    "trades": "trades.csv",
    "stress_summary": "stress_summary.csv",
    "monte_carlo_paths": "monte_carlo_paths.csv",
    "strategy_metrics_oos": "strategy_metrics_oos.csv",
    "strategy_per_regime_metrics_oos": "strategy_per_regime_metrics_oos.csv",
    "strategy_regime_rankings_oos": "strategy_regime_rankings_oos.csv",
    "oos_regime_blocks": "oos_regime_blocks.csv",
}

OPTIONAL_LARGE_TABLES = {
    "regime_features": "regime_features.csv",
    "returns": "strategy_returns.csv",
    "stress_paths": "stress_paths.csv",
}

ROOT_TABLE_FILES = {
    "oos_regime_model_rankings": "oos_regime_model_rankings.csv",
}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_model_variants(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    variants = manifest.get("dashboard_model_variants")
    if isinstance(variants, list) and variants:
        return [dict(item) for item in variants if isinstance(item, dict) and item.get("model_id")]

    model_id = str(manifest.get("regime_diagnostics", {}).get("selected_model", manifest.get("regime_model", ""))).strip()
    if not model_id:
        model_id = str(manifest.get("regime_model", "current_model")).strip() or "current_model"
    return [
        {
            "model_id": model_id,
            "display_label": model_id,
            "artifact_subdir": None,
            "recommendation_source": "legacy",
            "rank_metric": "selected",
            "rank_value": 1.0,
            "rank": 1,
            "is_best": True,
        }
    ]


def _select_model_variant(variants: list[dict[str, Any]], model_id: str | None) -> dict[str, Any]:
    if model_id is not None:
        for variant in variants:
            if str(variant.get("model_id", "")) == model_id:
                return variant
    for variant in variants:
        if bool(variant.get("is_best", False)):
            return variant
    return variants[0]


def _table_map_from_dir(base_dir: Path, files: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {key: _read_csv_if_exists(base_dir / filename) for key, filename in files.items()}


def _parse_dates(payload: dict[str, Any]) -> None:
    for key in ("price_regimes", "equity", "signals", "returns"):
        frame = payload.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if isinstance(payload.get("regime_stability"), pd.DataFrame) and not payload["regime_stability"].empty:
        for column in ("window_start", "window_end"):
            if column in payload["regime_stability"].columns:
                payload["regime_stability"][column] = pd.to_datetime(payload["regime_stability"][column], errors="coerce")
    if isinstance(payload.get("trades"), pd.DataFrame) and not payload["trades"].empty:
        for column in ("entry_date", "exit_date"):
            if column in payload["trades"].columns:
                payload["trades"][column] = pd.to_datetime(payload["trades"][column], errors="coerce")


def load_dashboard_run(run_dir: str | Path, model_id: str | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    manifest = _read_json_if_exists(run_path / "manifest.json")
    model_variants = _resolve_model_variants(manifest)
    selected_variant = _select_model_variant(model_variants, model_id)

    active_dir = run_path / str(selected_variant.get("artifact_subdir", "")) if selected_variant.get("artifact_subdir") else run_path
    model_manifest = _read_json_if_exists(active_dir / "manifest.json")

    payload: dict[str, Any] = {
        "run_dir": str(run_path),
        "active_run_dir": str(active_dir),
        "manifest": manifest,
        "model_manifest": model_manifest,
        "model_variants": model_variants,
        "selected_model_id": str(selected_variant.get("model_id", "")),
        "selected_model": selected_variant,
    }

    payload.update(_table_map_from_dir(active_dir, ACTIVE_TABLE_FILES))
    payload.update(_table_map_from_dir(run_path, ROOT_TABLE_FILES))

    for key in OPTIONAL_LARGE_TABLES:
        payload[key] = pd.DataFrame()
    for key, filename in OPTIONAL_LARGE_TABLES.items():
        payload[key] = _read_csv_if_exists(active_dir / filename)

    if payload["monte_carlo_paths"].empty and not payload["stress_paths"].empty:
        payload["monte_carlo_paths"] = payload["stress_paths"].copy()
    elif payload["stress_paths"].empty and not payload["monte_carlo_paths"].empty:
        payload["stress_paths"] = payload["monte_carlo_paths"].copy()

    _parse_dates(payload)
    return payload
