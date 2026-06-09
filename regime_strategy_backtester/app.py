from __future__ import annotations

from html import escape
import math
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regime_strategy_backtester.config import OUTPUT_ROOT
from regime_strategy_backtester.config import DEFAULT_BARS_PER_YEAR
from regime_strategy_backtester.viz import load_dashboard_run
from regime_strategy_backtester.backtesting.vectorized import _metrics_from_returns


REGIME_COLORS = ["#6ee7a8", "#8ecae6", "#f2c94c", "#f47b9d"]
CHART_COLORS = ["#8ecae6", "#6ee7a8", "#f47b9d", "#f2c94c", "#c7d2fe", "#fca5a5", "#86efac"]


def _manifest_for_run(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        import json

        return json.loads(manifest_path.read_text())
    except Exception:
        return {}


def _is_primary_dashboard_run(path: Path, manifest: dict) -> bool:
    rows = int(manifest.get("data_summary", {}).get("rows", 0) or 0)
    return (
        manifest.get("mode") == "full"
        and (path / "metrics.csv").exists()
        and (path / "equity_curves.csv").exists()
        and rows >= 100_000
        and "smoke" not in path.name.lower()
    )


def _run_sort_key(path: Path) -> tuple[int, int, int, float]:
    manifest = _manifest_for_run(path)
    rows = int(manifest.get("data_summary", {}).get("rows", 0) or 0)
    is_primary = int(_is_primary_dashboard_run(path, manifest))
    is_full = int(manifest.get("mode") == "full")
    return (is_primary, is_full, rows, (path / "manifest.json").stat().st_mtime)


def _available_runs(output_root: Path, include_smoke_runs: bool = False) -> list[Path]:
    if not output_root.exists():
        return []
    runs = [path for path in output_root.iterdir() if (path / "manifest.json").exists()]
    if not include_smoke_runs:
        runs = [path for path in runs if _is_primary_dashboard_run(path, _manifest_for_run(path))]
    return sorted(runs, key=_run_sort_key, reverse=True)


def _run_display_label(path: Path) -> str:
    manifest = _manifest_for_run(path)
    summary = manifest.get("data_summary", {})
    symbol = manifest.get("symbol", "?")
    mode = manifest.get("mode", "?")
    start = summary.get("start_date", "?")
    end = summary.get("end_date", "?")
    rows = int(summary.get("rows", 0) or 0)
    return f"{path.name} | {symbol} | {start} to {end} | {rows:,} rows | {mode}"


def _downsample(frame: pd.DataFrame, max_points: int = 6000) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    stride = max(1, len(frame) // max_points)
    return frame.iloc[::stride].copy()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #060607;
            --panel: rgba(11, 11, 13, 0.94);
            --panel-accent: rgba(3, 3, 5, 0.92);
            --panel-soft: rgba(255, 255, 255, 0.03);
            --line: rgba(141, 177, 236, 0.10);
            --line-strong: rgba(77, 227, 210, 0.22);
            --ink-strong: #eef4ff;
            --ink: #dce5f3;
            --ink-soft: #aab7cb;
            --ink-faint: #7287a6;
            --teal: #4de3d2;
            --cyan: #8fd9ff;
            --rose: #ff6f8f;
            --gold: #f3b34d;
            --shadow: 0 22px 64px rgba(0, 0, 0, 0.34);
        }
        .stApp {
            background:
                radial-gradient(circle at 14% 10%, rgba(255, 255, 255, 0.045), transparent 18rem),
                radial-gradient(circle at 88% 12%, rgba(77, 227, 210, 0.06), transparent 18rem),
                radial-gradient(circle at 82% 78%, rgba(243, 179, 77, 0.06), transparent 16rem),
                linear-gradient(180deg, #090909 0%, #050506 34%, #010101 100%);
            color: var(--ink);
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(116, 143, 185, 0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(116, 143, 185, 0.06) 1px, transparent 1px),
                linear-gradient(transparent 0, transparent 78%, rgba(77, 227, 210, 0.08) 100%);
            background-size: 64px 64px, 64px 64px, 100% 100%;
            opacity: 0.72;
            z-index: 0;
        }
        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                radial-gradient(circle at 76% 6%, rgba(90, 166, 255, 0.08), transparent 24rem),
                radial-gradient(circle at 18% 90%, rgba(44, 207, 143, 0.06), transparent 22rem),
                repeating-linear-gradient(90deg, transparent 0 34px, rgba(141, 177, 236, 0.025) 34px 35px),
                linear-gradient(180deg, transparent 0%, rgba(77, 227, 210, 0.03) 100%);
            opacity: 0.7;
            mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.8), transparent 96%);
            z-index: 0;
        }
        .block-container {
            max-width: 1380px;
            padding-top: 1.0rem;
            padding-bottom: 3rem;
            position: relative;
            z-index: 1;
        }
        header[data-testid="stHeader"], footer, #MainMenu { visibility: hidden; }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(11, 11, 13, 0.98), rgba(3, 3, 5, 0.96)),
                radial-gradient(circle at top right, rgba(77, 227, 210, 0.08), transparent 42%);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] * { color: var(--ink); }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: var(--ink-faint);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        div[data-baseweb="tag"] {
            background: rgba(255, 255, 255, 0.035);
            border-color: rgba(141, 177, 236, 0.10);
            border-radius: 8px;
            color: var(--ink-strong);
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="base-input"] input,
        div[data-baseweb="input"] input,
        .stDateInput input {
            color: var(--ink-strong) !important;
        }
        div[data-baseweb="tag"] {
            background: rgba(77, 227, 210, 0.10);
            border-color: rgba(77, 227, 210, 0.20);
        }
        button[kind="secondary"],
        button[kind="primary"] {
            border-radius: 8px !important;
            border: 1px solid rgba(141, 177, 236, 0.12) !important;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.012)),
                rgba(0, 0, 0, 0.18) !important;
            color: var(--ink-strong) !important;
            box-shadow: none !important;
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease !important;
        }
        button[kind="secondary"]:hover,
        button[kind="primary"]:hover {
            transform: translateY(-1px);
            border-color: rgba(90, 166, 255, 0.26) !important;
            background: rgba(16, 34, 58, 0.82) !important;
        }
        [data-testid="stSegmentedControl"] button,
        [data-testid="stPills"] button {
            border-radius: 8px !important;
            border: 1px solid rgba(141, 177, 236, 0.12) !important;
            background: rgba(255, 255, 255, 0.035) !important;
            color: var(--ink-soft) !important;
        }
        [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stPills"] button[aria-selected="true"],
        [data-testid="stPills"] button[aria-checked="true"] {
            border-color: rgba(77, 227, 210, 0.24) !important;
            background: rgba(44, 207, 143, 0.12) !important;
            color: var(--ink-strong) !important;
            box-shadow: 0 10px 24px rgba(44, 207, 143, 0.14);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(141, 177, 236, 0.08);
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--ink-faint);
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ink-strong);
        }
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 20px;
        }
        .brand-lockup {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 12px;
        }
        .brand-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 38px;
            padding: 0 16px;
            border-radius: 8px;
            border: 1px solid rgba(77, 227, 210, 0.28);
            background: rgba(6, 25, 24, 0.72);
            color: var(--ink-strong);
            font-size: 0.80rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            box-shadow: 0 0 0 1px rgba(77, 227, 210, 0.08), 0 16px 36px rgba(0, 0, 0, 0.26);
        }
        .brand-badge::before {
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--cyan);
            box-shadow: 0 0 12px rgba(90, 166, 255, 0.45);
        }
        .brand-copy,
        .topbar-note {
            color: var(--ink-soft);
            font-size: 0.98rem;
            font-weight: 600;
        }
        .topbar-note {
            text-align: right;
            max-width: 26rem;
        }
        .preview-command {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.82fr);
            gap: 22px;
            margin-bottom: 22px;
        }
        .command-layout {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.82fr);
            gap: 22px;
            margin-bottom: 22px;
        }
        .command-panel,
        .desk-panel,
        .preview-filter-panel,
        .lab-section-card,
        .recent-trade-panel {
            position: relative;
            overflow: hidden;
            min-width: 0;
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.11);
            background:
                linear-gradient(180deg, rgba(11, 11, 13, 0.94), rgba(3, 3, 5, 0.92)),
                repeating-linear-gradient(90deg, transparent 0 31px, rgba(141, 177, 236, 0.022) 31px 32px),
                linear-gradient(180deg, transparent 0 86%, rgba(77, 227, 210, 0.025) 100%),
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.08), transparent 42%);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }
        .command-panel::before,
        .desk-panel::before,
        .preview-filter-panel::before,
        .lab-section-card::before,
        .recent-trade-panel::before {
            content: "";
            position: absolute;
            inset: 0 0 auto;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(141, 177, 236, 0.24), transparent);
        }
        .command-panel {
            padding: 24px;
        }
        .desk-panel,
        .preview-filter-panel,
        .lab-section-card,
        .recent-trade-panel {
            padding: 18px;
        }
        .command-header,
        .panel-heading {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: start;
        }
        .command-copy {
            min-width: 0;
            position: relative;
            z-index: 1;
        }
        .section-kicker,
        .eyebrow {
            color: var(--ink-faint);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 0 0 8px;
            text-shadow: 0 0 14px rgba(90, 166, 255, 0.18);
        }
        .dashboard-title {
            margin: 0;
            font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
            font-size: clamp(2.4rem, 4.2vw, 4.2rem);
            line-height: 0.95;
            letter-spacing: -0.04em;
            color: var(--ink-strong);
            max-width: 12ch;
        }
        .subline {
            margin: 14px 0 0;
            color: var(--ink-soft);
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 54ch;
        }
        .command-badges {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 8px;
            max-width: 24rem;
            position: relative;
            z-index: 1;
        }
        .status-pill,
        .mode-pill,
        .lab-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 38px;
            padding: 0 14px;
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.14);
            background: rgba(255, 255, 255, 0.045);
            color: var(--ink);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .status-pill::before,
        .mode-pill::before {
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--teal);
            box-shadow: 0 0 0 6px rgba(44, 207, 143, 0.10);
        }
        .status-pill.negative::before,
        .mode-pill.negative::before {
            background: var(--rose);
            box-shadow: 0 0 0 6px rgba(255, 111, 143, 0.10);
        }
        .command-sparkline {
            position: relative;
            overflow: hidden;
            margin: 22px 0 18px;
            min-height: 132px;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid rgba(141, 177, 236, 0.08);
            background:
                linear-gradient(180deg, rgba(6, 10, 16, 0.92), rgba(2, 4, 8, 0.90)),
                linear-gradient(90deg, rgba(90, 166, 255, 0.05), rgba(77, 227, 210, 0.04));
        }
        .command-sparkline::before {
            content: "";
            position: absolute;
            inset: 12px;
            border-radius: 8px;
            background:
                repeating-linear-gradient(90deg, transparent 0 23px, rgba(255, 255, 255, 0.02) 23px 24px),
                linear-gradient(180deg, transparent 0 70%, rgba(255, 111, 143, 0.04) 70% 71%, transparent 71% 100%);
            pointer-events: none;
        }
        .mini-sparkline {
            width: 100%;
            height: 108px;
            display: block;
            position: relative;
            z-index: 1;
        }
        .mini-sparkline .sparkline-grid {
            stroke: rgba(141, 177, 236, 0.12);
            stroke-width: 1;
        }
        .mini-sparkline .sparkline-fill {
            fill: rgba(90, 166, 255, 0.12);
        }
        .mini-sparkline .sparkline-line {
            fill: none;
            stroke: var(--cyan);
            stroke-width: 3;
            stroke-linecap: round;
            stroke-linejoin: round;
            filter: drop-shadow(0 0 16px rgba(90, 166, 255, 0.32));
        }
        .mini-sparkline .sparkline-baseline {
            stroke: rgba(243, 179, 77, 0.28);
            stroke-dasharray: 4 4;
        }
        .mini-sparkline .sparkline-end {
            fill: #07111f;
            stroke: var(--cyan);
            stroke-width: 2;
        }
        .empty-sparkline {
            min-height: 106px;
            display: grid;
            place-items: center;
            color: var(--ink-soft);
            font-weight: 700;
            position: relative;
            z-index: 1;
        }
        .summary-grid,
        .strategy-card-grid,
        .route-metrics,
        .mc-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 18px;
        }
        .kpi-card,
        .summary-card,
        .strategy-summary-card,
        .route-metric,
        .console-card {
            position: relative;
            padding: 14px 15px;
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.08);
            background:
                linear-gradient(180deg, rgba(16, 16, 20, 0.78), rgba(6, 6, 8, 0.90)),
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.06), transparent 44%);
            isolation: isolate;
        }
        .kpi-card::before,
        .summary-card::before,
        .strategy-summary-card::before,
        .route-metric::before {
            content: "";
            position: absolute;
            left: 16px;
            right: 16px;
            top: 0;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(90, 166, 255, 0.22), rgba(255, 255, 255, 0));
        }
        .kpi-card.good,
        .summary-card.positive,
        .strategy-summary-card.positive,
        .route-metric.positive {
            box-shadow:
                inset 0 0 0 1px rgba(44, 207, 143, 0.08),
                0 0 0 1px rgba(44, 207, 143, 0.04),
                0 16px 34px rgba(7, 28, 22, 0.18);
        }
        .kpi-card.bad,
        .summary-card.negative,
        .strategy-summary-card.negative,
        .route-metric.negative {
            box-shadow:
                inset 0 0 0 1px rgba(255, 111, 143, 0.08),
                0 0 0 1px rgba(255, 111, 143, 0.04),
                0 16px 34px rgba(38, 12, 20, 0.18);
        }
        .kpi-card.good::before,
        .summary-card.positive::before,
        .strategy-summary-card.positive::before,
        .route-metric.positive::before {
            background: linear-gradient(90deg, rgba(44, 207, 143, 0.8), rgba(44, 207, 143, 0));
        }
        .kpi-card.bad::before,
        .summary-card.negative::before,
        .strategy-summary-card.negative::before,
        .route-metric.negative::before {
            background: linear-gradient(90deg, rgba(255, 111, 143, 0.8), rgba(255, 111, 143, 0));
        }
        .kpi-label,
        .console-label,
        .summary-card span,
        .strategy-summary-card span,
        .route-metric span,
        .desk-item span,
        .desk-note span {
            display: block;
            color: var(--ink-faint);
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .kpi-value,
        .console-value,
        .summary-card strong,
        .strategy-summary-card strong,
        .route-metric strong,
        .desk-item strong {
            display: block;
            color: var(--ink-strong);
            font-size: 1.18rem;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .kpi-note,
        .console-note,
        .summary-card p,
        .strategy-summary-card p,
        .route-metric p,
        .desk-note p {
            margin: 8px 0 0;
            color: var(--ink-soft);
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .kpi-value.good,
        .summary-card.positive strong,
        .strategy-summary-card.positive strong,
        .route-metric.positive strong,
        .console-value.positive,
        .desk-item.positive strong {
            color: var(--teal);
            text-shadow: 0 0 14px rgba(44, 207, 143, 0.14);
        }
        .kpi-value.bad,
        .summary-card.negative strong,
        .strategy-summary-card.negative strong,
        .route-metric.negative strong,
        .console-value.negative,
        .desk-item.negative strong {
            color: var(--rose);
            text-shadow: 0 0 14px rgba(255, 111, 143, 0.14);
        }
        .command-insights,
        .console-grid,
        .desk-grid,
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 16px;
        }
        .insight-card,
        .desk-note,
        .desk-item,
        .note-card,
        .route-tab,
        .coverage-row,
        .matrix-row,
        .visual-row,
        .band-viz {
            padding: 14px 15px;
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.08);
            background: rgba(255, 255, 255, 0.03);
        }
        .insight-label {
            color: var(--ink-faint);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .insight-copy {
            color: var(--ink-strong);
            font-size: 0.95rem;
            line-height: 1.5;
            font-weight: 600;
        }
        .desk-panel {
            display: grid;
            gap: 16px;
        }
        .console-title {
            margin: 0;
            font-size: 1.8rem;
            color: var(--ink-strong);
        }
        .console-time {
            color: var(--ink-strong);
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0;
        }
        .panel-heading h2,
        .section-heading h2 {
            margin: 0;
            color: var(--ink-strong);
            font-size: 1.3rem;
        }
        .section-heading {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 14px;
            margin: 18px 0 10px;
        }
        .panel-heading p,
        .section-heading p {
            margin: 4px 0 0;
            color: var(--ink-soft);
            font-size: 0.92rem;
        }
        .preview-filter-panel {
            margin-bottom: 18px;
        }
        .filter-grid-shell,
        .workspace-shell {
            display: grid;
            gap: 16px;
        }
        .filter-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            align-items: end;
        }
        .workspace-note {
            color: var(--ink-soft);
            font-size: 0.92rem;
        }
        .command-help {
            color: var(--ink-soft);
            font-size: 0.92rem;
            line-height: 1.6;
            margin: 6px 0 0;
        }
        .recent-trade-panel {
            margin-bottom: 18px;
        }
        .recent-trade-shell {
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(280px, 0.92fr);
            gap: 18px;
            align-items: stretch;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.10);
            background:
                linear-gradient(180deg, rgba(10, 12, 16, 0.90), rgba(4, 5, 8, 0.92)),
                repeating-linear-gradient(90deg, transparent 0 30px, rgba(141, 177, 236, 0.02) 30px 31px),
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.06), transparent 42%);
        }
        .recent-trade-shell.positive {
            box-shadow:
                inset 0 0 0 1px rgba(44, 207, 143, 0.08),
                0 0 0 1px rgba(44, 207, 143, 0.04),
                0 18px 40px rgba(7, 28, 22, 0.20);
        }
        .recent-trade-shell.negative {
            box-shadow:
                inset 0 0 0 1px rgba(255, 111, 143, 0.08),
                0 0 0 1px rgba(255, 111, 143, 0.04),
                0 18px 40px rgba(38, 12, 20, 0.20);
        }
        .recent-trade-copy h3 {
            margin: 0;
            color: var(--ink-strong);
            font-size: 1.35rem;
        }
        .recent-trade-status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
        }
        .recent-trade-status,
        .recent-trade-date {
            display: inline-flex;
            align-items: center;
            min-height: 34px;
            padding: 0 12px;
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.12);
            background: rgba(255, 255, 255, 0.04);
            color: var(--ink-soft);
            font-size: 0.76rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .recent-trade-status.positive {
            color: var(--teal);
            border-color: rgba(44, 207, 143, 0.20);
        }
        .recent-trade-status.negative {
            color: var(--rose);
            border-color: rgba(255, 111, 143, 0.20);
        }
        .recent-trade-meta,
        .recent-trade-note {
            margin-top: 10px;
            color: var(--ink-soft);
            line-height: 1.6;
        }
        .recent-trade-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .recent-trade-stat {
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.08);
            background: rgba(255, 255, 255, 0.025);
            padding: 14px;
        }
        .recent-trade-stat span {
            color: var(--ink-faint);
            font-size: 0.76rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }
        .recent-trade-stat strong {
            display: block;
            margin-top: 8px;
            color: var(--ink-strong);
            font-size: 1.05rem;
        }
        .lab-pill-row,
        .run-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }
        .lab-section-card {
            margin-bottom: 18px;
        }
        .lab-card-accent {
            background:
                linear-gradient(135deg, rgba(77, 227, 210, 0.055), rgba(90, 166, 255, 0.025)),
                rgba(3, 3, 5, 0.82);
        }
        .feature-grid,
        .split-grid,
        .compare-grid,
        .mc-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }
        .strategy-rail {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 18px;
        }
        .strategy-tile {
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.08);
            background: rgba(255, 255, 255, 0.025);
            padding: 14px;
            display: grid;
            gap: 10px;
            min-height: 148px;
        }
        .strategy-tile.champion {
            background:
                linear-gradient(135deg, rgba(77, 227, 210, 0.055), rgba(90, 166, 255, 0.025)),
                rgba(3, 3, 5, 0.82);
            border-color: rgba(77, 227, 210, 0.16);
        }
        .strategy-kicker {
            color: var(--ink-faint);
            font-size: 0.72rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }
        .strategy-name {
            color: var(--ink-strong);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }
        .strategy-score {
            color: var(--teal);
            font-size: 1.12rem;
            font-weight: 800;
        }
        .strategy-score.warn { color: var(--gold); }
        .strategy-score.bad { color: var(--rose); }
        .strategy-meta {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }
        .strategy-meta-item {
            padding-top: 8px;
            border-top: 1px solid rgba(141, 177, 236, 0.08);
        }
        .strategy-meta-label {
            color: var(--ink-faint);
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .strategy-meta-value {
            color: var(--ink-strong);
            font-size: 0.88rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .visual-list,
        .band-stack,
        .matrix-board,
        .table-stack {
            display: grid;
            gap: 12px;
        }
        .visual-row-top,
        .band-viz-top {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 12px;
        }
        .visual-row-top strong,
        .band-viz-top strong,
        .route-tab strong,
        .coverage-row strong,
        .matrix-row strong {
            color: var(--ink-strong);
        }
        .visual-row-top span,
        .band-viz-top span,
        .band-labels span,
        .route-tab span,
        .coverage-row span,
        .matrix-row span {
            color: var(--ink-soft);
            font-size: 0.86rem;
        }
        .visual-row p {
            color: var(--ink-faint);
            font-size: 0.84rem;
            margin: 0;
        }
        .visual-track,
        .band-track {
            height: 8px;
            border-radius: 999px;
            background: rgba(141, 177, 236, 0.12);
            overflow: hidden;
            position: relative;
        }
        .visual-fill {
            display: block;
            height: 100%;
            min-width: 4px;
            border-radius: inherit;
            background: var(--cyan);
        }
        .visual-fill.negative { background: var(--rose); }
        .visual-fill.watch { background: var(--gold); }
        .band-range {
            position: absolute;
            inset-block: 0;
            border-radius: inherit;
            background: linear-gradient(90deg, rgba(90, 166, 255, 0.18), rgba(77, 227, 210, 0.7));
        }
        .band-mid {
            position: absolute;
            top: -4px;
            bottom: -4px;
            width: 2px;
            border-radius: 999px;
            background: var(--ink-strong);
            box-shadow: 0 0 16px rgba(238, 244, 255, 0.32);
        }
        .band-labels {
            display: flex;
            justify-content: space-between;
        }
        .mc-band-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 12px;
        }
        .mc-workbench-shell {
            display: grid;
            gap: 14px;
        }
        .mc-toolbar-note {
            color: var(--ink-soft);
            font-size: 0.90rem;
            line-height: 1.55;
            margin: -2px 0 4px;
        }
        .mc-inline-toolbar {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 12px;
            padding: 14px;
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.08);
            background:
                linear-gradient(180deg, rgba(18, 20, 24, 0.88), rgba(8, 10, 14, 0.92)),
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.05), transparent 42%);
        }
        .mc-chart-caption {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-top: 2px;
            color: var(--ink-soft);
            font-size: 0.90rem;
        }
        .mc-chart-caption strong {
            color: var(--ink-strong);
            font-weight: 700;
        }
        .mc-results-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }
        .mc-table-note {
            color: var(--ink-soft);
            font-size: 0.86rem;
            line-height: 1.5;
            margin: 0 0 10px;
        }
        .chart-shell {
            border-radius: 8px;
            border: 1px solid rgba(141, 177, 236, 0.08);
            background:
                repeating-linear-gradient(90deg, transparent 0 34px, rgba(141, 177, 236, 0.025) 34px 35px),
                rgba(255, 255, 255, 0.018);
            padding: 12px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(141, 177, 236, 0.08);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(3, 3, 5, 0.82);
        }
        [data-testid="stPlotlyChart"] > div {
            border-radius: 8px;
        }
        .sidebar-note {
            border: 1px solid rgba(141, 177, 236, 0.10);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.025);
            color: var(--ink-soft);
            padding: 12px;
            margin: 8px 0 14px;
            font-size: 0.86rem;
            line-height: 1.5;
        }
        h1, h2, h3, p, span, label { letter-spacing: 0; }
        [data-testid="stCaptionContainer"] {
            color: var(--ink-soft);
        }
        @media (max-width: 1120px) {
            .topbar,
            .command-header,
            .panel-heading,
            .preview-command,
            .filter-row,
            .feature-grid,
            .split-grid,
            .compare-grid,
            .mc-grid,
            .mc-inline-toolbar,
            .strategy-rail,
            .summary-grid,
            .strategy-card-grid,
            .route-metrics,
            .mc-kpi-grid,
            .mc-results-grid,
            .mc-band-grid,
            .recent-trade-shell,
            .recent-trade-stats,
            .strategy-meta,
            .console-grid,
            .desk-grid,
            .command-insights {
                grid-template-columns: 1fr;
            }
            .topbar,
            .command-header,
            .panel-heading {
                flex-direction: column;
                align-items: flex-start;
            }
            .topbar-note {
                text-align: left;
            }
            .command-badges {
                justify-content: flex-start;
            }
        }
        @media (max-width: 760px) {
            .block-container {
                padding-top: 0.8rem;
            }
            .command-panel {
                padding: 18px;
            }
            .desk-panel,
            .preview-filter-panel,
            .lab-section-card,
            .recent-trade-panel {
                padding: 16px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_pct(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "0.00%"
    return f"{number:.2%}"


def _format_money(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "$0.00"
    return f"${number:,.2f}"


def _format_signed_money(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "$0.00"
    return f"{'+' if number > 0 else ''}${number:,.2f}"


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return "0.00"
    if not math.isfinite(number):
        return "Uncapped"
    return f"{number:,.2f}"


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _html(value: object) -> str:
    return escape(str(value), quote=True)


def _display_name(value: object) -> str:
    text = str(value or "").replace("::", " / ").replace("__", " / ").replace("_", " ")
    return " ".join(text.split())


def _format_timestamp(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Current run"
    return _html(text.replace("T", " ")[:19])


def _format_signed_pct(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "0.00%"
    return f"{number:+.2%}"


def _tone(value: float, positive_good: bool = True) -> str:
    if not math.isfinite(value):
        return "warn"
    if abs(value) < 1e-12:
        return "warn"
    good = value > 0 if positive_good else value < 0
    return "good" if good else "bad"


def _metric_html(label: str, value: str, note: str = "", tone: str = "good") -> str:
    return (
        f"<div class='kpi-card {tone}'>"
        f"<div class='kpi-label'>{_html(label)}</div>"
        f"<div class='kpi-value {tone}'>{_html(value)}</div>"
        f"<div class='kpi-note'>{_html(note)}</div>"
        "</div>"
    )


def _console_card(label: str, value: str, note: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return (
        "<div class='console-card'>"
        f"<div class='console-label'>{_html(label)}</div>"
        f"<div class='console-value{tone_class}'>{_html(value)}</div>"
        f"<div class='console-note'>{_html(note)}</div>"
        "</div>"
    )


def _insight_html(label: str, copy: str, tone: str = "good") -> str:
    return (
        f"<div class='insight-card {tone}'>"
        f"<div class='insight-label'>{_html(label)}</div>"
        f"<div class='insight-copy'>{_html(copy)}</div>"
        "</div>"
    )


def _route_metric_html(label: str, value: str, detail: str, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return (
        f"<article class='route-metric{tone_class}'>"
        f"<span>{_html(label)}</span>"
        f"<strong>{_html(value)}</strong>"
        f"<p>{_html(detail)}</p>"
        "</article>"
    )


def _band_viz_html(label: str, p05: object, p50: object, p95: object, formatter) -> str:
    p05_num = _safe_float(p05)
    p50_num = _safe_float(p50)
    p95_num = _safe_float(p95)
    if p05_num is None or p50_num is None or p95_num is None:
        return f"<div class='route-tab'><span>{_html(label)} unavailable for this scope.</span></div>"
    low = min(p05_num, p50_num, p95_num, 0.0)
    high = max(p05_num, p50_num, p95_num, 1.0)
    span = high - low if abs(high - low) > 1e-12 else 1.0
    start = ((p05_num - low) / span) * 100.0
    end = ((p95_num - low) / span) * 100.0
    middle = ((p50_num - low) / span) * 100.0
    return (
        "<div class='band-viz'>"
        f"<div class='band-viz-top'><strong>{_html(label)}</strong><span>{_html(formatter(p50_num))} median</span></div>"
        "<div class='band-track'>"
        f"<span class='band-range' style='left:{start:.2f}%; width:{max(end - start, 3.0):.2f}%'></span>"
        f"<span class='band-mid' style='left:{middle:.2f}%'></span>"
        "</div>"
        f"<div class='band-labels'><span>{_html(formatter(p05_num))}</span><span>{_html(formatter(p95_num))}</span></div>"
        "</div>"
    )


def _recent_trade_spotlight_html(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<div class='route-tab'><span>No trades match the current scope.</span></div>"

    frame = trades.copy()
    if "exit_date" in frame.columns:
        frame = frame.sort_values("exit_date")
    elif "entry_date" in frame.columns:
        frame = frame.sort_values("entry_date")
    recent = frame.iloc[-1]
    pnl = _safe_float(recent.get("pnl", 0.0)) or 0.0
    trade_return = _safe_float(recent.get("return", 0.0)) or 0.0
    tone = "positive" if pnl >= 0 else "negative"
    title = f"{_display_name(recent.get('strategy', 'Strategy'))} {str(recent.get('side', '')).title()}".strip()
    status = "Winning trade" if pnl >= 0 else "Losing trade"
    exit_date = recent.get("exit_date") if pd.notna(recent.get("exit_date")) else recent.get("entry_date")
    date_label = pd.Timestamp(exit_date).strftime("%b %d, %Y %I:%M %p") if pd.notna(exit_date) else "Date unavailable"
    stats = [
        ("P&L", _format_signed_money(pnl)),
        ("Return", _format_signed_pct(trade_return)),
        ("Bars held", str(int(_safe_float(recent.get("bars_held", 0.0)) or 0))),
        ("Exit", str(recent.get("exit_reason", "n/a")).replace("_", " ").title()),
    ]
    stat_html = "".join(
        f"<div class='recent-trade-stat'><span>{_html(label)}</span><strong>{_html(value)}</strong></div>"
        for label, value in stats
    )
    meta = []
    if pd.notna(recent.get("entry_date")):
        meta.append(f"Entry: {pd.Timestamp(recent.get('entry_date')).strftime('%b %d %I:%M %p')}")
    if "entry_price" in recent and _safe_float(recent.get("entry_price")) is not None:
        meta.append(f"Entry px: {_format_number(recent.get('entry_price'))}")
    if "exit_price" in recent and _safe_float(recent.get("exit_price")) is not None:
        meta.append(f"Exit px: {_format_number(recent.get('exit_price'))}")
    return (
        f"<div class='recent-trade-shell {tone}'>"
        "<div class='recent-trade-copy'>"
        f"<div class='recent-trade-status-row'><span class='recent-trade-status {tone}'>{_html(status)}</span><span class='recent-trade-date'>{_html(date_label)}</span></div>"
        f"<h3>{_html(title)}</h3>"
        f"<div class='recent-trade-meta'>{_html(' | '.join(meta) if meta else 'No additional trade metadata available.')}</div>"
        f"<div class='recent-trade-note'>{_html('Latest execution in the current scope, useful as a quick tape check before drilling deeper into the comparison views.')}</div>"
        "</div>"
        f"<div class='recent-trade-stats'>{stat_html}</div>"
        "</div>"
    )


def _sparkline_svg(equity: pd.DataFrame, strategy: str) -> str:
    if equity.empty or not strategy or "strategy" not in equity.columns or "equity" not in equity.columns:
        return "<div class='empty-sparkline'>No equity curve loaded for this run.</div>"

    frame = equity[equity["strategy"] == strategy].copy()
    if frame.empty:
        return "<div class='empty-sparkline'>Select a strategy to preview its equity curve.</div>"
    if "date" in frame.columns:
        frame = frame.sort_values("date")

    values = pd.to_numeric(frame["equity"], errors="coerce").dropna()
    if values.empty:
        return "<div class='empty-sparkline'>No numeric equity values found.</div>"
    if len(values) > 120:
        values = values.iloc[:: max(1, len(values) // 120)]

    width = 640
    height = 112
    x_min = 12
    x_span = width - 24
    y_min = 14
    y_span = height - 28
    low = float(values.min())
    high = float(values.max())
    denom = high - low if abs(high - low) > 1e-12 else 1.0
    count = len(values)
    points: list[str] = []
    for idx, value in enumerate(values):
        x = x_min + (idx / max(1, count - 1)) * x_span
        y = y_min + (1.0 - ((float(value) - low) / denom)) * y_span
        points.append(f"{x:.1f},{y:.1f}")

    baseline_y = y_min + (1.0 - ((float(values.iloc[0]) - low) / denom)) * y_span
    end_x, end_y = points[-1].split(",")
    fill_points = f"{x_min},{height - 12} {' '.join(points)} {width - 12},{height - 12}"
    grid_lines = "".join(f"<line class='sparkline-grid' x1='12' x2='{width - 12}' y1='{y}' y2='{y}' />" for y in (28, 56, 84))
    return (
        f"<svg class='mini-sparkline' viewBox='0 0 {width} {height}' role='img' aria-label='{_html(strategy)} equity sparkline'>"
        f"{grid_lines}"
        f"<line class='sparkline-baseline' x1='12' x2='{width - 12}' y1='{baseline_y:.1f}' y2='{baseline_y:.1f}' />"
        f"<polygon class='sparkline-fill' points='{fill_points}' />"
        f"<polyline class='sparkline-line' points='{' '.join(points)}' />"
        f"<circle class='sparkline-end' cx='{end_x}' cy='{end_y}' r='5' />"
        "</svg>"
    )


def _style_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#070a0b",
        font={"color": "#d7dedb", "family": "Inter, Arial, sans-serif"},
        title={"font": {"color": "#eef3ef", "size": 18}},
        colorway=CHART_COLORS,
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.08)",
            "font": {"color": "#9aa5ae"},
        },
        margin={"l": 20, "r": 20, "t": 48, "b": 28},
        hoverlabel={"bgcolor": "#111315", "bordercolor": "#254541", "font": {"color": "#eef3ef"}},
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)", color="#9aa5ae")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)", color="#9aa5ae")
    return fig


def _augment_metrics(metrics: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    required = {"num_trades", "profit_factor", "expected_r", "win_rate"}
    if required.issubset(metrics.columns):
        return metrics.sort_values("sharpe", ascending=False).reset_index(drop=True)
    if trades.empty or "strategy" not in trades.columns:
        augmented = metrics.copy()
        for column in sorted(required - set(augmented.columns)):
            augmented[column] = 0.0
        return augmented.sort_values("sharpe", ascending=False).reset_index(drop=True)

    rows = []
    for strategy, group in trades.groupby("strategy"):
        pnl = group["pnl"].astype(float).fillna(0.0)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(losses.abs().sum()) if len(losses) else 0.0
        risk_unit = float(losses.abs().mean()) if len(losses) else 0.0
        rows.append(
            {
                "strategy": strategy,
                "num_trades": int(len(group)),
                "win_rate": float((pnl > 0.0).mean()) if len(group) else 0.0,
                "profit_factor": (gross_profit / gross_loss) if gross_loss > 1e-12 else (float("inf") if gross_profit > 0 else 0.0),
                "expected_r": float((pnl / risk_unit).mean()) if risk_unit > 1e-12 else 0.0,
            }
        )
    extra = pd.DataFrame(rows)
    augmented = metrics.merge(extra, on="strategy", how="left", suffixes=("", "_trade"))
    for column in required:
        trade_col = f"{column}_trade"
        if trade_col in augmented.columns:
            augmented[column] = augmented[column] if column in metrics.columns else augmented[trade_col]
            augmented[column] = augmented[column].fillna(augmented[trade_col])
            augmented = augmented.drop(columns=[trade_col])
    return augmented.sort_values("sharpe", ascending=False).reset_index(drop=True)


def _sort_metric_frame(frame: pd.DataFrame, metric: str, leading_cols: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_cols: list[str] = []
    ascending: list[bool] = []
    for column in leading_cols or []:
        if column in frame.columns and column not in sort_cols:
            sort_cols.append(column)
            ascending.append(True)
    for column in [metric, "expectancy", "sharpe", "bars"]:
        if column in frame.columns and column not in sort_cols:
            sort_cols.append(column)
            ascending.append(False)
    if not sort_cols:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _mark_manual_override(flag_key: str) -> None:
    st.session_state[flag_key] = True


def _strategy_preset_selection(ranked_strategies: list[str], mode: str) -> list[str]:
    if not ranked_strategies:
        return []
    if mode == "top_3":
        return ranked_strategies[:3]
    if mode == "all":
        return ranked_strategies
    if mode in {"top_5", "champion_plus_challengers"}:
        return ranked_strategies[:5]
    return ranked_strategies[:5]


def _apply_strategy_preset(
    selection_key: str,
    manual_key: str,
    preset_key: str,
    focus_keys: list[str],
    ranked_strategies: list[str],
    mode: str,
) -> None:
    selection = _strategy_preset_selection(ranked_strategies, mode)
    st.session_state[selection_key] = selection
    st.session_state[manual_key] = False
    st.session_state[preset_key] = mode
    if selection:
        for key in focus_keys:
            if mode == "champion_plus_challengers" or st.session_state.get(key) not in selection:
                st.session_state[key] = selection[0]


def _reset_filter_state(
    date_key: str,
    date_default: tuple[object, object],
    regime_key: str,
    regime_default: list[str],
    ranking_key: str,
    ranking_default: str,
    mc_model_key: str,
    mc_model_default: object,
    mc_regime_key: str,
    mc_regime_default: object,
    selection_key: str,
    manual_key: str,
    preset_key: str,
    focus_keys: list[str],
) -> None:
    st.session_state[date_key] = date_default
    st.session_state[regime_key] = regime_default
    st.session_state[ranking_key] = ranking_default
    st.session_state[mc_model_key] = mc_model_default
    st.session_state[mc_regime_key] = mc_regime_default
    st.session_state[selection_key] = []
    st.session_state[manual_key] = False
    st.session_state[preset_key] = "top_5"
    for key in focus_keys:
        st.session_state[key] = None


def _best_regime_lookup(per_regime: pd.DataFrame, ranking_metric: str) -> dict[str, tuple[str, float]]:
    if per_regime.empty or "strategy" not in per_regime.columns or "regime_name" not in per_regime.columns:
        return {}
    ranking = _sort_metric_frame(per_regime, ranking_metric)
    best: dict[str, tuple[str, float]] = {}
    for strategy, group in ranking.groupby("strategy", sort=False):
        row = group.iloc[0]
        best[str(strategy)] = (str(row.get("regime_name", "")), _safe_float(row.get(ranking_metric, 0.0)) or 0.0)
    return best


def _strategy_tile_html(
    row: pd.Series,
    ranking_label: str,
    ranking_metric: str,
    best_regime_lookup: dict[str, tuple[str, float]],
    champion: bool = False,
) -> str:
    strategy = str(row.get("strategy", ""))
    best_regime, best_regime_score = best_regime_lookup.get(strategy, ("", 0.0))
    score = _safe_float(row.get(ranking_metric, 0.0)) or 0.0
    score_tone = _tone(score)
    drawdown = _safe_float(row.get("max_drawdown", 0.0)) or 0.0
    return (
        f"<div class='strategy-tile{' champion' if champion else ''}'>"
        f"<div class='strategy-kicker'>{'Champion' if champion else 'In view'}</div>"
        f"<div class='strategy-name'>{_html(_display_name(strategy))}</div>"
        f"<div class='strategy-score {score_tone}'>{_html(ranking_label)} {_html(_format_number(score))}</div>"
        "<div class='strategy-meta'>"
        f"<div class='strategy-meta-item'><div class='strategy-meta-label'>Total Return</div><div class='strategy-meta-value'>{_html(_format_pct(row.get('total_return', 0.0)))}</div></div>"
        f"<div class='strategy-meta-item'><div class='strategy-meta-label'>Win Rate</div><div class='strategy-meta-value'>{_html(_format_pct(row.get('win_rate', 0.0)))}</div></div>"
        f"<div class='strategy-meta-item'><div class='strategy-meta-label'>Max Drawdown</div><div class='strategy-meta-value'>{_html(_format_pct(drawdown))}</div></div>"
        f"<div class='strategy-meta-item'><div class='strategy-meta-label'>Best Regime</div><div class='strategy-meta-value'>{_html(best_regime or 'No regime data')}</div></div>"
        "</div>"
        "</div>"
    )


def _render_strategy_rail(
    metrics: pd.DataFrame,
    ranking_label: str,
    ranking_metric: str,
    best_regime_lookup: dict[str, tuple[str, float]],
) -> None:
    if metrics.empty:
        return
    cards = []
    for idx, (_, row) in enumerate(metrics.iterrows()):
        cards.append(_strategy_tile_html(row, ranking_label, ranking_metric, best_regime_lookup, champion=idx == 0))
    st.markdown(f"<div class='strategy-rail'>{''.join(cards)}</div>", unsafe_allow_html=True)


def _mc_band_frame(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty or "step" not in paths.columns or "equity" not in paths.columns:
        return pd.DataFrame()
    frame = paths[["step", "equity"]].copy()
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame = frame.dropna(subset=["equity"])
    if frame.empty:
        return pd.DataFrame()
    quantiles = (
        frame.groupby("step")["equity"]
        .quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        .unstack()
        .rename(columns={0.05: "p05", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.95: "p95"})
        .reset_index()
    )
    mean = frame.groupby("step", as_index=False)["equity"].mean().rename(columns={"equity": "mean"})
    return quantiles.merge(mean, on="step", how="left").sort_values("step").reset_index(drop=True)


def _mc_local_focus_options(
    summary: pd.DataFrame,
    selected_strategies: list[str],
    selected_regimes: list[str],
) -> dict[str, list[str] | str | None]:
    if summary.empty:
        return {
            "strategies": [],
            "regimes": [],
            "models": [],
            "default_strategy": None,
            "default_regime": None,
            "default_model": None,
        }

    frame = summary.copy()
    available_strategies = sorted(frame["strategy"].dropna().astype(str).unique().tolist()) if "strategy" in frame.columns else []
    available_regimes = sorted(frame["regime_filter"].dropna().astype(str).unique().tolist()) if "regime_filter" in frame.columns else []
    available_models = sorted(frame["stress_model"].dropna().astype(str).unique().tolist()) if "stress_model" in frame.columns else []

    strategy_options = [item for item in selected_strategies if item in available_strategies] or available_strategies
    if selected_regimes:
        regime_options = [item for item in available_regimes if item in selected_regimes]
    else:
        regime_options = available_regimes
    if not regime_options:
        regime_options = available_regimes

    default_strategy = strategy_options[0] if strategy_options else None
    if selected_regimes:
        default_regime = next((item for item in selected_regimes if item in regime_options), regime_options[0] if regime_options else None)
    else:
        default_regime = "all" if "all" in regime_options else (regime_options[0] if regime_options else None)
    default_model = available_models[0] if available_models else None

    return {
        "strategies": strategy_options,
        "regimes": regime_options,
        "models": available_models,
        "default_strategy": default_strategy,
        "default_regime": default_regime,
        "default_model": default_model,
    }


def _reset_monte_carlo_view_state(
    strategy_key: str,
    strategy_value: str | None,
    regime_key: str,
    regime_value: str | None,
    model_key: str,
    model_value: str | None,
    density_key: str,
    distribution_key: str,
) -> None:
    st.session_state[strategy_key] = strategy_value
    st.session_state[regime_key] = regime_value
    st.session_state[model_key] = model_value
    st.session_state[density_key] = "Balanced"
    st.session_state[distribution_key] = "terminal return"


def _style_mc_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig = _style_figure(fig, height)
    fig.update_layout(
        title={"text": ""},
        showlegend=False,
        margin={"l": 18, "r": 18, "t": 12, "b": 24},
        plot_bgcolor="#121316",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="", gridcolor="rgba(141, 177, 236, 0.05)", zeroline=False)
    fig.update_yaxes(
        title_text="",
        gridcolor="rgba(141, 177, 236, 0.14)",
        zerolinecolor="rgba(238, 244, 255, 0.16)",
        tickprefix="$",
        separatethousands=True,
    )
    return fig


def _monte_carlo_overlay_figure(paths: pd.DataFrame, max_paths: int = 48) -> go.Figure:
    if paths.empty or "step" not in paths.columns or "equity" not in paths.columns:
        return go.Figure()

    frame = paths.copy()
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame = frame.dropna(subset=["equity"])
    if frame.empty:
        return go.Figure()

    fig = go.Figure()
    palette = [
        "#46c2ff",
        "#8b5cf6",
        "#f3b34d",
        "#38d39f",
        "#ff6f8f",
        "#2dd4bf",
        "#f97316",
        "#22c55e",
        "#a78bfa",
        "#facc15",
        "#60a5fa",
        "#fb7185",
    ]
    simulation_ids = frame["simulation"].drop_duplicates().tolist()[:max_paths]
    raw = frame[frame["simulation"].isin(simulation_ids)].copy()
    for idx, (_, group) in enumerate(raw.groupby("simulation", sort=False)):
        ordered = group.sort_values("step")
        fig.add_trace(
            go.Scatter(
                x=ordered["step"],
                y=ordered["equity"],
                mode="lines",
                line={"color": palette[idx % len(palette)], "width": 1.45},
                opacity=0.62,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    avg = frame.groupby("step", as_index=False)["equity"].mean().sort_values("step")
    fig.add_trace(
        go.Scatter(
            x=avg["step"],
            y=avg["equity"],
            mode="lines",
            name="Average path",
            line={"color": "#eef4ff", "width": 3.8},
            hovertemplate="Step %{x}<br>Average %{y:.0f}<extra></extra>",
        )
    )

    median = frame.groupby("step", as_index=False)["equity"].median().sort_values("step")
    fig.add_trace(
        go.Scatter(
            x=median["step"],
            y=median["equity"],
            mode="lines",
            name="Median path",
            line={"color": "#8fd9ff", "width": 2.2, "dash": "dash"},
            hovertemplate="Step %{x}<br>Median %{y:.0f}<extra></extra>",
        )
    )

    fig.update_layout(xaxis_title="Step", yaxis_title="Equity")
    return _style_mc_figure(fig, 500)


def _monte_carlo_band_figure(paths: pd.DataFrame, display_mode: str) -> go.Figure:
    if paths.empty:
        return go.Figure()
    fig = go.Figure()
    band_frame = _mc_band_frame(paths)
    if display_mode in {"Percentile bands", "Both"} and not band_frame.empty:
        fig.add_trace(
            go.Scatter(
                x=band_frame["step"],
                y=band_frame["p95"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_frame["step"],
                y=band_frame["p05"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(142, 202, 230, 0.10)",
                name="P05-P95",
                hovertemplate="Step %{x}<br>P05 %{y:.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_frame["step"],
                y=band_frame["p75"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_frame["step"],
                y=band_frame["p25"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(110, 231, 168, 0.18)",
                name="P25-P75",
                hovertemplate="Step %{x}<br>P25 %{y:.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_frame["step"],
                y=band_frame["p50"],
                mode="lines",
                name="Median",
                line={"color": "#8ecae6", "width": 2.8},
                hovertemplate="Step %{x}<br>Median %{y:.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_frame["step"],
                y=band_frame["mean"],
                mode="lines",
                name="Mean",
                line={"color": "#f2c94c", "width": 1.8, "dash": "dash"},
                hovertemplate="Step %{x}<br>Mean %{y:.0f}<extra></extra>",
            )
        )

    if display_mode in {"Raw paths", "Both"}:
        raw = paths.copy()
        raw["path_id"] = raw["simulation"].astype(str)
        path_ids = raw["path_id"].drop_duplicates().tolist()[:40]
        raw = raw[raw["path_id"].isin(path_ids)]
        for path_id, group in raw.groupby("path_id", sort=False):
            ordered = group.sort_values("step")
            fig.add_trace(
                go.Scatter(
                    x=ordered["step"],
                    y=ordered["equity"],
                    mode="lines",
                    line={"color": "rgba(244, 123, 157, 0.22)", "width": 1.1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(title="Monte Carlo Path Explorer", xaxis_title="Step", yaxis_title="Equity")
    return _style_figure(fig, 440)


def _mc_distribution_figure(terminal: pd.DataFrame, metric_key: str) -> go.Figure:
    if terminal.empty:
        return go.Figure()
    metric_map = {
        "terminal return": ("terminal_return", "Terminal Return Distribution", ".1%"),
        "terminal equity": ("terminal_equity", "Terminal Equity Distribution", ",.0f"),
        "max drawdown": ("max_drawdown", "Max Drawdown Distribution", ".1%"),
    }
    column, title, tick_format = metric_map[metric_key]
    frame = terminal.copy()
    if column == "terminal_equity" and "terminal_equity" not in frame.columns:
        frame["terminal_equity"] = frame["terminal_return"] + 1.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return go.Figure()
    fig = px.histogram(frame, x=column, nbins=40, title=title, height=420, color_discrete_sequence=["#8ecae6"])
    p5 = float(values.quantile(0.05))
    p50 = float(values.quantile(0.50))
    p95 = float(values.quantile(0.95))
    for value, label, color in ((p5, "P05", "#f47b9d"), (p50, "P50", "#f2c94c"), (p95, "P95", "#6ee7a8")):
        fig.add_vline(x=value, line_width=2, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top")
    fig.update_layout(bargap=0.05)
    fig.update_xaxes(tickformat=tick_format)
    return _style_figure(fig, 420)


def _mc_focus_summary(
    summary: pd.DataFrame,
    focus_strategy: str,
    focus_regime: str,
    focus_model: str,
) -> pd.Series | None:
    if summary.empty:
        return None
    frame = summary.copy()
    if "strategy" in frame.columns:
        frame = frame[frame["strategy"].astype(str) == str(focus_strategy)]
    if "regime_filter" in frame.columns:
        frame = frame[frame["regime_filter"].astype(str) == str(focus_regime)]
    if "stress_model" in frame.columns:
        frame = frame[frame["stress_model"].astype(str) == str(focus_model)]
    if frame.empty:
        return None
    return frame.iloc[0]


def _mc_results_cards_html(
    terminal: pd.DataFrame,
    focus_summary: pd.Series | None,
    initial_capital: float,
) -> str:
    if terminal.empty and focus_summary is None:
        return ""

    terminal_equity = pd.to_numeric(terminal.get("terminal_equity"), errors="coerce").dropna() if not terminal.empty else pd.Series(dtype=float)
    average_equity = float(terminal_equity.mean()) if not terminal_equity.empty else initial_capital
    median_equity = float(terminal_equity.median()) if not terminal_equity.empty else initial_capital
    min_equity = float(terminal_equity.min()) if not terminal_equity.empty else initial_capital
    max_equity = float(terminal_equity.max()) if not terminal_equity.empty else initial_capital

    def balance_tone(value: float) -> str:
        if value >= initial_capital:
            return "positive"
        return "negative"

    def risk_tone(value: float, warning: float, danger: float) -> str:
        if value >= danger:
            return "negative"
        if value <= warning:
            return "positive"
        return ""

    loss_probability = _safe_float(focus_summary.get("probability_of_loss", 0.0)) if focus_summary is not None else 0.0
    drawdown_p50 = _safe_float(focus_summary.get("max_drawdown_p50", 0.0)) if focus_summary is not None else None
    drawdown_p95 = _safe_float(focus_summary.get("max_drawdown_p95", 0.0)) if focus_summary is not None else None
    risk_of_ruin = _safe_float(focus_summary.get("risk_of_ruin", 0.0)) if focus_summary is not None else 0.0

    cards = [
        _route_metric_html("Average ending balance", _format_money(average_equity), "Average across the saved simulation paths in view.", balance_tone(average_equity)),
        _route_metric_html("Median ending balance", _format_money(median_equity), "Median terminal equity from the focused scenario.", balance_tone(median_equity)),
        _route_metric_html("Min ending balance", _format_money(min_equity), "Worst simulated finish from the saved path set.", balance_tone(min_equity)),
        _route_metric_html("Max ending balance", _format_money(max_equity), "Best simulated finish from the saved path set.", balance_tone(max_equity)),
        _route_metric_html("Probability of loss", _format_pct(loss_probability or 0.0), "Share of paths that finish below starting capital.", risk_tone(loss_probability or 0.0, 0.35, 0.55)),
        _route_metric_html("Max drawdown P50", _format_pct(drawdown_p50 or 0.0), "Median path-level maximum drawdown.", "positive" if (drawdown_p50 or 0.0) > -0.05 else "negative"),
        _route_metric_html("Max drawdown P95", _format_pct(drawdown_p95 or 0.0), "95th percentile path-level maximum drawdown.", "positive" if (drawdown_p95 or 0.0) > -0.10 else "negative"),
        _route_metric_html("Risk of ruin", _format_pct(risk_of_ruin or 0.0), "Configured ruin proxy for the focused scenario.", risk_tone(risk_of_ruin or 0.0, 0.05, 0.12)),
    ]
    return f"<div class='mc-results-grid'>{''.join(cards)}</div>"


def _mc_comparison_figure(summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    if summary.empty or metric not in summary.columns:
        return go.Figure()
    frame = summary.copy()
    frame["scenario"] = frame["stress_model"].astype(str) + " / " + frame["regime_filter"].astype(str)
    fig = px.bar(
        frame,
        x="strategy",
        y=metric,
        color="scenario",
        barmode="group",
        title=title,
        height=430,
    )
    if "drawdown" in metric:
        fig.update_yaxes(tickformat=".1%")
    elif "prob" in metric or "risk" in metric:
        fig.update_yaxes(tickformat=".0%")
    elif "terminal" in metric:
        fig.update_yaxes(tickformat=".1%")
    return _style_figure(fig, 430)


def _filter_frame_by_date(frame: pd.DataFrame, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None, column: str = "date") -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame.copy()
    out = frame.copy()
    if start_date is not None:
        out = out[out[column] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out[column] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    return out


def _filter_by_date_and_regime(
    price_regimes: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    selected_regimes: list[str],
) -> pd.DataFrame:
    frame = price_regimes.copy()
    if frame.empty:
        return frame
    if start_date is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if selected_regimes and "regime_name" in frame.columns:
        frame = frame[frame["regime_name"].isin(selected_regimes)]
    return frame


def _filter_returns(
    returns: pd.DataFrame,
    filtered_price_regimes: pd.DataFrame,
    selected_strategies: list[str],
) -> pd.DataFrame:
    if returns.empty:
        return returns.copy()
    frame = returns.copy()
    if not filtered_price_regimes.empty and "date" in filtered_price_regimes.columns and "date" in frame.columns:
        allowed_dates = filtered_price_regimes[["date"]].drop_duplicates()
        frame = allowed_dates.merge(frame, on="date", how="inner")
    columns = ["date"] + [strategy for strategy in selected_strategies if strategy in frame.columns]
    if len(columns) == 1:
        return pd.DataFrame(columns=["date"])
    return frame[columns].sort_values("date").reset_index(drop=True)


def _equity_from_returns(returns: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if returns.empty or "date" not in returns.columns:
        return pd.DataFrame()
    pieces: list[pd.DataFrame] = []
    for strategy in [column for column in returns.columns if column != "date"]:
        series = pd.to_numeric(returns[strategy], errors="coerce").fillna(0.0)
        equity = initial_capital * (1.0 + series).cumprod()
        pieces.append(pd.DataFrame({"date": returns["date"], "strategy": strategy, "equity": equity, "return": series}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _filter_trades(
    trades: pd.DataFrame,
    filtered_price_regimes: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    selected_strategies: list[str],
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    frame = trades.copy()
    if selected_strategies and "strategy" in frame.columns:
        frame = frame[frame["strategy"].isin(selected_strategies)]
    if "entry_date" in frame.columns:
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    if "exit_date" in frame.columns:
        frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    if start_date is not None and "exit_date" in frame.columns:
        frame = frame[frame["exit_date"] >= pd.Timestamp(start_date)]
    if end_date is not None and "entry_date" in frame.columns:
        frame = frame[frame["entry_date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if not filtered_price_regimes.empty and "entry_date" in frame.columns:
        allowed_dates = set(pd.to_datetime(filtered_price_regimes["date"], errors="coerce").dropna().tolist())
        if allowed_dates:
            frame = frame[frame["entry_date"].isin(allowed_dates) | frame["exit_date"].isin(allowed_dates)]
    return frame.reset_index(drop=True)


def _compute_strategy_metrics(returns: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if returns.empty or "date" not in returns.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for strategy in [column for column in returns.columns if column != "date"]:
        series = pd.to_numeric(returns[strategy], errors="coerce").fillna(0.0)
        equity = initial_capital * (1.0 + series).cumprod()
        metrics = _metrics_from_returns(series, equity, DEFAULT_BARS_PER_YEAR)
        strategy_trades = trades[trades["strategy"].astype(str) == strategy] if not trades.empty and "strategy" in trades.columns else pd.DataFrame()
        pnl = pd.to_numeric(strategy_trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(losses.abs().sum()) if len(losses) else 0.0
        risk_unit = float(losses.abs().mean()) if len(losses) else 0.0
        metrics.update(
            {
                "strategy": strategy,
                "num_trades": int(len(strategy_trades)),
                "profit_factor": (gross_profit / gross_loss) if gross_loss > 1e-12 else (float("inf") if gross_profit > 0 else 0.0),
                "expected_r": float((pnl / risk_unit).mean()) if risk_unit > 1e-12 else 0.0,
                "win_rate": float((pnl > 0.0).mean()) if len(strategy_trades) else metrics.get("bar_win_rate", 0.0),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def _compute_per_regime_metrics(returns: pd.DataFrame, price_regimes: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if returns.empty or price_regimes.empty or "date" not in returns.columns or "date" not in price_regimes.columns:
        return pd.DataFrame()
    regime_lookup = price_regimes[["date", "regime_id", "regime_name"]].drop_duplicates(subset=["date"])
    frame = returns.merge(regime_lookup, on="date", how="inner").dropna(subset=["regime_id"])
    rows: list[dict[str, object]] = []
    for strategy in [column for column in returns.columns if column != "date"]:
        strategy_frame = frame[["date", "regime_id", "regime_name", strategy]].copy()
        strategy_frame["return"] = pd.to_numeric(strategy_frame[strategy], errors="coerce").fillna(0.0)
        for (regime_id, regime_name), group in strategy_frame.groupby(["regime_id", "regime_name"]):
            local_equity = initial_capital * (1.0 + group["return"]).cumprod()
            metrics = _metrics_from_returns(group["return"], local_equity, DEFAULT_BARS_PER_YEAR)
            metrics["strategy"] = strategy
            metrics["regime_id"] = int(regime_id)
            metrics["regime_name"] = str(regime_name)
            metrics["win_rate"] = metrics.get("bar_win_rate", 0.0)
            rows.append(metrics)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def _build_regime_ranking(per_regime: pd.DataFrame, ranking_metric: str) -> pd.DataFrame:
    if per_regime.empty:
        return pd.DataFrame()
    ranking = (
        per_regime.groupby(["strategy", "regime_id", "regime_name"], as_index=False)
        .agg(
            bars=("bars", "sum"),
            total_return=("total_return", "mean"),
            expectancy=("expectancy", "mean"),
            sharpe=("sharpe", "mean"),
            max_drawdown=("max_drawdown", "mean"),
            win_rate=("win_rate", "mean"),
        )
    )
    return _sort_metric_frame(ranking, ranking_metric, leading_cols=["regime_id"])


def _build_model_leaderboard(model_variants: list[dict[str, object]], oos_ranking: pd.DataFrame) -> pd.DataFrame:
    variants_frame = pd.DataFrame(model_variants)
    if not oos_ranking.empty:
        frame = oos_ranking.copy()
        frame["model_id"] = frame["candidate"].astype(str) + "::" + frame["model_family"].astype(str)
        display_lookup = variants_frame.set_index("model_id")["display_label"].to_dict() if not variants_frame.empty else {}
        frame["display_label"] = frame["model_id"].map(display_lookup).fillna(frame["model_id"].map(_display_name))
        best_lookup = variants_frame.set_index("model_id")["is_best"].to_dict() if not variants_frame.empty else {}
        frame["recommended"] = frame["model_id"].map(best_lookup).fillna(False)
        frame["label"] = frame.apply(
            lambda row: f"★ Recommended - {row['display_label']}" if bool(row.get("recommended", False)) else str(row["display_label"]),
            axis=1,
        )
        return frame.sort_values(["status", "oos_score_mean"], ascending=[True, False]).reset_index(drop=True)

    if variants_frame.empty:
        return pd.DataFrame()
    variants_frame = variants_frame.copy()
    variants_frame["label"] = variants_frame.apply(
        lambda row: f"★ Recommended - {row['display_label']}" if bool(row.get("is_best", False)) else str(row["display_label"]),
        axis=1,
    )
    sort_cols = [column for column in ["rank", "rank_value"] if column in variants_frame.columns]
    if not sort_cols:
        return variants_frame.reset_index(drop=True)
    ascending = [True if column == "rank" else False for column in sort_cols]
    return variants_frame.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _price_regime_figure(price_regimes: pd.DataFrame) -> go.Figure:
    frame = _downsample(price_regimes.dropna(subset=["date", "close"]), max_points=8000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["close"], mode="lines", name="Close", line={"color": "#8ecae6", "width": 1.6}))
    if "regime_name" in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["close"],
                mode="markers",
                marker={"color": frame["regime_id"], "colorscale": "Tealrose", "size": 4, "opacity": 0.55},
                name="Regime",
                text=frame["regime_name"],
                hovertemplate="%{x}<br>Close=%{y:.2f}<br>%{text}<extra></extra>",
            )
        )
    fig.update_layout(title="Regime Timeline Explorer", xaxis_title="", yaxis_title="Price")
    return _style_figure(fig, 470)


def _equity_figure(equity: pd.DataFrame, strategies: list[str]) -> go.Figure:
    frame = equity[equity["strategy"].isin(strategies)].copy()
    frame = _downsample(frame, max_points=12000)
    fig = px.line(frame, x="date", y="equity", color="strategy", title="Strategy Equity Curves", height=430)
    fig.update_traces(line={"width": 2})
    return _style_figure(fig, 430)


def _drawdown_figure(equity: pd.DataFrame, strategies: list[str]) -> go.Figure:
    frame = equity[equity["strategy"].isin(strategies)].copy()
    if frame.empty:
        return go.Figure()
    pieces = []
    for strategy, group in frame.groupby("strategy"):
        ordered = group.sort_values("date").copy()
        ordered["drawdown"] = ordered["equity"] / ordered["equity"].cummax().replace(0.0, pd.NA) - 1.0
        pieces.append(ordered)
    out = _downsample(pd.concat(pieces, ignore_index=True), max_points=12000)
    fig = px.line(out, x="date", y="drawdown", color="strategy", title="Strategy Drawdowns", height=390)
    fig.update_traces(line={"width": 2})
    return _style_figure(fig, 390)


def _signal_figure(signals: pd.DataFrame, price_regimes: pd.DataFrame, strategy: str) -> go.Figure:
    if signals.empty or strategy not in signals.columns or price_regimes.empty:
        return go.Figure()
    frame = price_regimes[["date", "close"]].merge(signals[["date", strategy]], on="date", how="inner")
    frame = _downsample(frame, max_points=6000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["close"], mode="lines", name="Close", line={"color": "#8ecae6", "width": 1.2}))
    longs = frame[frame[strategy] > 0]
    shorts = frame[frame[strategy] < 0]
    fig.add_trace(go.Scatter(x=longs["date"], y=longs["close"], mode="markers", name="Long signal", marker={"color": "#6ee7a8", "size": 7, "symbol": "triangle-up"}))
    fig.add_trace(go.Scatter(x=shorts["date"], y=shorts["close"], mode="markers", name="Short signal", marker={"color": "#f47b9d", "size": 7, "symbol": "triangle-down"}))
    fig.update_layout(title=f"{strategy} Signals", xaxis_title="", yaxis_title="Price")
    return _style_figure(fig, 390)


def _regime_heatmap(per_regime: pd.DataFrame, metric: str) -> go.Figure:
    if per_regime.empty or metric not in per_regime.columns:
        return go.Figure()
    pivot = per_regime.pivot_table(index="strategy", columns="regime_name", values=metric, aggfunc="mean")
    fig = px.imshow(pivot, text_auto=".3f", aspect="auto", color_continuous_scale="RdYlGn", title=f"Strategy x Regime {metric}")
    return _style_figure(fig, 430)


def _transition_heatmap(transitions: pd.DataFrame) -> go.Figure:
    if transitions.empty:
        return go.Figure()
    table = transitions.set_index("from_regime")
    fig = px.imshow(table, text_auto=".2f", aspect="auto", color_continuous_scale="Teal", title="Regime Transition Probabilities")
    fig.update_layout(xaxis_title="To regime", yaxis_title="From regime")
    return _style_figure(fig, 390)


def _stability_figure(stability: pd.DataFrame) -> go.Figure:
    if stability.empty:
        return go.Figure()
    share_cols = [column for column in stability.columns if column.startswith("regime_") and column.endswith("_share")]
    if not share_cols:
        return go.Figure()
    long = stability.melt(id_vars=["window_start", "window_end", "bars"], value_vars=share_cols, var_name="regime", value_name="share")
    long["regime"] = long["regime"].str.replace("regime_", "", regex=False).str.replace("_share", "", regex=False)
    fig = px.area(long, x="window_start", y="share", color="regime", title="Rolling Regime Mix", height=390)
    fig.update_layout(yaxis_tickformat=".0%")
    return _style_figure(fig, 390)


def _mc_filters(paths: pd.DataFrame, strategies: list[str], regimes: list[str], models: list[str]) -> pd.DataFrame:
    frame = paths.copy()
    if strategies and "strategy" in frame.columns:
        frame = frame[frame["strategy"].isin(strategies)]
    if regimes and "regime_filter" in frame.columns:
        frame = frame[frame["regime_filter"].isin(regimes)]
    if models and "stress_model" in frame.columns:
        frame = frame[frame["stress_model"].isin(models)]
    return frame


def _monte_carlo_paths_figure(paths: pd.DataFrame) -> go.Figure:
    if paths.empty:
        return go.Figure()
    frame = _downsample(paths.copy(), max_points=20000)
    frame["path_id"] = (
        frame["strategy"].astype(str)
        + " | "
        + frame["regime_filter"].astype(str)
        + " | "
        + frame["stress_model"].astype(str)
        + " | "
        + frame["simulation"].astype(str)
    )
    fig = px.line(
        frame,
        x="step",
        y="equity",
        color="strategy",
        line_group="path_id",
        facet_row="stress_model",
        title="Monte Carlo Simulated Equity Paths",
        height=max(460, 230 * max(1, frame["stress_model"].nunique())),
    )
    fig.update_traces(opacity=0.35)
    return _style_figure(fig)


def _monte_carlo_terminal_frame(paths: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    idx = ["strategy", "regime_filter", "stress_model", "simulation"]
    terminal = paths.sort_values("step").groupby(idx, as_index=False).tail(1).copy()
    terminal["terminal_return"] = terminal["equity"] / initial_capital - 1.0
    terminal["terminal_equity"] = terminal["equity"]
    dd_rows = []
    for keys, group in paths.groupby(idx):
        ordered = group.sort_values("step")
        drawdown = ordered["equity"] / ordered["equity"].cummax().replace(0.0, pd.NA) - 1.0
        dd_rows.append((*keys, float(drawdown.min())))
    dd = pd.DataFrame(dd_rows, columns=idx + ["max_drawdown"])
    return terminal[idx + ["terminal_return", "terminal_equity"]].merge(dd, on=idx, how="left")


def main() -> None:
    st.set_page_config(page_title="Market Regime Strategy Backtester", layout="wide")
    _inject_styles()

    output_root = Path(st.sidebar.text_input("Output directory", str(OUTPUT_ROOT))).expanduser()
    st.sidebar.markdown(
        """
        <div class="sidebar-note">
          Global run and scope controls live here. The Monte Carlo tab has its own focused workbench controls for strategy, regime, and scenario selection.
        </div>
        """,
        unsafe_allow_html=True,
    )
    include_smoke_runs = st.sidebar.checkbox("Include demo / sample runs", value=True)
    runs = _available_runs(output_root, include_smoke_runs=include_smoke_runs)
    if not runs:
        st.warning("No dashboard runs found. Run `python run_demo.py` from the repository root, then refresh this dashboard.")
        return

    run_labels = [_run_display_label(path) for path in runs]
    selected_label = st.sidebar.selectbox("Run", run_labels)
    run_dir = runs[run_labels.index(selected_label)]
    selected_run_name = run_dir.name
    initial_payload = load_dashboard_run(run_dir)
    manifest = initial_payload["manifest"]

    model_variants = initial_payload["model_variants"]
    model_option_labels: list[str] = []
    label_to_model_id: dict[str, str] = {}
    default_model_label = ""
    for variant in model_variants:
        label = _display_name(variant.get("display_label") or variant.get("model_id", ""))
        if bool(variant.get("is_best", False)):
            label = f"★ Recommended - {label}"
            default_model_label = label
        model_option_labels.append(label)
        label_to_model_id[label] = str(variant.get("model_id", ""))
    if not default_model_label and model_option_labels:
        default_model_label = model_option_labels[0]
    selected_model_label = st.sidebar.selectbox(
        "Regime model",
        model_option_labels,
        index=model_option_labels.index(default_model_label) if default_model_label in model_option_labels else 0,
        key=f"regime_model::{run_dir}",
    )
    selected_model_id = label_to_model_id[selected_model_label]
    payload = initial_payload if selected_model_id == initial_payload["selected_model_id"] else load_dashboard_run(run_dir, model_id=selected_model_id)
    active_model = payload["selected_model"]
    active_manifest = payload["model_manifest"] or payload["manifest"]
    active_diag = active_manifest.get("regime_diagnostics", payload["manifest"].get("regime_diagnostics", {}))

    st.markdown(
        """
        <div class="topbar">
          <div class="brand-lockup">
            <span class="brand-badge">Performance Dashboard</span>
            <span class="brand-copy">Regime Strategy Review Terminal</span>
          </div>
          <div class="topbar-note">A trading-first command center for regime-aware strategy research.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='run-strip'>"
        f"<span class='lab-pill'>Run { _html(selected_run_name) }</span>"
        f"<span class='lab-pill'>Symbol { _html(manifest.get('symbol', '')) }</span>"
        f"<span class='lab-pill'>Mode { _html(manifest.get('mode', 'full')) }</span>"
        f"<span class='lab-pill'>Model { _html(_display_name(active_model.get('display_label') or active_model.get('model_id', manifest.get('regime_model', '')))) }</span>"
        f"<span class='lab-pill'>Selection { _html(str(active_model.get('recommendation_source', 'current')).replace('_', ' ')) }</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    hmm_status = active_manifest.get("hmm_status", manifest.get("hmm_status"))
    if hmm_status not in (None, "", "not_requested"):
        st.info(str(hmm_status))

    base_metrics = _augment_metrics(payload["metrics"], payload["trades"])
    price_regimes = payload["price_regimes"]
    signals = payload["signals"]
    regime_stats = payload["regime_stats"]
    regime_stability = payload["regime_stability"]
    regime_transitions = payload["regime_transitions"]
    stress_summary = payload["stress_summary"]
    stress_paths = payload["monte_carlo_paths"]
    initial_capital = _safe_float(active_manifest.get("costs", {}).get("initial_capital", 100000.0)) or 100000.0
    all_strategies = (
        base_metrics["strategy"].tolist()
        if not base_metrics.empty and "strategy" in base_metrics.columns
        else [column for column in payload["returns"].columns if column != "date"]
    )
    available_regimes = sorted(price_regimes["regime_name"].dropna().unique().tolist()) if not price_regimes.empty and "regime_name" in price_regimes.columns else []
    mc_models = sorted(stress_paths["stress_model"].dropna().unique().tolist()) if not stress_paths.empty and "stress_model" in stress_paths.columns else []
    ranking_labels = {"Expectancy": "expectancy", "Win Rate": "win_rate", "Sharpe": "sharpe"}

    if len(model_variants) <= 1:
        st.info(
            "This run only contains one saved regime-model variant. "
            "The dashboard can switch between KMeans and HMM when the run artifacts were generated with `--include-hmm`."
        )
    if len(mc_models) <= 1 and not stress_paths.empty:
        only_model = mc_models[0] if mc_models else "one stress model"
        st.info(
            f"This run only includes saved Monte Carlo artifacts for `{only_model}`. "
            "The Monte Carlo workbench supports multiple stress models and regime filters when those artifacts were generated for the run."
        )

    if not price_regimes.empty and "date" in price_regimes.columns:
        min_date = price_regimes["date"].min().date()
        max_date = price_regimes["date"].max().date()
    else:
        today = pd.Timestamp.today().date()
        min_date = max_date = today

    base_key = f"{run_dir}::{selected_model_id}"
    base_context_key = f"dashboard_base_context::{base_key}"
    scope_context_key = f"dashboard_scope_context::{base_key}"
    date_key = f"command_date::{base_key}"
    regime_key = f"command_regimes::{base_key}"
    ranking_key = f"command_ranking::{base_key}"
    strategies_key = f"command_strategies::{base_key}"
    manual_strategy_key = f"manual_strategy_override::{base_key}"
    preset_key = f"strategy_preset::{base_key}"
    focus_strategy_key = f"focused_strategy::{base_key}"
    mc_focus_strategy_key = f"mc_focused_strategy::{base_key}"
    mc_focus_model_key = f"mc_focused_model::{base_key}"
    mc_focus_regime_key = f"mc_focused_regime::{base_key}"
    mc_path_density_key = f"mc_path_density::{base_key}"
    mc_distribution_key = f"mc_distribution_metric::{base_key}"
    preset_apply_key = f"preset_apply::{base_key}"

    base_context_signature = f"{run_dir}|{selected_model_id}"
    if st.session_state.get(base_context_key) != base_context_signature:
        st.session_state[date_key] = (min_date, max_date)
        st.session_state[regime_key] = available_regimes
        st.session_state[ranking_key] = "Expectancy"
        st.session_state[strategies_key] = []
        st.session_state[manual_strategy_key] = False
        st.session_state[preset_key] = "top_5"
        st.session_state[focus_strategy_key] = None
        st.session_state[mc_focus_strategy_key] = None
        st.session_state[mc_focus_model_key] = mc_models[0] if mc_models else None
        st.session_state[mc_focus_regime_key] = None
        st.session_state[mc_path_density_key] = "Balanced"
        st.session_state[mc_distribution_key] = "terminal return"
        st.session_state[preset_apply_key] = "top_5"
        st.session_state[base_context_key] = base_context_signature

    st.sidebar.markdown("### Scope")
    selected_dates = st.sidebar.date_input(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        key=date_key,
    )
    selected_regimes = st.sidebar.multiselect("Regimes", available_regimes, key=regime_key)
    ranking_label = st.sidebar.radio("Ranking lens", list(ranking_labels.keys()), key=ranking_key)
    preset_mode = st.sidebar.radio(
        "Strategy preset",
        ["top_5", "top_3", "champion_plus_challengers", "all"],
        format_func=lambda value: {
            "top_5": "Top 5",
            "top_3": "Top 3",
            "champion_plus_challengers": "Champion + Challengers",
            "all": "All",
        }[value],
        key=preset_key,
    )
    if st.sidebar.button("Reset scope", key=f"btn_reset::{base_key}", use_container_width=True):
        _reset_filter_state(
            date_key,
            (min_date, max_date),
            regime_key,
            available_regimes,
            ranking_key,
            "Expectancy",
            mc_focus_model_key,
            mc_models[0] if mc_models else None,
            mc_focus_regime_key,
            None,
            strategies_key,
            manual_strategy_key,
            preset_key,
            [focus_strategy_key, mc_focus_strategy_key],
        )
        st.session_state[mc_path_density_key] = "Balanced"
        st.session_state[mc_distribution_key] = "terminal return"
        st.session_state[preset_apply_key] = "top_5"
        st.rerun()

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif isinstance(selected_dates, list) and len(selected_dates) == 2:
        start_date, end_date = selected_dates[0], selected_dates[1]
    else:
        start_date, end_date = min_date, max_date
    selected_regimes = list(selected_regimes or [])
    ranking_label = ranking_label or "Expectancy"
    ranking_metric = ranking_labels[ranking_label]

    scoped_price = _filter_by_date_and_regime(price_regimes, start_date, end_date, selected_regimes)
    scoped_returns_all = _filter_returns(payload["returns"], scoped_price, all_strategies)
    scoped_trades_all = _filter_trades(payload["trades"], scoped_price, start_date, end_date, all_strategies)
    scoped_metrics = _sort_metric_frame(_compute_strategy_metrics(scoped_returns_all, scoped_trades_all, initial_capital), ranking_metric)
    if scoped_metrics.empty and not base_metrics.empty:
        scoped_metrics = _sort_metric_frame(base_metrics.copy(), ranking_metric)
    ranked_strategies = scoped_metrics["strategy"].tolist() if not scoped_metrics.empty else list(all_strategies)

    scope_signature = "|".join(
        [
            base_context_signature,
            str(start_date),
            str(end_date),
            ",".join(sorted(selected_regimes)),
            ranking_metric,
        ]
    )
    scope_changed = st.session_state.get(scope_context_key) != scope_signature
    if scope_changed:
        current_selection = [strategy for strategy in st.session_state.get(strategies_key, []) if strategy in ranked_strategies]
        if not st.session_state.get(manual_strategy_key, False):
            st.session_state[strategies_key] = _strategy_preset_selection(ranked_strategies, st.session_state.get(preset_key, "top_5"))
        else:
            st.session_state[strategies_key] = current_selection or _strategy_preset_selection(ranked_strategies, st.session_state.get(preset_key, "top_5"))
        if st.session_state.get(focus_strategy_key) not in st.session_state.get(strategies_key, []):
            st.session_state[focus_strategy_key] = st.session_state[strategies_key][0] if st.session_state.get(strategies_key) else None
        st.session_state[scope_context_key] = scope_signature

    if st.session_state.get(preset_apply_key) != preset_mode:
        _apply_strategy_preset(
            strategies_key,
            manual_strategy_key,
            preset_key,
            [focus_strategy_key, mc_focus_strategy_key],
            ranked_strategies,
            preset_mode,
        )
        st.session_state[preset_apply_key] = preset_mode

    st.sidebar.markdown("### Strategy set")
    st.sidebar.multiselect(
        "Strategies in view",
        ranked_strategies,
        key=strategies_key,
        on_change=_mark_manual_override,
        args=(manual_strategy_key,),
    )
    st.sidebar.caption("Manual strategy changes stick for this session. Change the scope or preset and the dashboard will restage the comparison set automatically.")

    selected_strategies = [strategy for strategy in st.session_state.get(strategies_key, []) if strategy in ranked_strategies]
    if not selected_strategies:
        selected_strategies = _strategy_preset_selection(ranked_strategies, st.session_state.get(preset_key, "top_5"))
        st.session_state[strategies_key] = selected_strategies
        st.session_state[manual_strategy_key] = False

    filtered_returns = _filter_returns(payload["returns"], scoped_price, selected_strategies)
    filtered_trades = _filter_trades(payload["trades"], scoped_price, start_date, end_date, selected_strategies)
    filtered_metrics = scoped_metrics[scoped_metrics["strategy"].isin(selected_strategies)].copy() if not scoped_metrics.empty else pd.DataFrame()
    filtered_metrics = _sort_metric_frame(filtered_metrics, ranking_metric)
    filtered_equity = _equity_from_returns(filtered_returns, initial_capital)
    scoped_per_regime = _compute_per_regime_metrics(scoped_returns_all, scoped_price, initial_capital)
    filtered_per_regime = scoped_per_regime[scoped_per_regime["strategy"].isin(selected_strategies)].copy() if not scoped_per_regime.empty else pd.DataFrame()
    filtered_per_regime = _sort_metric_frame(filtered_per_regime, ranking_metric, leading_cols=["regime_id"])
    regime_ranking = _build_regime_ranking(filtered_per_regime, ranking_metric)

    if filtered_equity.empty and not payload["equity"].empty:
        filtered_equity = _filter_frame_by_date(payload["equity"], start_date, end_date)
        if selected_strategies and "strategy" in filtered_equity.columns:
            filtered_equity = filtered_equity[filtered_equity["strategy"].isin(selected_strategies)]
    if filtered_per_regime.empty and not payload["per_regime_metrics"].empty:
        filtered_per_regime = payload["per_regime_metrics"].copy()
        if selected_strategies and "strategy" in filtered_per_regime.columns:
            filtered_per_regime = filtered_per_regime[filtered_per_regime["strategy"].isin(selected_strategies)]
        if selected_regimes and "regime_name" in filtered_per_regime.columns:
            filtered_per_regime = filtered_per_regime[filtered_per_regime["regime_name"].isin(selected_regimes)]
        filtered_per_regime = _sort_metric_frame(filtered_per_regime, ranking_metric, leading_cols=["regime_id"])
        regime_ranking = _build_regime_ranking(filtered_per_regime, ranking_metric)

    filtered_signals = pd.DataFrame()
    if not signals.empty:
        signal_columns = ["date"] + [strategy for strategy in selected_strategies if strategy in signals.columns]
        if len(signal_columns) > 1:
            filtered_signals = signals[signal_columns].copy()
            if not scoped_price.empty and "date" in scoped_price.columns:
                filtered_signals = scoped_price[["date"]].drop_duplicates().merge(filtered_signals, on="date", how="inner")
            filtered_signals = filtered_signals.sort_values("date").reset_index(drop=True)

    filtered_regime_stats = regime_stats.copy()
    if not filtered_regime_stats.empty and selected_regimes and "regime_name" in filtered_regime_stats.columns:
        filtered_regime_stats = filtered_regime_stats[filtered_regime_stats["regime_name"].isin(selected_regimes)]
    filtered_regime_stability = regime_stability.copy()
    if not filtered_regime_stability.empty:
        filtered_regime_stability = _filter_frame_by_date(filtered_regime_stability, start_date, end_date, column="window_start")

    model_leaderboard = _build_model_leaderboard(payload["model_variants"], payload["oos_regime_model_rankings"])
    active_rank_metric = str(active_model.get("rank_metric", "rank_value"))
    active_rank_value = _safe_float(active_model.get(active_rank_metric, active_model.get("rank_value", 0.0))) or 0.0
    selected_model_display = _display_name(active_model.get("display_label") or active_model.get("model_id", active_manifest.get("regime_model", "")))
    best_regime_lookup = _best_regime_lookup(filtered_per_regime if not filtered_per_regime.empty else scoped_per_regime, ranking_metric)

    mc_scope_options = _mc_local_focus_options(stress_summary, selected_strategies, selected_regimes)
    mc_focus_strategy_options = list(mc_scope_options["strategies"])
    mc_focus_model_options = list(mc_scope_options["models"])
    mc_focus_regime_options = list(mc_scope_options["regimes"])

    if scope_changed or (
        st.session_state.get(mc_focus_strategy_key) not in mc_focus_strategy_options
        or st.session_state.get(mc_focus_model_key) not in mc_focus_model_options
        or st.session_state.get(mc_focus_regime_key) not in mc_focus_regime_options
    ):
        _reset_monte_carlo_view_state(
            mc_focus_strategy_key,
            str(mc_scope_options["default_strategy"]) if mc_scope_options["default_strategy"] is not None else None,
            mc_focus_regime_key,
            str(mc_scope_options["default_regime"]) if mc_scope_options["default_regime"] is not None else None,
            mc_focus_model_key,
            str(mc_scope_options["default_model"]) if mc_scope_options["default_model"] is not None else None,
            mc_path_density_key,
            mc_distribution_key,
        )

    st.session_state["regime_strategy_active_selection"] = {
        "run": selected_run_name,
        "model_id": selected_model_id,
        "date_range": [str(start_date), str(end_date)],
        "regimes": selected_regimes,
        "strategies": selected_strategies,
        "ranking_metric": ranking_metric,
    }

    tab_overview, tab_regimes, tab_strategies, tab_stress, tab_artifacts = st.tabs(["Overview", "Regimes", "Strategies", "Monte Carlo", "Artifacts"])

    with tab_overview:
        if filtered_metrics.empty:
            st.info("No strategy metrics match the current scope.")
        else:
            champion = filtered_metrics.iloc[0]
            champion_strategy = str(champion["strategy"])
            champion_label = _display_name(champion_strategy)
            total_return = _safe_float(champion.get("total_return", 0.0)) or 0.0
            win_rate = _safe_float(champion.get("win_rate", 0.0)) or 0.0
            expected_r = _safe_float(champion.get("expected_r", 0.0)) or 0.0
            max_drawdown = _safe_float(champion.get("max_drawdown", 0.0)) or 0.0
            num_trades = int(_safe_float(champion.get("num_trades", 0)) or 0)
            profit_factor = _safe_float(champion.get("profit_factor", 0.0)) or 0.0
            pnl = total_return * initial_capital
            champion_regime, champion_regime_score = best_regime_lookup.get(champion_strategy, ("", 0.0))
            shown_trades = int(filtered_metrics["num_trades"].sum()) if "num_trades" in filtered_metrics.columns else num_trades
            date_window = manifest.get("date_window", {})
            date_note = f"{date_window.get('start_date', '')} to {date_window.get('end_date', '')}"
            sparkline = _sparkline_svg(filtered_equity, champion_strategy)
            score_label = active_rank_metric.replace("_", " ").title()
            summary_filter = stress_summary.copy()
            if selected_strategies and "strategy" in summary_filter.columns:
                summary_filter = summary_filter[summary_filter["strategy"].isin(selected_strategies)]
            if selected_regimes and "regime_filter" in summary_filter.columns:
                summary_filter = summary_filter[summary_filter["regime_filter"].isin(selected_regimes)]
            mc_loss_value = _safe_float(summary_filter["probability_of_loss"].mean()) or 0.0 if not summary_filter.empty and "probability_of_loss" in summary_filter.columns else 0.0
            mc_tone = "good" if mc_loss_value < 0.35 else "warn" if mc_loss_value < 0.55 else "bad"
            kpis = [
                _metric_html("Champion", champion_label, f"{ranking_label} {_format_number(champion.get(ranking_metric, 0.0))}", _tone(_safe_float(champion.get(ranking_metric, 0.0)) or 0.0)),
                _metric_html("Total Return", _format_pct(total_return), "Champion total return", _tone(total_return)),
                _metric_html("Win Rate", _format_pct(win_rate), f"{num_trades:,} trades", _tone(win_rate - 0.5)),
                _metric_html("Expected R", _format_number(expected_r), "Per-trade risk proxy", _tone(expected_r)),
                _metric_html("Profit Factor", _format_number(profit_factor), "Gross profit / gross loss", _tone(profit_factor - 1.0)),
                _metric_html("MC Loss Risk", _format_pct(mc_loss_value), "Average across selected Monte Carlo scenarios", mc_tone),
                _metric_html("Best Regime", champion_regime or "No regime data", f"{ranking_label} {_format_number(champion_regime_score)}", "good" if champion_regime else "warn"),
                _metric_html("Max Drawdown", _format_pct(max_drawdown), "Champion pullback", "bad" if max_drawdown < 0 else "warn"),
                _metric_html("Selected Trades", f"{shown_trades:,}", "Across the current comparison set", "good"),
            ]
            insights = [
                _insight_html("Champion", f"{champion_label} owns the current {ranking_label.lower()} lead inside this scope.", _tone(_safe_float(champion.get(ranking_metric, 0.0)) or 0.0)),
                _insight_html("Model", f"{selected_model_display} is active across every tab in this view.", "good"),
                _insight_html("Risk", f"Average Monte Carlo loss probability is {_format_pct(mc_loss_value)} for the selected comparison set.", mc_tone),
            ]
            st.markdown(
                f"""
                <div class="command-layout">
                  <section class="command-panel">
                    <div class="command-header">
                      <div class="command-copy">
                        <div class="eyebrow">Performance dashboard</div>
                        <div class="dashboard-title">Market Regime<br/>Backtester</div>
                        <div class="subline">{_html(_format_money(pnl))} champion P&L | {_html(_format_pct(win_rate))} win rate | {_html(_format_number(expected_r))} expected R</div>
                      </div>
                      <div class="command-badges">
                        <span class="status-pill">Champion in focus</span>
                        <span class="status-pill">{shown_trades:,} trades</span>
                        <span class="status-pill negative">MDD {_html(_format_pct(abs(max_drawdown)))}</span>
                        <span class="status-pill">{_html(selected_model_display)}</span>
                      </div>
                    </div>
                    <div class="command-sparkline">{sparkline}</div>
                    <div class="command-insights">{''.join(insights)}</div>
                    <div class="kpi-grid">{''.join(kpis)}</div>
                  </section>
                  <aside class="desk-panel">
                    <div class="eyebrow">Desk status</div>
                    <div class="console-title">Update console</div>
                    <span class="status-pill">Scope synced</span>
                    <div class="console-time">{_format_timestamp(manifest.get("created_at", ""))}</div>
                    <div class="console-note">{len(all_strategies)} strategies across {len(available_regimes)} regimes. Window: {_html(date_note)}.</div>
                    <div class="console-grid">
                      {_console_card("Selected model", selected_model_display, "Active dashboard bundle", "good")}
                      {_console_card(score_label, _format_number(active_rank_value), "Recommendation score", "good")}
                      {_console_card("Ranking lens", ranking_label, "Top 5 defaults use this lens", "good")}
                      {_console_card("Current scope", str(manifest.get("symbol", "")), "Selected output run")}
                    </div>
                    <div class="note-card"><div class="console-label">Review note</div>
                    The command bar keeps scope changes close to the visuals, and the strategy set auto-refreshes to the top five unless you manually override it.</div>
                    <div class="note-card"><div class="console-label">Risk note</div>
                    The Monte Carlo tab now drills into one scenario at a time while keeping the broader risk table for comparison.</div>
                  </aside>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='section-heading'><div><p class='eyebrow'>Selection</p><h2>Champion and challengers</h2></div><p>The current comparison set starts with the best five strategies under the active ranking lens.</p></div>",
                unsafe_allow_html=True,
            )
            _render_strategy_rail(filtered_metrics.head(5), ranking_label, ranking_metric, best_regime_lookup)

            chart_cols = st.columns(2)
            with chart_cols[0]:
                if not filtered_equity.empty and selected_strategies:
                    st.plotly_chart(_equity_figure(filtered_equity, selected_strategies[:5]), use_container_width=True, key="overview_equity")
            with chart_cols[1]:
                if not filtered_equity.empty and selected_strategies:
                    st.plotly_chart(_drawdown_figure(filtered_equity, selected_strategies[:5]), use_container_width=True, key="overview_drawdown")

            compact_cols = [column for column in ["strategy", ranking_metric, "total_return", "win_rate", "expected_r", "max_drawdown", "num_trades"] if column in filtered_metrics.columns]
            st.markdown(
                f"<div class='section-heading'><div><p class='eyebrow'>Leaderboard</p><h2>Compact strategy leaderboard</h2></div><p>Sorted by {ranking_label.lower()} with expectancy, Sharpe, and bars as tie-breaks.</p></div>",
                unsafe_allow_html=True,
            )
            st.dataframe(filtered_metrics[compact_cols].head(10) if compact_cols else filtered_metrics.head(10), use_container_width=True)

            if not model_leaderboard.empty:
                st.markdown(
                    "<div class='section-heading'><div><p class='eyebrow'>Models</p><h2>Regime model leaderboard</h2></div><p>The starred row is the recommended default for this run.</p></div>",
                    unsafe_allow_html=True,
                )
                model_columns = [column for column in ["label", "oos_score_mean", "score", "blocks", "failed_blocks", "oos_silhouette", "oos_interpretability"] if column in model_leaderboard.columns]
                st.dataframe(model_leaderboard[model_columns] if model_columns else model_leaderboard, use_container_width=True)

    with tab_regimes:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows In View", f"{len(scoped_price):,}")
        c2.metric("Trade Days", f"{int(pd.to_datetime(scoped_price['date']).dt.normalize().nunique()) if not scoped_price.empty else 0:,}")
        c3.metric("Feature Rows", f"{len(payload['regime_features']):,}")
        c4.metric(active_rank_metric.replace("_", " ").title(), f"{active_rank_value:.3f}")

        if not scoped_price.empty:
            st.plotly_chart(_price_regime_figure(scoped_price), use_container_width=True, key="regime_price")

        c_left, c_right = st.columns(2)
        with c_left:
            if not filtered_regime_stats.empty:
                st.subheader("Regime Profiles")
                st.dataframe(filtered_regime_stats, use_container_width=True)
        with c_right:
            if not regime_transitions.empty:
                st.plotly_chart(_transition_heatmap(regime_transitions), use_container_width=True, key="regime_transition")

        if not filtered_regime_stability.empty:
            st.plotly_chart(_stability_figure(filtered_regime_stability), use_container_width=True, key="regime_stability")

        if not payload["oos_regime_blocks"].empty:
            st.subheader("Walk-Forward Block Scores")
            st.dataframe(payload["oos_regime_blocks"], use_container_width=True)

        with st.expander("Regime diagnostics JSON", expanded=False):
            st.json(active_diag)

    with tab_strategies:
        if filtered_metrics.empty:
            st.info("No strategy metrics match the current scope.")
        else:
            _render_strategy_rail(filtered_metrics.head(5), ranking_label, ranking_metric, best_regime_lookup)

            focus_options = selected_strategies or ranked_strategies
            if focus_options and st.session_state.get(focus_strategy_key) not in focus_options:
                st.session_state[focus_strategy_key] = focus_options[0]
            focus_cols = st.columns([1.3, 3.7])
            with focus_cols[0]:
                focused_strategy = st.selectbox("Focused strategy", focus_options, key=focus_strategy_key)
            with focus_cols[1]:
                top = filtered_metrics.iloc[0]
                st.markdown(
                    f"<div class='command-help'>The comparison canvas keeps the full selected set in view while signals and trades stay pinned to {_html(_display_name(focused_strategy))}.</div>",
                    unsafe_allow_html=True,
                )

            focus_row = filtered_metrics[filtered_metrics["strategy"].astype(str) == str(focused_strategy)]
            if not focus_row.empty:
                focus_metric = focus_row.iloc[0]
                focus_best_regime, focus_best_score = best_regime_lookup.get(focused_strategy, ("", 0.0))
                strategy_cards = [
                    _route_metric_html("Focused strategy", _display_name(focused_strategy), f"{ranking_label} lens is driving the current comparison set.", _tone(_safe_float(focus_metric.get(ranking_metric, 0.0)) or 0.0)),
                    _route_metric_html("Total return", _format_pct(focus_metric.get("total_return", 0.0)), "Return inside the active date and regime scope.", _tone(_safe_float(focus_metric.get("total_return", 0.0)) or 0.0)),
                    _route_metric_html("Win rate", _format_pct(focus_metric.get("win_rate", 0.0)), f"{int(_safe_float(focus_metric.get('num_trades', 0)) or 0):,} trades in scope.", _tone((_safe_float(focus_metric.get("win_rate", 0.0)) or 0.0) - 0.5)),
                    _route_metric_html("Best regime", focus_best_regime or "No regime data", f"{ranking_label} {_format_number(focus_best_score)} in the best-fit regime.", "positive" if focus_best_regime else ""),
                ]
                st.markdown(f"<div class='route-metrics'>{''.join(strategy_cards)}</div>", unsafe_allow_html=True)

            compare_cols = st.columns(2)
            with compare_cols[0]:
                if not filtered_equity.empty and selected_strategies:
                    st.plotly_chart(_equity_figure(filtered_equity, selected_strategies), use_container_width=True, key="strategy_equity")
            with compare_cols[1]:
                if not filtered_equity.empty and selected_strategies:
                    st.plotly_chart(_drawdown_figure(filtered_equity, selected_strategies), use_container_width=True, key="strategy_drawdown")

            lower_cols = st.columns(2)
            with lower_cols[0]:
                if not filtered_per_regime.empty and ranking_metric in filtered_per_regime.columns:
                    st.plotly_chart(_regime_heatmap(filtered_per_regime, ranking_metric), use_container_width=True, key="strategy_regime_heatmap")
                else:
                    st.info("No per-regime metrics available for this run.")
            with lower_cols[1]:
                if focused_strategy and not filtered_signals.empty:
                    st.plotly_chart(_signal_figure(filtered_signals, scoped_price if not scoped_price.empty else price_regimes, focused_strategy), use_container_width=True, key="strategy_signals")
                else:
                    st.info("No signal overlay available for the current scope.")

            st.markdown(
                "<div class='section-heading'><div><p class='eyebrow'>Comparison</p><h2>Full strategy table</h2></div><p>`expected_r` is a proxy R-multiple: average trade PnL normalized by the strategy’s average losing-trade magnitude.</p></div>",
                unsafe_allow_html=True,
            )
            st.dataframe(filtered_metrics, use_container_width=True)

            if not regime_ranking.empty:
                st.markdown(
                    "<div class='section-heading'><div><p class='eyebrow'>Regimes</p><h2>Best regime by strategy</h2></div><p>The heatmap above is the fast visual read; this table is the exact drilldown.</p></div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(regime_ranking, use_container_width=True)

            if not filtered_trades.empty:
                focus_trades = filtered_trades[filtered_trades["strategy"].astype(str) == str(focused_strategy)] if "strategy" in filtered_trades.columns else filtered_trades
                st.markdown(
                    "<div class='section-heading'><div><p class='eyebrow'>Execution</p><h2>Focused strategy trades</h2></div><p>Trade drilldown follows the focused strategy without clearing the broader comparison set.</p></div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(focus_trades, use_container_width=True, height=320)

    with tab_stress:
        summary_filter = stress_summary.copy()
        if selected_strategies and "strategy" in summary_filter.columns:
            summary_filter = summary_filter[summary_filter["strategy"].isin(selected_strategies)]
        if selected_regimes and "regime_filter" in summary_filter.columns:
            summary_filter = summary_filter[summary_filter["regime_filter"].isin(selected_regimes)]

        if stress_paths.empty or summary_filter.empty:
            st.info("No stress outputs available for this run.")
        else:
            st.markdown(
                "<div class='section-heading'><div><p class='eyebrow'>Risk lab</p><h2>Monte Carlo workbench</h2></div><p>Saved scenario paths only. Use the workbench below to focus the path cloud by strategy, regime, and stress model.</p></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='mc-toolbar-note'>This panel filters the saved Monte Carlo artifacts for the active run. The sidebar still controls the broader dashboard scope; the inputs below only change the focused Monte Carlo scenario.</div>",
                unsafe_allow_html=True,
            )

            toolbar_cols = st.columns([1.15, 1.15, 1.0, 0.82, 0.98, 0.82])
            with toolbar_cols[0]:
                if mc_focus_strategy_options:
                    st.selectbox("Monte Carlo strategy", mc_focus_strategy_options, key=mc_focus_strategy_key)
            with toolbar_cols[1]:
                if mc_focus_regime_options:
                    st.selectbox("Monte Carlo regime", mc_focus_regime_options, key=mc_focus_regime_key)
            with toolbar_cols[2]:
                if mc_focus_model_options:
                    st.selectbox("Stress model", mc_focus_model_options, key=mc_focus_model_key)
            with toolbar_cols[3]:
                st.selectbox("Path density", ["Low", "Balanced", "High"], key=mc_path_density_key)
            with toolbar_cols[4]:
                st.selectbox("Distribution metric", ["terminal return", "terminal equity", "max drawdown"], key=mc_distribution_key)
            with toolbar_cols[5]:
                reset_mc_view = st.button("Reset Monte Carlo view", key=f"btn_reset_mc::{base_key}", use_container_width=True)

            if reset_mc_view:
                _reset_monte_carlo_view_state(
                    mc_focus_strategy_key,
                    str(mc_scope_options["default_strategy"]) if mc_scope_options["default_strategy"] is not None else None,
                    mc_focus_regime_key,
                    str(mc_scope_options["default_regime"]) if mc_scope_options["default_regime"] is not None else None,
                    mc_focus_model_key,
                    str(mc_scope_options["default_model"]) if mc_scope_options["default_model"] is not None else None,
                    mc_path_density_key,
                    mc_distribution_key,
                )
                st.rerun()

            focus_mc_strategy = st.session_state.get(mc_focus_strategy_key)
            focus_mc_model = st.session_state.get(mc_focus_model_key)
            focus_mc_regime = st.session_state.get(mc_focus_regime_key)
            distribution_metric = st.session_state.get(mc_distribution_key, "terminal return")
            density_label = st.session_state.get(mc_path_density_key, "Balanced")
            density_map = {"Low": 24, "Balanced": 48, "High": 80}

            focus_paths = _mc_filters(stress_paths, [focus_mc_strategy], [focus_mc_regime], [focus_mc_model])
            focus_summary = _mc_focus_summary(summary_filter, focus_mc_strategy, focus_mc_regime, focus_mc_model)
            focus_terminal = _monte_carlo_terminal_frame(focus_paths, initial_capital)

            if focus_paths.empty or focus_summary is None:
                st.info("No Monte Carlo scenario matches the current strategy / regime / model combination.")
            else:
                st.markdown(
                    f"<div class='mc-chart-caption'><span>Viewing <strong>{_html(_display_name(focus_mc_strategy or 'selected strategy'))}</strong> in <strong>{_html(_display_name(focus_mc_regime or 'selected regime'))}</strong> under <strong>{_html(_display_name(focus_mc_model or 'selected model'))}</strong>.</span><span>{_html(density_label)} path density from the saved simulation set.</span></div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _monte_carlo_overlay_figure(focus_paths, max_paths=density_map.get(density_label, 48)),
                    use_container_width=True,
                    key="mc_paths_overlay",
                )

                cards_html = _mc_results_cards_html(focus_terminal, focus_summary, initial_capital)
                if cards_html:
                    st.markdown(cards_html, unsafe_allow_html=True)

                if not focus_terminal.empty:
                    st.markdown(
                        "<div class='section-heading'><div><p class='eyebrow'>Outcomes</p><h2>Distribution explorer</h2></div><p>The main chart is the path read. Use this distribution view for the terminal outcome profile of the same focused scenario.</p></div>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(_mc_distribution_figure(focus_terminal, distribution_metric), use_container_width=True, key="mc_distribution")
                else:
                    st.info("No Monte Carlo terminal paths available for this scenario.")

                compare_frame = summary_filter.copy()
                if focus_mc_model and "stress_model" in compare_frame.columns:
                    compare_frame = compare_frame[compare_frame["stress_model"].astype(str) == str(focus_mc_model)]
                if focus_mc_regime and "regime_filter" in compare_frame.columns:
                    compare_frame = compare_frame[compare_frame["regime_filter"].astype(str) == str(focus_mc_regime)]
                if compare_frame.empty:
                    compare_frame = summary_filter.copy()

                comparison_columns = [
                    "strategy",
                    "stress_model",
                    "regime_filter",
                    "simulations",
                    "horizon",
                    "terminal_equity_p50",
                    "terminal_equity_p95",
                    "max_drawdown_p95",
                    "probability_of_loss",
                    "risk_of_ruin",
                ]
                available_columns = [column for column in comparison_columns if column in compare_frame.columns]
                compare_frame = compare_frame[available_columns].sort_values(
                    ["probability_of_loss", "risk_of_ruin"] if {"probability_of_loss", "risk_of_ruin"}.issubset(compare_frame.columns) else available_columns[:1],
                    ascending=True,
                )
                st.markdown(
                    "<div class='section-heading'><div><p class='eyebrow'>Scenarios</p><h2>Compact scenario table</h2></div><p>Same scope, same saved artifacts, with the focused model and regime held in view for exact comparisons.</p></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='mc-table-note'>The chart above is the fast read. This table is the exact scenario drilldown for the active Monte Carlo slice.</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(compare_frame, use_container_width=True)

    with tab_artifacts:
        st.subheader("Run Manifest")
        st.json(manifest)
        st.subheader("Active Model Manifest")
        st.json(active_manifest)
        st.subheader("Loaded Tables")
        table_rows = []
        for key, value in payload.items():
            if isinstance(value, pd.DataFrame):
                table_rows.append({"table": key, "rows": len(value), "columns": len(value.columns)})
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


if __name__ == "__main__":
    main()
