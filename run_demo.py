from __future__ import annotations

import argparse

from market_regime_platform.config import OUTPUT_ROOT, resolve_data_path
from market_regime_platform.pipeline import build_parser, run_pipeline


def build_demo_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bundled Market Regime Detection & Backtesting Platform demo workflow")
    parser.add_argument("--run-id", default="public_demo")
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--stress-horizon", type=int, default=60)
    parser.add_argument("--max-saved-paths", type=int, default=5)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return parser


def main() -> None:
    demo_args = build_demo_parser().parse_args()
    bundled_data = resolve_data_path("NQ")

    pipeline_args = build_parser().parse_args(
        [
            "--symbol",
            "NQ",
            "--data-path",
            str(bundled_data),
            "--include-hmm",
            "--sample-rows",
            str(demo_args.sample_rows),
            "--simulations",
            str(demo_args.simulations),
            "--stress-horizon",
            str(demo_args.stress_horizon),
            "--max-saved-paths",
            str(demo_args.max_saved_paths),
            "--max-equity-points-per-strategy",
            "1500",
            "--output-dir",
            demo_args.output_dir,
            "--run-id",
            demo_args.run_id,
        ]
    )
    run_dir = run_pipeline(pipeline_args)
    print(f"Demo run complete: {run_dir}")
    print("Launch the dashboard with: streamlit run market_regime_platform/app.py")


if __name__ == "__main__":
    main()
