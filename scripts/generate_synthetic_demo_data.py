from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "regime_strategy_backtester" / "data" / "demo" / "NQ_1M_demo.csv"


def build_synthetic_nq_rows(minutes: int = 7_500, seed: int = 181) -> pd.DataFrame:
    """Build a deterministic futures-style minute dataset for public demos.

    The schema mirrors the raw Databento-style columns accepted by the loader,
    but every price and volume value is generated locally from a seeded process.
    """

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-02 14:30:00", periods=minutes, freq="min", tz="UTC")

    regimes = np.resize(np.repeat(np.arange(4), minutes // 8), minutes)
    drift = np.array([0.00002, 0.00012, -0.00009, 0.00000])[regimes]
    volatility = np.array([0.00018, 0.00035, 0.00048, 0.00085])[regimes]
    shocks = rng.normal(drift, volatility)
    close = 16_800.0 * np.cumprod(1.0 + shocks)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(0.25, np.abs(rng.normal(1.6, 0.45, minutes)))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume_front = rng.integers(80, 900, size=minutes) + regimes * 45
    volume_next = rng.integers(5, 180, size=minutes)

    rows = []
    contracts = [("NQH4", 8401, 0.0), ("NQM4", 8402, 8.0)]
    for symbol, instrument_id, price_offset in contracts:
        volume = volume_front if symbol == "NQH4" else volume_next
        rows.append(
            pd.DataFrame(
                {
                    "ts_event": timestamps.strftime("%Y-%m-%dT%H:%M:%S.%f000Z"),
                    "rtype": 33,
                    "publisher_id": 1,
                    "instrument_id": instrument_id,
                    "open": open_ + price_offset,
                    "high": high + price_offset,
                    "low": low + price_offset,
                    "close": close + price_offset,
                    "volume": volume,
                    "symbol": symbol,
                }
            )
        )

    frame = pd.concat(rows, ignore_index=True).sort_values(["ts_event", "symbol"]).reset_index(drop=True)
    price_columns = ["open", "high", "low", "close"]
    frame[price_columns] = frame[price_columns].round(2)
    return frame


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = build_synthetic_nq_rows()
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(frame):,} synthetic rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
