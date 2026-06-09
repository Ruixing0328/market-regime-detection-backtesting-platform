from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import pandas as pd

from regime_strategy_backtester.config import CACHE_ROOT, NY_TZ, portable_path

OUTRIGHT_PATTERNS = {
    "ES": re.compile(r"^ES[HMUZ]\d{1,2}$"),
    "NQ": re.compile(r"^NQ[HMUZ]\d{1,2}$"),
}


@dataclass
class FrontContractSummary:
    root_symbol: str
    rows: int
    trade_days: int
    start_date: str
    end_date: str
    cache_path: str


def parse_hhmm(value: str) -> int:
    hour, minute = [int(part) for part in value.split(":", 1)]
    return hour * 60 + minute


def _cache_label(session_start: Optional[str], session_end: Optional[str]) -> str:
    if session_start is None or session_end is None:
        return "all_day"
    return f"{session_start.replace(':', '')}_{session_end.replace(':', '')}"


def _cache_path(
    cache_root: Path,
    root_symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    session_start: Optional[str],
    session_end: Optional[str],
    back_adjust: bool,
) -> Path:
    start_label = "all" if start_date is None else start_date.replace("-", "")
    end_label = "all" if end_date is None else end_date.replace("-", "")
    session_label = _cache_label(session_start, session_end)
    if back_adjust:
        name = f"{root_symbol.lower()}_front_1m_{start_label}_{end_label}_{session_label}_back_adjusted.csv"
    else:
        name = f"{root_symbol.lower()}_front_1m_{start_label}_{end_label}_{session_label}.csv"
    return cache_root / name


def _base_filter(
    chunk: pd.DataFrame,
    root_symbol: str,
    start_day: Optional[pd.Timestamp],
    end_day: Optional[pd.Timestamp],
    session_start: Optional[str],
    session_end: Optional[str],
) -> pd.DataFrame:
    pattern = OUTRIGHT_PATTERNS.get(root_symbol)
    if pattern is None:
        raise ValueError(f"Unsupported root symbol: {root_symbol}")

    ts_ny = pd.to_datetime(chunk["ts_event"], utc=True).dt.tz_convert(NY_TZ)
    local_dt = ts_ny.dt.tz_localize(None)
    local_day = local_dt.dt.normalize()

    mask = chunk["symbol"].astype(str).str.match(pattern)
    if start_day is not None:
        mask &= local_day >= start_day
    if end_day is not None:
        mask &= local_day <= end_day

    if session_start is not None and session_end is not None:
        minute_of_day = local_dt.dt.hour * 60 + local_dt.dt.minute
        start_minute = parse_hhmm(session_start)
        end_minute = parse_hhmm(session_end)
        mask &= (minute_of_day >= start_minute) & (minute_of_day < end_minute)

    out = chunk.loc[mask].copy()
    if out.empty:
        return out

    out["date"] = local_dt.loc[mask]
    out["trade_day"] = local_day.loc[mask]
    return out


def filter_time_window(df: pd.DataFrame, session_start: str, session_end: str) -> pd.DataFrame:
    out = df.copy()
    minute_of_day = out["date"].dt.hour * 60 + out["date"].dt.minute
    start_minute = parse_hhmm(session_start)
    end_minute = parse_hhmm(session_end)
    mask = (minute_of_day >= start_minute) & (minute_of_day < end_minute)
    return out.loc[mask].sort_values(["symbol", "date"]).reset_index(drop=True)


def _compute_roll_adjustments(front_symbols: pd.DataFrame, roll_prices: pd.DataFrame) -> pd.DataFrame:
    if front_symbols.empty or roll_prices.empty:
        return pd.DataFrame(columns=["trade_day", "old_symbol", "new_symbol", "roll_gap", "source"])

    symbols = front_symbols[["trade_day", "front_symbol"]].sort_values("trade_day").reset_index(drop=True).copy()
    symbols["old_symbol"] = symbols["front_symbol"].shift(1)
    switches = symbols[symbols["front_symbol"] != symbols["old_symbol"]].iloc[1:].copy()
    if switches.empty:
        return pd.DataFrame(columns=["trade_day", "old_symbol", "new_symbol", "roll_gap", "source"])

    roll_rows: list[dict[str, object]] = []
    for switch in switches.itertuples(index=False):
        day_prices = roll_prices[roll_prices["trade_day"] == switch.trade_day]
        if day_prices.empty:
            continue

        old_rows = (
            day_prices[day_prices["symbol"] == switch.old_symbol][["date", "close"]]
            .drop_duplicates(subset=["date"], keep="last")
            .rename(columns={"close": "old_close"})
        )
        new_rows = (
            day_prices[day_prices["symbol"] == switch.front_symbol][["date", "close"]]
            .drop_duplicates(subset=["date"], keep="last")
            .rename(columns={"close": "new_close"})
        )
        if old_rows.empty or new_rows.empty:
            continue

        overlap = old_rows.merge(new_rows, on="date", how="inner")
        if not overlap.empty:
            roll_gap = float((overlap["new_close"] - overlap["old_close"]).median())
            source = "median_overlap_close"
        else:
            roll_gap = float(new_rows["new_close"].iloc[0] - old_rows["old_close"].iloc[-1])
            source = "first_new_minus_last_old"

        roll_rows.append(
            {
                "trade_day": pd.Timestamp(switch.trade_day),
                "old_symbol": str(switch.old_symbol),
                "new_symbol": str(switch.front_symbol),
                "roll_gap": roll_gap,
                "source": source,
            }
        )

    if not roll_rows:
        return pd.DataFrame(columns=["trade_day", "old_symbol", "new_symbol", "roll_gap", "source"])
    return pd.DataFrame(roll_rows).sort_values("trade_day").reset_index(drop=True)


def _apply_additive_back_adjustment(
    df: pd.DataFrame,
    front_symbols: pd.DataFrame,
    roll_prices: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return df

    adjustments = _compute_roll_adjustments(front_symbols, roll_prices)
    if adjustments.empty:
        return df

    out = df.copy()
    out["trade_day"] = out["date"].dt.normalize()

    day_table = front_symbols[["trade_day"]].drop_duplicates().sort_values("trade_day").reset_index(drop=True)
    gap_by_day = adjustments.set_index("trade_day")["roll_gap"]
    day_table["roll_gap"] = day_table["trade_day"].map(gap_by_day).fillna(0.0)
    day_table["back_adjustment"] = day_table["roll_gap"].iloc[::-1].cumsum().iloc[::-1] - day_table["roll_gap"]

    out = out.merge(day_table[["trade_day", "back_adjustment"]], on="trade_day", how="left")
    out["back_adjustment"] = out["back_adjustment"].fillna(0.0)
    for column in ("open", "high", "low", "close"):
        out[column] = out[column] + out["back_adjustment"]

    return out.drop(columns=["trade_day", "back_adjustment"])


def load_front_contract_minute_series(
    data_path: str | Path,
    root_symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    session_start: Optional[str] = None,
    session_end: Optional[str] = None,
    back_adjust: bool = False,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    chunksize: int = 750_000,
) -> tuple[pd.DataFrame, FrontContractSummary]:
    root_symbol = str(root_symbol).upper()
    path = Path(data_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else CACHE_ROOT
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(
        cache_root=cache_root,
        root_symbol=root_symbol,
        start_date=start_date,
        end_date=end_date,
        session_start=session_start,
        session_end=session_end,
        back_adjust=back_adjust,
    )

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        cached["symbol"] = root_symbol
        summary = FrontContractSummary(
            root_symbol=root_symbol,
            rows=int(len(cached)),
            trade_days=int(cached["date"].dt.normalize().nunique()) if not cached.empty else 0,
            start_date=str(cached["date"].min().date()) if not cached.empty else "",
            end_date=str(cached["date"].max().date()) if not cached.empty else "",
            cache_path=portable_path(cache_path),
        )
        return cached, summary

    start_day = pd.Timestamp(start_date).normalize() if start_date else None
    end_day = pd.Timestamp(end_date).normalize() if end_date else None

    volume_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=["ts_event", "symbol", "volume"], chunksize=chunksize):
        filtered = _base_filter(
            chunk,
            root_symbol=root_symbol,
            start_day=start_day,
            end_day=end_day,
            session_start=session_start,
            session_end=session_end,
        )
        if filtered.empty:
            continue
        grouped = filtered.groupby(["trade_day", "symbol"], as_index=False)["volume"].sum()
        volume_parts.append(grouped)

    if not volume_parts:
        raise ValueError(f"No rows left after filtering raw minute data for {root_symbol}")

    volume_df = pd.concat(volume_parts, ignore_index=True)
    volume_df = volume_df.groupby(["trade_day", "symbol"], as_index=False)["volume"].sum()
    dominant = (
        volume_df.sort_values(["trade_day", "volume", "symbol"], ascending=[True, False, True])
        .drop_duplicates(subset=["trade_day"], keep="first")
        .rename(columns={"symbol": "front_symbol"})
    )
    front_by_day = dict(zip(dominant["trade_day"], dominant["front_symbol"]))

    row_parts: list[pd.DataFrame] = []
    roll_parts: list[pd.DataFrame] = []
    switch_days: set[pd.Timestamp] = set()
    if back_adjust:
        switch_days = set(dominant.loc[dominant["front_symbol"] != dominant["front_symbol"].shift(1), "trade_day"].iloc[1:])
    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        filtered = _base_filter(
            chunk,
            root_symbol=root_symbol,
            start_day=start_day,
            end_day=end_day,
            session_start=session_start,
            session_end=session_end,
        )
        if filtered.empty:
            continue
        if back_adjust and switch_days:
            switch_rows = filtered[filtered["trade_day"].isin(switch_days)]
            if not switch_rows.empty:
                roll_parts.append(switch_rows[["trade_day", "date", "symbol", "close"]].copy())
        filtered["front_symbol"] = filtered["trade_day"].map(front_by_day)
        filtered = filtered[filtered["symbol"] == filtered["front_symbol"]].copy()
        if filtered.empty:
            continue
        filtered["symbol"] = root_symbol
        row_parts.append(filtered[["date", "open", "high", "low", "close", "volume", "symbol"]])

    if not row_parts:
        raise ValueError(f"Unable to build front-contract series for {root_symbol}")

    out = (
        pd.concat(row_parts, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if back_adjust and roll_parts:
        roll_prices = pd.concat(roll_parts, ignore_index=True)
        out = _apply_additive_back_adjustment(out, dominant[["trade_day", "front_symbol"]], roll_prices)
    out.to_csv(cache_path, index=False)

    summary = FrontContractSummary(
        root_symbol=root_symbol,
        rows=int(len(out)),
        trade_days=int(out["date"].dt.normalize().nunique()),
        start_date=str(out["date"].min().date()),
        end_date=str(out["date"].max().date()),
        cache_path=portable_path(cache_path),
    )
    return out, summary
