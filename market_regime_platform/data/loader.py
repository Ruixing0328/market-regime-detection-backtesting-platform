from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Optional

import pandas as pd

from market_regime_platform.config import NY_TZ, portable_path, resolve_data_path
from market_regime_platform.data.front_contract import load_front_contract_minute_series


OUTRIGHT_PATTERNS = {
    "ES": re.compile(r"^ES[HMUZ]\d{1,2}$"),
    "NQ": re.compile(r"^NQ[HMUZ]\d{1,2}$"),
}


@dataclass
class DataSummary:
    symbol: str
    rows: int
    trade_days: int
    start_date: str
    end_date: str
    source_path: str
    loader: str
    sample_rows: Optional[int] = None
    rows_removed: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_hhmm(value: str) -> int:
    hour, minute = [int(part) for part in value.split(":", 1)]
    return hour * 60 + minute


def _filter_session(frame: pd.DataFrame, session_start: Optional[str], session_end: Optional[str]) -> pd.DataFrame:
    if session_start is None or session_end is None or frame.empty:
        return frame
    minute_of_day = frame["date"].dt.hour * 60 + frame["date"].dt.minute
    mask = (minute_of_day >= _parse_hhmm(session_start)) & (minute_of_day < _parse_hhmm(session_end))
    return frame.loc[mask].copy()


def _normalize_raw_chunk(
    chunk: pd.DataFrame,
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    session_start: Optional[str],
    session_end: Optional[str],
) -> pd.DataFrame:
    pattern = OUTRIGHT_PATTERNS[symbol]
    mask = chunk["symbol"].astype(str).str.match(pattern)
    out = chunk.loc[mask].copy()
    if out.empty:
        return out

    ts = pd.to_datetime(out["ts_event"], utc=True, errors="coerce")
    out["date"] = ts.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    out = out.dropna(subset=["date"]).copy()

    if start_date:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date:
        # Treat the CLI end date as an inclusive calendar day.
        out = out[out["date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    out = _filter_session(out, session_start, session_end)
    if out.empty:
        return out

    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    out = out[(out["high"] >= out["low"]) & (out["open"].between(out["low"], out["high"])) & (out["close"].between(out["low"], out["high"]))].copy()
    if out.empty:
        return out

    # For a sample run, choose the highest-volume outright contract at each minute.
    out = (
        out.sort_values(["date", "volume", "symbol"], ascending=[True, False, True])
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["symbol"] = symbol
    return out[["date", "open", "high", "low", "close", "volume", "symbol"]]


def _load_sample_from_raw(
    path: Path,
    symbol: str,
    sample_rows: int,
    start_date: Optional[str],
    end_date: Optional[str],
    session_start: Optional[str],
    session_end: Optional[str],
) -> pd.DataFrame:
    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    chunks: list[pd.DataFrame] = []
    chunk_size = max(10_000, min(250_000, int(sample_rows) * 8))
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        normalized = _normalize_raw_chunk(chunk, symbol, start_date, end_date, session_start, session_end)
        if normalized.empty:
            continue
        chunks.append(normalized)
        rows = sum(len(part) for part in chunks)
        if rows >= sample_rows:
            break

    if not chunks:
        raise ValueError(f"No {symbol} rows left after filtering {path}")

    out = (
        pd.concat(chunks, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="first")
        .head(sample_rows)
        .reset_index(drop=True)
    )
    return out


def _validate_clean_data(frame: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
    out = out[(out["high"] >= out["low"]) & (out["open"].between(out["low"], out["high"])) & (out["close"].between(out["low"], out["high"]))].copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    out["symbol"] = symbol
    return out[["date", "open", "high", "low", "close", "volume", "symbol"]], before - len(out)


def load_futures_data(
    symbol: str,
    data_path: str | Path | None = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sample_rows: Optional[int] = None,
    session_start: Optional[str] = None,
    session_end: Optional[str] = None,
    back_adjust: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, DataSummary]:
    """Load ES/NQ 1-minute futures data into a clean OHLCV frame.

    Sample runs use a fast local raw read. Full runs use the in-package
    front-contract stitcher with an internal cache under
    ``market_regime_platform/output``.
    """

    symbol = symbol.upper()
    path = resolve_data_path(symbol, data_path)

    if sample_rows is not None:
        data = _load_sample_from_raw(
            path=path,
            symbol=symbol,
            sample_rows=int(sample_rows),
            start_date=start_date,
            end_date=end_date,
            session_start=session_start,
            session_end=session_end,
        )
        data, removed = _validate_clean_data(data, symbol)
        loader = "raw_sample_highest_volume_per_minute"
    else:
        data, front_summary = load_front_contract_minute_series(
            data_path=path,
            root_symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            session_start=session_start,
            session_end=session_end,
            back_adjust=back_adjust,
            use_cache=use_cache,
        )
        data, removed = _validate_clean_data(data, symbol)
        loader = f"front_contract_back_adjusted={bool(back_adjust)} cache={front_summary.cache_path}"

    if data.empty:
        raise ValueError("No data left after preprocessing")

    summary = DataSummary(
        symbol=symbol,
        rows=int(len(data)),
        trade_days=int(data["date"].dt.normalize().nunique()),
        start_date=str(data["date"].min().date()),
        end_date=str(data["date"].max().date()),
        source_path=portable_path(path),
        loader=loader,
        sample_rows=sample_rows,
        rows_removed=int(removed),
    )
    return data, summary
