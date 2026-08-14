from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLED_DATA_DIR = PACKAGE_ROOT / "data" / "demo"
OUTPUT_ROOT = PACKAGE_ROOT / "output"
CACHE_ROOT = OUTPUT_ROOT / "_cache"

DEFAULT_DATA_PATHS: dict[str, Path | None] = {
    "ES": None,
    "NQ": BUNDLED_DATA_DIR / "NQ_1M_demo.csv",
}

DEFAULT_BARS_PER_YEAR = 252 * 390
NY_TZ = "America/New_York"


def resolve_data_path(symbol: str, data_path: str | Path | None = None) -> Path:
    root_symbol = symbol.upper()
    if root_symbol not in DEFAULT_DATA_PATHS:
        raise ValueError(f"Unsupported symbol {symbol!r}. Use ES or NQ.")

    if data_path is not None:
        candidate = Path(data_path).expanduser()
    else:
        bundled = DEFAULT_DATA_PATHS[root_symbol]
        if bundled is None:
            raise FileNotFoundError(
                f"No bundled dataset is included for {root_symbol}. "
                "Pass --data-path to use your own raw 1-minute futures CSV."
            )
        candidate = bundled

    if not candidate.exists():
        raise FileNotFoundError(f"Data path not found: {candidate}")
    return candidate


def portable_path(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate

    try:
        relative = resolved.relative_to(PACKAGE_ROOT)
    except ValueError:
        return str(candidate)
    return str(Path("market_regime_platform") / relative)
