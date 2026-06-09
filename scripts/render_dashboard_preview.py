from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "regime_strategy_backtester" / "output" / "demo_submission"
OUTPUT_PATH = ROOT / "docs" / "dashboard_preview.png"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_num(value: float) -> str:
    return f"{value:.2f}"


def _display_strategy(name: str) -> str:
    parts = name.replace("__", " / ").replace("_", " ").split(" / ")
    return " / ".join(part.title() for part in parts)


def _draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, value: str, note: str, accent: str) -> None:
    draw.rounded_rectangle(box, radius=10, fill="#111827", outline="#243044", width=1)
    x0, y0, x1, _ = box
    draw.rectangle((x0, y0, x1, y0 + 4), fill=accent)
    draw.text((x0 + 18, y0 + 18), title.upper(), fill="#8fa3bf", font=_font(15, bold=True))
    value_font_size = 24 if len(value) > 16 else 30
    draw.text((x0 + 18, y0 + 46), value, fill="#f8fafc", font=_font(value_font_size, bold=True))
    draw.text((x0 + 18, y0 + 86), note, fill="#a8b3c7", font=_font(15))


def _draw_equity_chart(draw: ImageDraw.ImageDraw, equity: pd.DataFrame, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill="#0f172a", outline="#243044", width=1)
    draw.text((x0 + 22, y0 + 18), "Champion Equity Curve", fill="#f8fafc", font=_font(22, bold=True))

    plot = (x0 + 50, y0 + 62, x1 - 28, y1 - 38)
    px0, py0, px1, py1 = plot
    for idx in range(5):
        y = py0 + idx * (py1 - py0) / 4
        draw.line((px0, y, px1, y), fill="#1e293b", width=1)

    values = equity["equity"].astype(float).to_numpy()
    if len(values) > 500:
        step = max(1, len(values) // 500)
        values = values[::step]
    vmin = float(values.min())
    vmax = float(values.max())
    scale = vmax - vmin or 1.0
    points = []
    for idx, value in enumerate(values):
        x = px0 + idx * (px1 - px0) / max(1, len(values) - 1)
        y = py1 - (float(value) - vmin) / scale * (py1 - py0)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill="#4de3d2", width=4, joint="curve")
    draw.text((px0, py1 + 8), f"${vmin:,.0f}", fill="#8fa3bf", font=_font(13))
    draw.text((px1 - 90, py0 - 22), f"${vmax:,.0f}", fill="#8fa3bf", font=_font(13))


def _draw_strategy_table(draw: ImageDraw.ImageDraw, metrics: pd.DataFrame, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, _ = box
    draw.rounded_rectangle(box, radius=10, fill="#111827", outline="#243044", width=1)
    draw.text((x0 + 20, y0 + 18), "Top Strategies", fill="#f8fafc", font=_font(22, bold=True))
    headers = [("Strategy", 20), ("Expectancy", 315), ("Sharpe", 445)]
    for label, offset in headers:
        draw.text((x0 + offset, y0 + 58), label.upper(), fill="#8fa3bf", font=_font(13, bold=True))
    top = metrics.sort_values(["expectancy", "sharpe"], ascending=False).head(5)
    for row_idx, row in enumerate(top.itertuples(index=False), start=0):
        y = y0 + 88 + row_idx * 34
        fill = "#172033" if row_idx % 2 == 0 else "#111827"
        draw.rectangle((x0 + 14, y - 6, x1 - 14, y + 24), fill=fill)
        draw.text((x0 + 20, y), _display_strategy(str(row.strategy)), fill="#e5eefb", font=_font(14))
        draw.text((x0 + 315, y), _format_num(float(row.expectancy)), fill="#4de3d2", font=_font(14, bold=True))
        draw.text((x0 + 445, y), _format_num(float(row.sharpe)), fill="#f3b34d", font=_font(14, bold=True))


def render_preview() -> None:
    metrics = pd.read_csv(RUN_DIR / "metrics.csv")
    equity = pd.read_csv(RUN_DIR / "equity_curves.csv")
    champion_name = str(metrics.sort_values(["expectancy", "sharpe"], ascending=False).iloc[0]["strategy"])
    champion = metrics[metrics["strategy"] == champion_name].iloc[0]
    champion_equity = equity[equity["strategy"] == champion_name].copy()

    image = Image.new("RGB", (1280, 720), "#05070b")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 720), fill="#05070b")
    draw.ellipse((-180, -220, 420, 380), fill="#0e2d34")
    draw.ellipse((930, -160, 1450, 340), fill="#2b1f0e")
    draw.rectangle((0, 0, 1280, 720), outline="#0f172a", width=12)

    draw.text((56, 42), "Market Regime Strategy Backtester", fill="#f8fafc", font=_font(38, bold=True))
    draw.text((58, 92), "Synthetic NQ demo - regime detection, vectorized backtests, and Monte Carlo stress testing", fill="#a8b3c7", font=_font(18))
    for idx, label in enumerate(["Regime-aware signals", "Backtest metrics", "Stress dashboard"]):
        x = 58 + idx * 205
        draw.rounded_rectangle((x, 126, x + 180, 156), radius=15, fill="#122033", outline="#25435d")
        draw.text((x + 14, 133), label, fill="#8fd9ff", font=_font(13, bold=True))

    _draw_card(draw, (58, 186, 305, 318), "Champion", _display_strategy(champion_name), "Ranked by expectancy", "#4de3d2")
    _draw_card(draw, (326, 186, 573, 318), "Total Return", _format_pct(float(champion["total_return"])), "Synthetic demo run", "#8fd9ff")
    _draw_card(draw, (594, 186, 841, 318), "Sharpe", _format_num(float(champion["sharpe"])), "Risk-adjusted score", "#f3b34d")
    _draw_card(draw, (862, 186, 1109, 318), "Max Drawdown", _format_pct(float(champion["max_drawdown"])), "Peak-to-trough", "#ff6f8f")

    _draw_equity_chart(draw, champion_equity, (58, 350, 708, 654))
    _draw_strategy_table(draw, metrics, (742, 350, 1198, 654))

    draw.text((58, 676), "Preview generated from demo artifacts. Run the Streamlit app for the interactive dashboard.", fill="#7287a6", font=_font(14))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    render_preview()
