"""Render docs/wer_curve.svg from the checked-in metrics.

Standard library only, deterministic: the chart is a build artifact of
results/validation_metrics.csv and results/zero_shot_baselines.json, so it
can never drift from the numbers it plots.

Usage: python3 scripts/make_wer_chart.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Colors chosen to stay legible on both GitHub light and dark backgrounds.
INK = "#768390"
CURVE = "#e2601f"
TINY_REF = "#8b949e"
SMALL_REF = "#539bf5"

W, H = 720, 380
ML, MR, MT, MB = 64, 24, 40, 52  # margins
PW, PH = W - ML - MR, H - MT - MB


def load_points() -> tuple[list[tuple[int, float]], float, float]:
    with open(ROOT / "results" / "validation_metrics.csv") as f:
        rows = list(csv.DictReader(f))
    curve = [(int(r["step"]), float(r["eval_wer"]) * 100) for r in rows]
    with open(ROOT / "results" / "zero_shot_baselines.json") as f:
        base = {b["system"]: b["corpus_wer"] * 100 for b in json.load(f)}
    return curve, base["openai/whisper-tiny"], base["openai/whisper-small"]


def main() -> None:
    curve, tiny, small = load_points()
    xs = [s for s, _ in curve]
    x_min, x_max = 0, max(xs)
    y_min, y_max = 0.0, max(tiny, *(w for _, w in curve)) * 1.12

    def X(v: float) -> float:
        return ML + (v - x_min) / (x_max - x_min) * PW

    def Y(v: float) -> float:
        return MT + PH - (v - y_min) / (y_max - y_min) * PH

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12">'
    )
    parts.append(
        f'<text x="{ML}" y="20" fill="{INK}" font-size="14">'
        "Corpus WER during LoRA adaptation of whisper-tiny (Singaporean English)</text>"
    )

    # gridlines + y labels
    for wer in range(0, int(y_max) + 1, 10):
        y = Y(wer)
        parts.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" '
            f'stroke="{INK}" stroke-opacity="0.18" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ML - 8}" y="{y + 4:.1f}" fill="{INK}" '
            f'text-anchor="end">{wer}%</text>'
        )

    # x labels
    for s in xs:
        parts.append(
            f'<text x="{X(s):.1f}" y="{H - MB + 18}" fill="{INK}" '
            f'text-anchor="middle">{s}</text>'
        )
    parts.append(
        f'<text x="{ML + PW / 2:.1f}" y="{H - 10}" fill="{INK}" text-anchor="middle">'
        "training step (validation split)</text>"
    )

    # zero-shot reference lines (test split)
    for wer, color, label in (
        (tiny, TINY_REF, f"whisper-tiny zero-shot · {tiny:.1f}% (test)"),
        (small, SMALL_REF, f"whisper-small zero-shot · {small:.1f}% (test)"),
    ):
        y = Y(wer)
        parts.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="6 5"/>'
        )
        parts.append(
            f'<text x="{W - MR}" y="{y - 6:.1f}" fill="{color}" '
            f'text-anchor="end">{label}</text>'
        )

    # LoRA curve
    pts = " ".join(f"{X(s):.1f},{Y(w):.1f}" for s, w in curve)
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="{CURVE}" stroke-width="2.5"/>'
    )
    for s, w in curve:
        parts.append(
            f'<circle cx="{X(s):.1f}" cy="{Y(w):.1f}" r="3.5" fill="{CURVE}"/>'
        )
    last_s, last_w = curve[-1]
    parts.append(
        f'<text x="{X(last_s) - 8:.1f}" y="{Y(last_w) - 10:.1f}" fill="{CURVE}" '
        f'text-anchor="end">LoRA validation WER · {last_w:.1f}%</text>'
    )

    parts.append("</svg>")
    out = ROOT / "docs" / "wer_curve.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
