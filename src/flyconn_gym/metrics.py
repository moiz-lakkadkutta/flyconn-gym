"""Pre-registered metrics (``docs/metrics.md``).

spread-v1: looming turn-away, drift-corrected against no_brain.
spread-v2: looming turn-away against a symmetric replay of each episode's own drive, with a
Bonferroni headline interval.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd

NULLS = ("rewired", "sign_shuffled", "lat_rewired")
WIRED = ("real", *NULLS)


def turn_away_deg(dheading_deg: float, side: Literal["left", "right"]) -> float:
    """``-sigma * dheading`` with sigma = +1 for a loom on the left; positive = turned away."""
    sigma = 1.0 if side == "left" else -1.0
    return -sigma * dheading_deg


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def _summary(x: np.ndarray, n_boot: int = 10_000) -> dict[str, float]:
    lo, hi = bootstrap_ci(x, n_boot=n_boot)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "ci_lo": lo,
        "ci_hi": hi,
    }


def spread_v1(episodes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute spread-v1 from one row per episode: ``variant, seed, side, dheading_deg``.

    Returns ``per_seed`` (E_v(s)), ``brain_effect`` (E_v summary per wired variant) and
    ``wiring_effect`` (paired E_real - E_null per null, with a verdict).
    """
    df = episodes.copy()
    df["away"] = [turn_away_deg(h, s) for h, s in zip(df["dheading_deg"], df["side"], strict=True)]
    wide = df.pivot_table(index=["seed", "side"], columns="variant", values="away")
    per_seed = cast(
        "pd.DataFrame",
        pd.DataFrame({v: wide[v] - wide["no_brain"] for v in WIRED if v in wide})
        .groupby(level="seed")
        .mean(),
    )
    brain = pd.DataFrame(
        [{"variant": v, **_summary(per_seed[v].to_numpy())} for v in per_seed.columns]
    )
    rows: list[dict[str, object]] = []
    for null in NULLS:
        if null not in per_seed or "real" not in per_seed:
            continue
        s = _summary((per_seed["real"] - per_seed[null]).to_numpy())
        verdict = (
            "real > null" if s["ci_lo"] > 0 else "real < null" if s["ci_hi"] < 0 else "no evidence"
        )
        rows.append({"null": null, **s, "verdict": verdict})
    return {"per_seed": per_seed, "brain_effect": brain, "wiring_effect": pd.DataFrame(rows)}


REPLAY_TOLERANCE_DEG = 1e-6


def _summary_v2(x: np.ndarray, bonf_level: float, n_boot: int) -> dict[str, float]:
    lo, hi = bootstrap_ci(x, n_boot=n_boot, level=bonf_level)
    return {**_summary(x, n_boot), "bonf_lo": lo, "bonf_hi": hi}


def spread_v2(episodes: pd.DataFrame, n_comparisons: int, n_boot: int = 10_000) -> dict[str, Any]:
    """Compute spread-v2 from rows ``variant, seed, side, dheading_deg, dheading_replay_deg``.

    ``Δθ = θ_away(episode) - θ_away(replay)``; ``E_v(s)`` is its mean over sides. Verdicts use
    the Bonferroni interval at level ``1 - 0.05 / n_comparisons``; the 95 % CI is descriptive.
    ``replay_check`` reports whether no_brain episodes match their replays (they must).
    spread-v3 uses the same computation with ``n_boot=100_000`` and its own ``m``.
    """
    bonf_level = 1 - 0.05 / n_comparisons
    df = episodes.copy()
    away = [turn_away_deg(h, s) for h, s in zip(df["dheading_deg"], df["side"], strict=True)]
    away_rep = [
        turn_away_deg(h, s) for h, s in zip(df["dheading_replay_deg"], df["side"], strict=True)
    ]
    df["delta"] = np.array(away) - np.array(away_rep)
    nb = cast("pd.Series", df.loc[df["variant"] == "no_brain", "delta"])
    max_nb = float(nb.abs().max()) if len(nb) else float("nan")
    wide = df.pivot_table(index=["seed", "side"], columns="variant", values="delta")
    per_seed = cast(
        "pd.DataFrame",
        pd.DataFrame({v: wide[v] for v in WIRED if v in wide}).groupby(level="seed").mean(),
    )
    brain = pd.DataFrame(
        [
            {"variant": v, **_summary_v2(per_seed[v].to_numpy(), bonf_level, n_boot)}
            for v in per_seed
        ]
    )
    rows: list[dict[str, object]] = []
    for null in NULLS:
        if null not in per_seed or "real" not in per_seed:
            continue
        s = _summary_v2((per_seed["real"] - per_seed[null]).to_numpy(), bonf_level, n_boot)
        verdict = (
            "real > null"
            if s["bonf_lo"] > 0
            else "real < null"
            if s["bonf_hi"] < 0
            else "no evidence"
        )
        rows.append({"null": null, **s, "verdict": verdict})
    return {
        "per_seed": per_seed,
        "brain_effect": brain,
        "wiring_effect": pd.DataFrame(rows),
        "bonf_level": bonf_level,
        "replay_check": {
            "max_abs_no_brain_deg": max_nb,
            "valid": bool(max_nb < REPLAY_TOLERANCE_DEG),
        },
    }
