"""Run the pre-registered spread-v2 design (docs/metrics.md) on one connectome, or summarise.

Usage:
  ``uv run python scripts/spread_v2.py --dataset malecns@1.0``  (one run; needs FLYCONN_CACHE)
  ``uv run python scripts/spread_v2.py --summarize``  (combine finished runs)
Writes ``runs/spread-v2/<dataset>/{episodes.csv,summary.json}``,
``docs/results/spread-v2-<dataset>.md`` and, with ``--summarize``, ``docs/results/spread-v2.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from flyconn_gym.brain import V2_VARIANTS
from flyconn_gym.coupling import COUPLING_LEVEL, DNDecoder, LevelAPolicy
from flyconn_gym.datasets import REAL_DATASETS, SUBGRAPH_K, SUBGRAPH_MIN_WEIGHT, load_looming
from flyconn_gym.env import LoomingWalkEnv
from flyconn_gym.metrics import spread_v2
from flyconn_gym.rollout import replay_drive, run_episode, symmetric

# Registered in docs/metrics.md (spread-v2); do not change.
N_COMPARISONS = 6
CALIBRATION_REFERENCE = 1.0
CALIBRATION_STEPS = 300
DECODER_V2 = DNDecoder(
    tonic=1.0, forward_gain=0.25, turn_gain=0.5, backward_gain=0.5, escape_gain=0.5
)
RUNS = Path("runs/spread-v2")
RESULTS = Path("docs/results")


def run(dataset: str, seeds: int, steps: int) -> None:
    out = RUNS / dataset
    out.mkdir(parents=True, exist_ok=True)
    m, io = load_looming(dataset)
    print(f"{dataset}: {m.n} neurons, {m.weights.nnz} edges", flush=True)
    rows: list[dict[str, Any]] = []
    prov: dict[str, Any] = {}
    t0 = time.perf_counter()
    for side in ("left", "right"):
        env = LoomingWalkEnv(max_steps=steps, loom_side=side)
        for seed in range(seeds):
            for variant in V2_VARIANTS:
                policy = LevelAPolicy.build(m, io, variant, seed=seed)
                policy.decoder = DECODER_V2
                groups = policy.calibrate(CALIBRATION_REFERENCE, CALIBRATION_STEPS)
                ep = run_episode(env, policy, seed=seed)
                rep = replay_drive(env, symmetric(ep["drive"]), seed=seed)
                s = ep["summary"]
                dn_mean = dict(zip(ep["dn_names"], ep["dn"].mean(axis=0), strict=True))
                rows.append(
                    {
                        "variant": variant.value,
                        "seed": seed,
                        "side": side,
                        "dheading_deg": s["final_heading_deg"],
                        "dheading_replay_deg": rep["final_heading_deg"],
                        "min_loom_distance_mm": s["min_loom_distance_mm"],
                        "path_length_mm": s["path_length_mm"],
                        "path_length_replay_mm": rep["path_length_mm"],
                        "hit": s["hit"],
                        "flipped": s["flipped"],
                        "steps": s["steps"],
                        "replay_steps": rep["steps"],
                        **{f"dn_{k}": float(v) for k, v in dn_mean.items()},
                        **{f"scale_{g}": v for g, v in groups.items()},
                    }
                )
                prov.setdefault(variant.value, ep["provenance"])
            print(f"{side} seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)
        env.close()
    wall = time.perf_counter() - t0

    episodes = pd.DataFrame(rows)
    episodes.to_csv(out / "episodes.csv", index=False)
    res = spread_v2(episodes, n_comparisons=N_COMPARISONS)
    v1_style = _v1_style(episodes)
    summary = {
        "metric": "spread-v2",
        "dataset": dataset,
        "coupling_level": COUPLING_LEVEL,
        "iomap": io.label(),
        "n_neurons": m.n,
        "n_edges": int(m.weights.nnz),
        "subgraph": None
        if dataset == "toy"
        else {"k": SUBGRAPH_K, "min_weight": SUBGRAPH_MIN_WEIGHT},
        "seeds": seeds,
        "steps": steps,
        "wall_s": round(wall, 1),
        "bonf_level": res["bonf_level"],
        "replay_check": res["replay_check"],
        "brain_effect": res["brain_effect"].to_dict("records"),
        "wiring_effect": res["wiring_effect"].to_dict("records"),
        "v1_style_E": v1_style.to_dict("records"),
        "provenance_by_variant": prov,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _report(dataset, episodes, summary, res)
    print(f"wrote {out}/ in {wall:.0f}s; replay check {res['replay_check']}")


def _v1_style(episodes: pd.DataFrame) -> pd.DataFrame:
    """Secondary: v1's E (against no_brain) under the v2 decoder, to compare with v1."""
    from flyconn_gym.metrics import spread_v1

    return spread_v1(episodes)["brain_effect"]


def _report(
    dataset: str, episodes: pd.DataFrame, summary: dict[str, Any], res: dict[str, Any]
) -> None:
    cols = ["path_length_mm", "path_length_replay_mm", "min_loom_distance_mm", "hit", "flipped"]
    cols += sorted(c for c in episodes.columns if c.startswith(("dn_", "scale_")))
    secondary = cast("pd.DataFrame", episodes.groupby("variant")[cols].mean())
    unreachable = {
        c.removeprefix("scale_"): int(cast("pd.Series", episodes[c] == 0).sum())
        for c in episodes.columns
        if c.startswith("scale_")
    }
    pct = f"{100 * res['bonf_level']:.2f} %"
    lines = [
        f"# spread-v2 on {dataset}",
        "",
        f"Generated by `scripts/spread_v2.py` on {datetime.now(UTC):%Y-%m-%d}; metric registered",
        "in `docs/metrics.md` (spread-v2). **Level A.** "
        + (
            "Synthetic toy: pipeline check only, not one of the 6 comparisons."
            if dataset == "toy"
            else f"k-hop subgraph (k={SUBGRAPH_K}, min_weight={SUBGRAPH_MIN_WEIGHT}): "
            f"{summary['n_neurons']} neurons, {summary['n_edges']} edges."
        ),
        f"{summary['seeds']} seeds x 2 sides x 4 variants, {summary['steps']} ms each, each "
        f"with a symmetric replay; wall {summary['wall_s']:.0f} s.",
        f"Replay check (no_brain episode vs its replay): {res['replay_check']}.",
        "",
        f"## Wiring effect (paired E_real - E_null, degrees; verdict from the {pct} interval)",
        "",
        res["wiring_effect"].round(3).to_markdown(index=False),
        "",
        "## Brain effect E_v (turn-away minus symmetric replay, degrees)",
        "",
        res["brain_effect"].round(3).to_markdown(index=False),
        "",
        "## Secondary (descriptive)",
        "",
        "v1-style E (against no_brain, v2 decoder):",
        "",
        pd.DataFrame(summary["v1_style_E"]).round(3).to_markdown(index=False),
        "",
        "Per-variant means (readouts normalised; `scale_*` = calibration scale, raw units):",
        "",
        secondary.round(4).to_markdown(),
        "",
        f"Episodes with an unreachable group (scale 0), by group: {unreachable}.",
        "",
    ]
    (RESULTS / f"spread-v2-{dataset}.md").write_text("\n".join(lines))


def summarize() -> None:
    rows: list[dict[str, Any]] = []
    for ds in REAL_DATASETS:
        path = RUNS / ds / "summary.json"
        if not path.exists():
            raise SystemExit(f"missing {path}: run --dataset {ds} first")
        s = json.loads(path.read_text())
        for w in s["wiring_effect"]:
            rows.append(
                {
                    "dataset": ds,
                    "neurons": s["n_neurons"],
                    "null": w["null"],
                    "mean": w["mean"],
                    "bonf_lo": w["bonf_lo"],
                    "bonf_hi": w["bonf_hi"],
                    "ci95_lo": w["ci_lo"],
                    "ci95_hi": w["ci_hi"],
                    "verdict": w["verdict"],
                    "replay_valid": s["replay_check"]["valid"],
                }
            )
    df = pd.DataFrame(rows)
    lines = [
        "# spread-v2 headline (6 registered comparisons)",
        "",
        f"Generated by `scripts/spread_v2.py --summarize` on {datetime.now(UTC):%Y-%m-%d}. "
        "Paired E_real - E_null in degrees; the verdict uses the Bonferroni interval",
        f"(level 1 - 0.05/{N_COMPARISONS} = 99.17 %); 95 % CIs are descriptive. Per-dataset",
        "detail: `spread-v2-<dataset>.md`.",
        "",
        df.round(3).to_markdown(index=False),
        "",
    ]
    (RESULTS / "spread-v2.md").write_text("\n".join(lines))
    print(df.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    elif args.dataset:
        run(args.dataset, args.seeds, args.steps)
    else:
        ap.error("give --dataset or --summarize")


if __name__ == "__main__":
    main()
