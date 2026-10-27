# Progress log

Running log so a later session can resume. Newest entry first. Scope: `PLAN.md`.

## 2026-10-27: spread-v3 robustness (branch m4-spread-v3, stacked on m3-spread-v2)

- The owner approved four checks. They were registered in `docs/metrics.md` and committed
  before any code that produces numbers:
  - held-out seeds 10–29;
  - a shared turn scale (B);
  - a laterality-preserving null `lat_rewired` (`nulls.lateral_rewire`, logged in
    `upstream-notes.md`);
  - readout-coverage matching by min_weight (malecns 25, flywire 15, BANC 5, chosen before
    any episode).
- m = 30, Bonferroni 99.833 %, 100k resamples. 10 runs, 41–78 min each, 4 in parallel. All
  replay checks were exact, and all lat draws were fully shuffled.
- Result (`docs/results/spread-v3.md`, reading in `spread-v3-reading.md`):
  - All five v2 claims replicate on held-out seeds under per-brain scaling (A).
  - Under a shared scale (B), the real brains steer < 1°. Only malecns real > sign_shuffled
    survives (+0.54°). The other four are normalisation-dependent.
  - Real > lat_rewired everywhere. This is carried by lat_rewired turning hard toward the
    loom (−47° to −68°), not by the real brains turning away.
  - At BANC-level coverage, malecns (−5.4°) and flywire (−29.8°) turn toward, like BANC
    (−31.9°). Coverage can explain BANC's sign. The v2 sign depends on the edge threshold.
- Next (needs the owner): option B (wider subgraph); a second coverage-matching method; a
  scale-free readout.

## 2026-10-27: spread-v2 (branch m3-spread-v2, stacked on m2-subgraph)

- The owner chose option A (spread-v2 before widening the subgraph). Registered in
  `docs/metrics.md` before the code and before any run:
  - **replay baseline:** each episode's own drive is replayed symmetrically on the same seed,
    so speed is matched and steering is removed;
  - **fixed readout scale:** open-loop calibration per brain, with one shared scale for
    turn_left and turn_right; decoder gains 0.25 / 0.5 / 0.5 / 0.5;
  - **headline:** a Bonferroni 99.17 % interval over m = 6 comparisons.
- Code: `LevelAPolicy.calibrate`, `rollout.replay_drive` / `symmetric`, `metrics.spread_v2`,
  `datasets.load_looming`, `scripts/spread_v2.py`. A physics test checks that a replay
  reproduces its episode exactly; in all runs the replay check is 0.0°.
- Result (`docs/results/spread-v2.md`, reading in `spread-v2-reading.md`):
  - malecns real > both nulls (+29.1°, +26.7°);
  - flywire real > both nulls (+15.4°, +17.2°);
  - BANC real < rewired (−31.2°: it turns *toward* the loom), and no evidence against
    sign_shuffled;
  - toy: no evidence (pipeline check).
- Exploratory diagnostics:
  - the effect comes from the scale fix (the raw turn signal is about 1e-3 of saturation),
    not from the replay;
  - side labels are consistent (LPLC2/LC4 → giant fibre is 100 % ipsilateral everywhere);
  - no direct loom → DNa02 synapses;
  - BANC has low coverage: DNa02 in-subgraph input is 24 % vs 56–70 %, and LPLC2 → GF is 473
    vs 4,862 synapses in malecns, lopsided.
- Not claimed: that flies steer away using DNa02 (Level A, silent rest of brain, amplified
  1e-3 signal).
- Next (each needs a registration or the owner's go-ahead): robustness of the sign
  (reference amplitude, held-out seeds); a coverage-matched comparison; option B (wider
  subgraph) measured with v2.

## 2026-10-26: M2 real-connectome subgraphs, DN direction, verified citations (branch m2-subgraph)

- **Citations** checked by DOI (subagent, web). Fixes: LC4 is von Reyn et al. 2017
  *Neuron*, not Nat Neurosci; DNa02 now cites Rayshubskiy et al. 2025 eLife (the published
  version) plus Yang et al. 2024 Cell (which tests DNa02 and DNg13, not DNa01); DNp09 only
  partly supports "forward" (P9 walking includes ipsilateral turning); the giant fibre cites
  Namiki 2018 eLife (DNp01) + von Reyn 2017. The resolved maps were regenerated: same
  neuron ids, new citation text.
- **Subgraph brains** (`subgraph.py`): k-hop input→readout paths, BFS over the store a
  frontier at a time. With k=3 and min_weight 5: malecns 2,721 neurons / 100,596 edges,
  flywire 1,539 / 29,962, banc 803 / 6,308. Under 1 s to build, at most 1.5 GB RSS.
  Normalisation uses whole-dataset input totals (`input_synapses_total`, or the sum of stored
  edges for flywire).
- **DN direction** (`checks.py`, physics test + `scripts/dn_direction.py`, 10 seeds): clamped
  turn_left=0.1 → +10.0° (SD 1.1), turn_right=0.1 → −10.1° over 0.3 s; linear up to 0.5
  (±49°). A readout of 0.001 gives about 0.1°. Correct per Yang 2024 / Rayshubskiy 2025
  (ipsilateral turning).
- **Open loop** (`scripts/open_loop.py`): real subgraphs drive escape (DNp01) to 0.01–0.14
  but steering DNs only to about 1e-3; the rewired nulls give about 1e-2 on every readout.
  This probe ran before the registration and is disclosed there.
- **spread-v1 on real subgraphs** (registered in `docs/metrics.md` before running; results in
  `docs/results/spread-v1-real.md`): 5 of 6 wiring comparisons give no evidence. BANC
  real − rewired = +0.27° (CI 0.02 to 0.51) passes the registered rule but not Bonferroni
  (99.17 % CI −0.06 to 0.60), and correlates with per-seed drift (r = 0.64). The likely
  cause is the uneven left/right escape slowing interacting with drift, not steering.
- Next (each needs owner input or a new registration):
  - spread-v2: remove the drift × slowing confound and fix the readout scale up front;
  - larger k or a lower min_weight to see whether more of the steering pathway is reached
    (memory grows fast past k=3);
  - LIF stepping API upstream (owner's batched flyconn milestone).

## 2026-10-25: CI green, M1 I/O maps, spread-v1 on the toy (branches ci-green, m1-iomaps)

- **CI (PR #1, ci-green):** run 40176798299 passed on ubuntu-24.04 and macos-latest × py3.12/3.13
  (17 passed on each). MuJoCo/FlyGym install and run headless on the Linux runner, so the physics
  tests run there. `FLYCONN_GYM_SKIP_PHYSICS=1` skips them if a runner ever can't.
- **M1 maps:** `LOOMING_V1` spec (LPLC2+LC4 per lobe → DNp09 forward, MDN backward
  (optional), DNa02 L/R turn, DNp01 escape) is resolved from cached flyconn stores (neuron
  tables only) into `data/iomaps/looming@1/{malecns@1.0,banc@888,flywire@783}.json` with
  store provenance; counts are in `docs/iomaps.md`. Loom-input counts:

  | dataset | loom_left | loom_right |
  |---|---|---|
  | malecns@1.0 | 165 | 146 |
  | banc@888 | 135 | 153 |
  | flywire@783 | 162 | 152 |

  The DN sets have 1–4 neurons each. flywire@783 has no MDN annotation in any column (MaleCNS's
  `fafb_783_cell_type` says "MDN"), so `backward` is optional and missing there.
  manc@1.2.1 can't host looming (no optic lobe). Citations are marked UNVERIFIED (from memory).
- **spread-v1** (pre-registered in `docs/metrics.md`, committed before the run; 10 seeds ×
  2 sides × 4 variants, 401 s wall):
  - controller drift (no_brain) is seed-dependent: mean −1.75°, SD 8.95°, range −13.4 to +12.8°.
    The "+12.8° drift" in the 2026-10-24 entry was seed 0 only.
  - Brain effect E_v, drift-corrected turn-away:

    | variant | E_v | 95 % CI |
    |---|---|---|
    | real | +1.26° | 0.57 to 1.92 |
    | rewired | −0.86° | −2.92 to 1.24 |
    | sign_shuffled | +1.25° | −0.67 to 3.02 |

  - Wiring effect: real − rewired +2.11° (CI −0.03 to 4.39), real − sign_shuffled +0.01°
    (−1.52 to 1.75). Both are **no evidence**, as pre-registered for an arbitrary toy map.
- Next:
  - run the loop on a real-connectome subnetwork: a k-hop subgraph between the loom inputs and the
    DN readouts, so the rate model stays small (no whole-brain runs on the shared laptop);
  - check the citations;
  - a dedicated DN-direction check (does DNa02-left actually turn the body left at Level A
    through the decoder?);
  - the LIF stepping API is going into a batched flyconn milestone (owner's call).

## 2026-10-24: M0 + Level A tracer bullet (branch m0-tracer)

- Scaffold: uv, Python 3.12 (`>=3.12,<3.15`), src layout, Apache-2.0, ruff, pyright strict
  (same relaxations as flyconn), pytest with hypothesis, `scripts/check.sh`, and a CI file
  (`.github/workflows/ci.yml`; not pushed, no remote exists).
  Resolved: flyconn 0.4.0 (PyPI), flygym 2.1.0, mujoco 3.9.0, gymnasium 1.4.0, torch 2.14.1.
- **flygym 2.x findings:** the walking controllers moved out of `flygym` into the bundled
  `flygym_demo.complex_terrain` module. `HybridTurningController.step(descending_signal (2,), obs)`
  sets CPG amplitude per side and reverses frequency for negative drive. Measured with
  `scripts/spike_walk.py`, 0.5 s:
  - (1, 1) walks 6.0 mm forward;
  - (1.2, 0.4) turns right (Δy −2.9 mm);
  - (0.4, 1.2) turns left (Δy +4.0 mm);
  - (0, 0) stands still.
  Convention: left drive > right drive turns right.
- **Speed (M4 Pro CPU, headless):** 1,444 physics steps/s at a 0.1 ms timestep, 0.14× real time
  (`spike_walk.py`). In the full loop with rendering (rate brain on the 120-neuron toy graph
  plus a 320×240 camera): 1,294–1,355 steps/s, so the toy brain costs almost nothing.
- **Stepping API:** flyconn's `simulate()` can't advance in chunks with persistent state
  (details and proposed API in `docs/upstream-notes.md`). v0 uses `RateBrain`, a rate model
  over flyconn's signed, input-normalised `ConnectivityMatrix`.
- **Loop:** `LoomingWalkEnv` (Gymnasium; action = FlyGym descending drive) + `LevelAPolicy`
  (geometric looming → toy sensory sets → `RateBrain` → toy DN sets forward / turn_left /
  turn_right → `DNDecoder` with tonic drive). Controls: `BrainVariant` real / rewired /
  sign_shuffled / no_brain. A unit test shows the three wired variants give different DN
  traces under the same left-looming input; `no_brain` gives constant drive.
- **Toy run** (`scripts/render_toy.py --steps 800`, seed 0, loom from the left; output in
  `runs/toy/`, gitignored). Final heading change:

  | variant | heading change |
  |---|---|
  | real | −9.9° |
  | rewired | +12.4° |
  | sign_shuffled | +5.8° |
  | no_brain | +12.8° |

  The no_brain number is the controller's own drift. One seed on a synthetic graph: it shows
  the pipeline works, not any result.
- Not done: no real-connectome I/O maps; no multi-seed runs or metrics; no retina-based
  vision; ADR only as a draft (`docs/flyconn-adr-draft.md`).
- Next:
  - M1 I/O maps (LPLC2/LC4 → giant fibre / DNp01 first; MaleCNS + BANC);
  - multi-seed spread, which needs controller seeding (the CPG has its own RNG);
  - upstream LIF stepping API in flyconn.
