# Progress log

Running log so a later session can resume. Newest entry first. Scope: `PLAN.md`.

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
