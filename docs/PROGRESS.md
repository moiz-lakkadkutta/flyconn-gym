# Progress log

Running log so a later session can resume. Newest entry first. Scope: `PLAN.md`.

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
