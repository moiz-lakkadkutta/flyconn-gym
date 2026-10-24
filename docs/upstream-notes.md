# Upstream needs in flyconn

Findings from flyconn-gym that should become changes in flyconn. They are not made here:
the flyconn repo is edited only from its own sessions, with tests and an ADR.

## 1. A stepping API for the LIF engine (needed for a spiking brain in the loop)

Checked against flyconn 0.4.0 (`flyconn/sim/engine.py`), 2026-10-24.

- `simulate(net, *, stimulate, input_events, n_steps, n_trials, seed, ...)` builds a fresh
  `_Stepper` (v = v_rest, g = 0, refractory = 0) on every call and returns only spike
  counts and events (`SimResult`). There is no way to pass state in or get it out.
- `_Stepper` (per-trial v, g and refractory counters, plus a `.tick()`) is private, and the
  delayed-synapse ring (`pending` / `contrib`, `delay_steps` deep) lives in local variables
  of `simulate`, so even `_Stepper` alone cannot resume a run correctly.
- Inputs are fixed for the whole call (Poisson rates in `stimulate`, or a precomputed
  `input_events` list), so a closed loop can't change the drive from body observations
  between ticks.

**Proposed upstream API** (sketch):

```python
sess = LIFSession(net, n_trials=1, seed=0, dtype="float32", device="cpu")
counts = sess.advance(n_steps=10, rates={nid: hz, ...})  # keeps v, g, refractory, delay ring
sess.state() / sess.load_state(s)                         # checkpoints, for reproducibility
```

`simulate()` would become a thin wrapper (one session, one `advance`), so the existing
golden tests pin the new path spike for spike. Determinism: the per-chunk Poisson draw has
to use a seed stream keyed by (trial, absolute step) so results don't depend on how the
run is split into chunks.

Until this exists, flyconn-gym v0 uses its own rate model (`flyconn_gym.brain.RateBrain`)
over flyconn's signed `ConnectivityMatrix`.

## 2. Candidates to move upstream later

- `IOMap` (named, versioned, cited input and readout sets) → `flyconn.io`, once the M1
  real-dataset maps prove generally useful.
