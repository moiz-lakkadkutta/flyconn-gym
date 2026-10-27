# flyconn-gym

An open, controlled brain-body loop. A fly connectome, loaded and simulated with
[flyconn](https://github.com/moiz-lakkadkutta/flyconn), drives a physics-simulated fly
([FlyGym](https://github.com/NeLy-EPFL/flygym) 2.x / NeuroMechFly in MuJoCo). Every run
has **null wirings** beside it, so you can see which behaviour comes from the wiring and
which from hand-written controllers.

> **Status: pre-alpha.** The loop runs end to end on a synthetic toy connectome and on
> small subgraphs of MaleCNS, BANC and FlyWire. No result here is a biological finding yet.

## Two coupling levels, always labelled

| Level | Path | What the connectome decides | What is hand-written |
|---|---|---|---|
| **A** (this repo today) | senses → input neurons → brain → descending-neuron (DN) readouts → FlyGym's 2-number descending drive | *which way and how hard to walk* | the whole stepping pattern: FlyGym's hybrid CPG + reflex controller, plus a tonic drive |
| **B** (stretch goal) | brain → DN → nerve-cord (VNC) connectome → motor neurons → joints | the movement itself | the motor-neuron → actuator mapping (open question) |

At Level A, legs that look natural tell you nothing about the connectome: the controller
would walk just as well with no brain at all. Only *differences between the real wiring
and its nulls* say anything, and only when they exceed the run-to-run spread.

## Controls, on from day one

`BrainVariant`: `real`, `rewired` (flyconn `degree_preserving_rewire`), `sign_shuffled`
(flyconn `shuffle_signs`), `no_brain` (tonic drive only). Every episode records provenance:
coupling level, variant, seed, dataset, I/O map name and version, brain parameters and package versions.

## Quick start

```bash
uv sync --group dev
scripts/check.sh                                  # ruff, format, pyright, tests (all offline)
uv run python scripts/spike_walk.py 0.5           # headless walk, prints physics steps/s
uv run python scripts/render_toy.py --steps 800   # runs/toy/toy_level_a.gif + provenance JSON
```

```python
from flyconn_gym.brain import BrainVariant
from flyconn_gym.coupling import LevelAPolicy
from flyconn_gym.env import LoomingWalkEnv
from flyconn_gym.rollout import run_episode
from flyconn_gym.toy import toy_connectome, toy_iomap

m = toy_connectome()
policy = LevelAPolicy.build(m, toy_iomap(m), BrainVariant.REWIRED, seed=0)
ep = run_episode(LoomingWalkEnv(max_steps=500), policy)
print(ep["summary"], ep["provenance"]["coupling_level"])
```

## Real-connectome I/O maps (M1)

`flyconn_gym.maps.LOOMING_V1` names the input and readout cell types. `scripts/resolve_maps.py`
resolves it against cached flyconn stores into `data/iomaps/looming@1/<dataset>.json`
(MaleCNS, BANC and FlyWire; see `docs/iomaps.md` for counts). The inputs are LPLC2 + LC4 per optic
lobe. The readouts are DNp09 (forward), MDN (backward; not annotated in flywire@783), DNa02 left/right
(turn) and DNp01 (giant fibre → *slowing* at Level A, since the walking body can't take off).
Citations were checked against the papers by DOI on 2026-10-26 (`docs/iomaps.md`).

Metrics are pre-registered in `docs/metrics.md` before any run that uses them.
`scripts/spread.py` runs spread-v1 (drift-corrected turn-away over seeds × sides × variants).

## Real-connectome brains (M2)

`flyconn_gym.subgraph.load_subgraph` keeps only the neurons on short paths (≤ 3 hops, ≥ 5
synapses per edge) from the looming inputs to the DN readouts: 800–2,700 neurons, built in
under a second, never the whole brain. Weights stay divided by each neuron's whole-dataset
input, and the rest of the brain is silent.

- `scripts/dn_direction.py`: clamping DNa02-left turns the body left and DNa02-right turns
  it right, linearly, on every seed (`docs/results/dn-direction.md`). This matches the
  papers, but at Level A it holds by construction of the decoder.
- `scripts/open_loop.py`: a constant one-eye loom reaches the giant fibre (DNp01) but barely
  the steering DNs (about 1e-3) (`docs/results/open-loop-looming.md`).
- spread-v1 on MaleCNS, BANC and FlyWire subgraphs: no steering effect from the real wiring
  (`docs/results/spread-v1-real.md`). One nominal "real > rewired" (BANC, +0.27°) does not
  survive correction for multiple comparisons and tracks controller drift.
- spread-v2 (`docs/results/spread-v2-reading.md`): it compares each episode with a
  symmetric replay of its own drive and calibrates the readout scale up front. Under v2,
  real MaleCNS and FlyWire subgraphs steer away from the loom more than both nulls do
  (Bonferroni intervals), and BANC steers toward it. The effect rests on normalising a raw
  turn signal of about 1e-3, and BANC's subgraph covers much less of DNa02's input. Read the
  caveats before quoting any of it.

## What v0 is made of (and its caveats)

- **Brain:** a rate model, `tau dr/dt = -r + tanh⁺(gain · W_signed,in-normalised · r + I)`,
  stepped once per 1 ms of control time. Rates are dimensionless, not Hz. flyconn's LIF engine
  can't be stepped in chunks yet (`docs/upstream-notes.md`).
- **Senses:** a *geometric* looming stimulus (a sphere's angular expansion rate, split
  between the left and right eyes by bearing). No rendering, no retina, no identified neurons.
- **Body:** FlyGym 2.1 `HybridTurningController` (from `flygym_demo`); physics at 0.1 ms,
  control every 10 physics steps.
- **Tonic drive:** the rate model is silent without input, so the decoder adds a constant
  forward drive. `no_brain` is exactly that drive alone.
- **Controller drift:** with perfectly symmetric drive, the controller still turns, by an amount
  that depends on the seed (initial CPG phases): mean −1.8°, SD 9.0°, range −13° to +13° over
  0.8 s and 10 seeds (`docs/results/spread-v1-toy.md`). Read heading differences only against
  `no_brain` on the same seed.

Licence: Apache-2.0.
