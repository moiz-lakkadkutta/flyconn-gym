# flyconn-gym

An open, controlled brain-body loop. A fly connectome, loaded and simulated with
[flyconn](https://github.com/moiz-lakkadkutta/flyconn), drives a physics-simulated fly
([FlyGym](https://github.com/NeLy-EPFL/flygym) 2.x / NeuroMechFly in MuJoCo). Every run
has **null wirings** beside it, so you can see which behaviour comes from the wiring and
which from hand-written controllers.

> **Status: pre-alpha tracer bullet.** The loop runs end to end on a **synthetic toy
> connectome**. Nothing here is fly biology yet. Real-connectome I/O maps (M1) come next.

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
- **Controller bias:** even with perfectly symmetric drive, the controller turns by about
  +13° in 0.8 s. Read any heading differences against `no_brain` and the seed spread.

Licence: Apache-2.0.
