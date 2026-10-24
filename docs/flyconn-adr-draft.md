# ADR-00XX (draft for flyconn): brain-body coupling lives in a separate package, flyconn-gym

Status: draft (written in flyconn-gym; to be copied into `flyconn/docs/adr/` by a flyconn session)
Date: 2026-10-24

## Context

The flagship use of flyconn is a closed loop in which a connectome drives a simulated fly
body, run beside null wirings. That needs FlyGym (MuJoCo, about 100 MB of assets, Python
≥3.12 for flygym 2.1.0) and Gymnasium. flyconn supports Python ≥3.11, keeps its core
dependencies light, and runs unit tests offline on a synthetic connectome.

## Decision

- Build the loop as a separate package, **flyconn-gym** (Apache-2.0, Python >=3.12,<3.15),
  depending on `flyconn[sim]>=0.4,<0.5` and `flygym`. The same pattern is planned for flyconn-mcp.
- **Reuse, don't rebuild:** the body and its walking controller come from FlyGym 2.1
  (`flygym_demo.complex_terrain.HybridTurningController`, which takes a `(2,)` descending
  drive). Graphs, signs and nulls come from flyconn (`ConnectivityMatrix`,
  `degree_preserving_rewire`, `shuffle_signs`). flyconn-gym owns only the coupling: I/O maps,
  encoder, decoder, Gymnasium env and the controls harness.
- Coupling level is a first-class label (`A` = brain → DN command → FlyGym controller;
  `B` = brain → DN → VNC connectome → motor neurons) recorded in all provenance and videos.
- v0 brain is a rate model inside flyconn-gym, because flyconn's LIF engine has no stepping
  API. The stepping API is proposed upstream (flyconn-gym `docs/upstream-notes.md` §1) and
  needs its own ADR.

## Consequences

- flyconn stays free of MuJoCo/FlyGym and their Python floor.
- flyconn-gym depends on `flygym_demo`, a demo module whose API may change between flygym
  releases. Pin flygym and wrap the controller behind one adapter (`flyconn_gym.env`).
- If the I/O maps prove general, they move upstream to `flyconn.io` (new ADR then).

## Alternatives considered

- An extra inside flyconn (`flyconn[gym]`): drags Python ≥3.12 and MuJoCo into flyconn's CI matrix.
- flybody as the body (Nature 2025): not on PyPI; possible second body later.
