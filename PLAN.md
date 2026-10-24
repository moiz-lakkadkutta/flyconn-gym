# flyconn-gym — an open, honest brain-to-body link (plan and handoff)

Written 2026-10-24 at the end of a brainstorming session in the `flyconn` repo
(`/Users/moizp/google-fly`, github.com/moiz-lakkadkutta/flyconn, v0.4.0 on PyPI).
A new session starts here. Read this file, then flyconn's `CLAUDE.md` and `docs/caveats.md`.
This is the owner's **flagship** idea. Build `flyconn-mcp` first.

## What and why

A real fly connectome, simulated by flyconn, drives a physics-simulated fly body
(NeuroMechFly v2 / FlyGym in MuJoCo). The same experiment runs with **null wirings** beside it,
and a public **scoreboard** shows on which behaviours the real wiring actually beats its nulls,
and on which it doesn't.

The shareable artefact is a side-by-side video (real connectome | rewired | sign-shuffled)
plus the scoreboard. The selling point is honesty: say exactly which part of the movement comes
from the connectome and which from hand-written controllers.

Owner goals: (1) learn and build skills (MuJoCo, control/RL, real-time sim engineering),
(2) lots of people using it.

Evidence from the 2026-10-24 research pass (abstracts / press; verify before citing):
- Eon Systems (Mar 2026) claimed a FlyWire LIF brain driving NeuroMechFly with "91% behavioural
  accuracy". There is no paper and the coupling code is not released.
  `eonsystemspbc/fly-brain` (1,005★) contains only the simulation backends. A LessWrong critique
  ("No, we haven't uploaded a fly yet") says the loop reads a few hand-picked descending neurons
  (DNs) into NeuroMechFly's pretrained motor controllers.
- "The digital sphinx" (Brunton, Abe, Hu, Tuthill; eLife reviewed preprint 111516, 2026): a
  C. elegans connectome trained with RL also drives a fly body realistically, so behaviour match
  is not validation. Cowley and others call for random-connectome controls as standard.
- arXiv 2602.17997 uses the whole-brain connectome as an RL policy graph for whole-body
  locomotion. Pugliese et al. 2025 (bioRxiv 10.1101/2025.09.12.675944) found a walking-CPG
  circuit in MANC VNC simulations, and optogenetics confirmed DNb08. BANC (brain + cord, one
  animal) now makes a connectome-native DN → VNC → motor path possible.
- Possibly-related arXiv papers (2609.38665 "How much of fly walking is written in the wiring?";
  2609.39248, which shows null choice across the sensory-motor boundary decides results) are
  UNVERIFIED. Check whether they exist and who wrote them before citing.
- Hobby demand: desktop-fly 1,082★ (HN 383 pts), doomfly 439★. Every game hand-maps sensory
  inputs and DN outputs. The Sept-2026 hobby wave has saturated (late Show HNs 1–23 pts), so a
  rigorous version stands out more than another game.

## Decisions already made

- The owner picked this direction (the honest brain-body link) as one of the projects to plan.

## Defaults (my calls; owner can override)

- **Standalone repo and package `flyconn-gym`** (free on PyPI and GitHub on 2026-10-24),
  depending on `flyconn[sim]` and `flygym`, the same pattern as `flyconn-mcp`. Write an ADR in
  flyconn. If the sensory / DN I/O maps (M1) turn out generally useful, move them upstream into
  flyconn (`flyconn.io`).
- **Licence Apache-2.0.** flygym 2.1.0 and flybody are Apache-2.0 (read 2026-10-24).
  Python **≥3.12**, because flygym 2.1.0 requires `>=3.12,<3.15`.
- **Body: FlyGym first** (it has vision, olfaction, ascending feedback and a pip package).
  flybody (Nature 2025, not on PyPI) can be a second body later.
- **Two named coupling levels, always labelled in outputs and videos:**
  - **Level A, "brain → DN command → FlyGym controller":** what Eon-style demos do. The brain
    sets DN rates, mapped to FlyGym's descending drive for its CPG/turning controller. Honest
    if labelled: the movement pattern comes from the controller, and only the decision comes
    from the connectome.
  - **Level B, "brain → DN → VNC connectome → motor neurons → joints":** connectome-native
    (MANC, or BANC / MaleCNS whole-CNS). This is the research-grade stretch. The open question
    is the MN-to-actuator mapping.
- **Controls on by default**, reusing flyconn: `degree_preserving_rewire`, `shuffle_signs`,
  NT-sign sampling, random-DN readout (same count, random DNs), a "no brain" baseline (constant
  DN drive). The "foreign connectome" (sphinx) control is optional, and needs a worm connectome
  source with licence checked.
- Start with a **rate model or a sensory → DN subnetwork**, not whole-brain LIF in the loop.
  Measured on M4 Pro CPU: the MaleCNS W2 experiment (20 trials × 0.5 s plus controls) took 190 s.
  Whole-brain spiking at the physics step is far from real time. Videos don't need real time.
- Every run records provenance: dataset@version, I/O map version, coupling level, seeds,
  null family, and package versions.

## flyconn API to build on (verified 2026-10-24, flyconn 0.4.0)

- `flyconn.data.store.Store.open(ref)`; datasets in the registry: `malecns@1.0`, `flywire@783`,
  `flywire@630`, `banc@888`, `manc@1.2.1`, `hemibrain@1.2.1`.
- `flyconn.graph.ConnectivityMatrix.from_store`, `sign_table`, `neuron_signs`, `find_paths`.
- `flyconn.sim.LIFNetwork`, `simulate`, `ShiuParams`, `default_params`, `CALIBRATIONS`.
  The engine must be stepped in chunks inside the physics loop. Check whether the current API
  allows that; if not, add a stepping API to flyconn upstream (with tests and an ADR).
- `flyconn.uncertainty.degree_preserving_rewire`, `shuffle_signs`, `sample_signs`.
- `flyconn.compare.match_types` to carry I/O maps across datasets.

## Milestones (one branch each, tests green per commit)

1. **M0 scaffold:** repo, uv, CI; FlyGym installs and runs a walking episode headless on macOS;
   ADR in flyconn.
2. **M1 I/O maps:** named, versioned, cited sensory input sets (sugar/bitter GRNs, looming
   LPLC2/LC4, olfactory ORNs, mechanosensory) and DN readout sets (walk, turn L/R, stop,
   groom, escape such as the giant fibre) per dataset, with cross-dataset matches and a
   literature citation for each. Unit tests on flyconn's synthetic connectome.
3. **M2 Level A loop:** FlyGym observation → sensory encoder → flyconn brain (rate model first)
   → DN rates → FlyGym descending drive, as a Gymnasium env. One behaviour end to end, for
   example looming → stop/escape or odour → turn toward.
4. **M3 controls and scoreboard:** every behaviour runs real vs each null family over N seeds.
   Metrics are pre-registered in the repo before the runs. Output a static scoreboard page and
   side-by-side videos. Negative results get published too.
5. **M4 launch:** README video, HF Space or static page with the scoreboard, a post framed as
   "an open, controlled brain-body loop", and a link in awesome-fly.
6. **M5 stretch, Level B:** DN → MANC/BANC VNC → MN → joints for walking. Compare against the
   Pugliese CPG result (DNb08) as a first validation target.

## Risks

- Mapping simulator observations to identified neurons is a modelling assumption. Document each
  mapping and its citation, and don't call any of it "the fly".
- Speed: spiking whole-brain in the loop is slow. Mitigate with rate models, subnetworks, MPS,
  or chunked stepping.
- The real wiring may not beat its nulls on many behaviours. That is a result, not a failure.
  Publish it.
- flygym 2.x API changed from 1.x (the research notes cited 1.1.0). Read the current docs first.

## Open questions for the owner (ask early, one at a time)

- Which first behaviour makes the best video: looming escape, odour tracking, or sugar → feeding?
- Is Level B (VNC-native motor) a goal for v1, or explicitly later?
- Should the scoreboard accept outside submissions, such as running someone else's claimed loop
  against our nulls?
