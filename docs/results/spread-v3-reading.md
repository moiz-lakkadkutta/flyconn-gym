# Reading spread-v3 (hand-written; numbers from the generated reports)

Registration: `docs/metrics.md` (spread-v3), committed before the code and before any v3
episode. Coverage matching was chosen and committed before any episode too
(`spread-v3-coverage.md`). Generated headline: `spread-v3.md`. **Everything here is Level A**:
a hand-written decoder and controller turn the DN readouts into steering. The rest of the
brain is silent.

The 10 registered runs took 41–78 min each (4 parallel single-threaded processes). In every
run the replay check is exact (no_brain vs its replay: 0.0°), and every `lat_rewired` draw
was fully shuffled (0 unshuffled edges). Confirmatory seeds are 10–29 (n = 20). m = 30, so
every verdict below uses a Bonferroni 99.833 % interval (100,000 bootstrap resamples).

## Which v2 claims survive

Survival rule (registered): the same verdict in A (held-out seeds, v2 per-brain scale) and in
B (one shared scale for all brains).

| v2 claim | A (seeds 10–29) | B (shared scale) | status |
|---|---|---|---|
| malecns real > rewired | +27.4 (+21.4 to +33.0) | −1.0 (−4.9 to +2.6) | normalisation-dependent |
| malecns real > sign_shuffled | +22.9 (+12.0 to +39.4) | **+0.54 (+0.26 to +0.80)** | survives, at under 1° |
| flywire real > rewired | +12.5 (+6.3 to +19.7) | −0.4 (−3.5 to +3.3) | normalisation-dependent |
| flywire real > sign_shuffled | +18.4 (+6.5 to +30.5) | +0.24 (−0.05 to +0.47) | normalisation-dependent |
| banc real < rewired | −30.0 (−38.4 to −21.0) | +0.95 (−3.9 to +6.4) | normalisation-dependent |

In plain terms:

- **All five v2 claims replicate on held-out seeds** under the v2 normalisation (A), with
  similar sizes to v2. In A, BANC is now also real < sign_shuffled (−19.1°), where v2 found no
  evidence.
- **Four of the five do not survive a shared scale.** Under one scale for every brain, the
  real brains barely steer at all: brain effect malecns +0.68°, flywire +0.30°, BANC −0.10°
  (against +30.1°, +14.4° and −31.9° in A). The nulls stay noisy (rewired SD about 5°), so
  there is no evidence for real vs rewired anywhere.
- **The one survivor is real but small.** malecns real > sign_shuffled holds in B, at
  +0.54°. It is a reliable effect under the registered rule, but it is half a degree.
- So the v2 headline (malecns +29°, flywire +15°, BANC −31°) is a property of amplifying
  each brain's own raw turn signal to full scale. The *direction* of that signal is stable:
  malecns and flywire lean away and BANC leans toward, in A and B and on both seed sets. But
  the raw signal is 10–400× smaller than the nulls' raw turn scales, and the steering size
  comes from the normalisation.

Why B squashes the real brains (exploratory): the real brains' raw turn scales are
6e-5 to 1e-3, and the null draws' are 10–400× larger. `lat_rewired` sets the shared scale in
every B and C-s run. Without it, `rewired` alone would still have set a scale about 30×
the real one (malecns 0.032 vs 0.001), so the outcome does not hinge on adding the new null.
This was disclosed in the registration (using the real brain's scale would clip the nulls).

## Laterality-preserving null

Registered reading: if real > lat_rewired in B, the v2 effect is more than lateralisation.

- Real > lat_rewired in **every** condition and dataset, by large margins: B malecns +61.0°,
  flywire +47.6°, BANC +47.0°; A +88.6°, +82.8°, +24.9°.
- So by the registered rule, **the v2 effect is not explained by lateralisation**. The reason
  needs stating plainly, though. The comparison is carried by the null, not by the real brain:
  `lat_rewired` brains turn hard *toward* the loom (brain effect −47° to −68° in every
  condition). In B, the real brains sit near 0°.
- Mechanism (descriptive, from the raw asymmetry): keeping each edge's (pre side, post side)
  relation but randomising targets within the block sends a one-eye loom mostly to the same
  side's DNa02. That turns the fly toward the loom (raw asymmetry ratio ±20–240 × the real
  brain's S_turn, with signs that mean toward). The real subgraphs do *not* do this. Their raw
  turn asymmetry is small and, for malecns and flywire, has the away sign.
- What that supports: the real wiring cancels or reverses the ipsilateral bias that a
  laterality-preserving random graph would have. Even BANC, which turns toward, turns toward
  far less than its `lat_rewired` null (−31.9° vs −56.8° in A). What it does not support: that
  the real wiring produces turn-away steering of any size without the v2 amplification.

## BANC under matched coverage

Coverage matching (registered rule): at k = 3, raise min_weight until the readout DNs'
in-subgraph input fraction is closest to BANC's 0.333. malecns: min_weight 25 (coverage
0.345, 611 neurons). flywire: min_weight 15 (0.323, 543 neurons). BANC stays at
min_weight 5 (803 neurons).

| brain effect E_real (°) | A / v2 subgraphs | C-v2 / matched coverage | B | C-s |
|---|---|---|---|---|
| malecns | +30.1 | **−5.4** (−7.6 to −3.1) | +0.68 | −0.05 |
| flywire | +14.4 | **−29.8** (−31.1 to −28.3) | +0.30 | −0.37 |
| banc | −31.9 | (same as A) | −0.10 | (same as B) |

- At BANC-level coverage, malecns and flywire **flip to turning toward the loom**, like BANC.
  flywire at matched coverage (−29.8°) is close to BANC (−31.9°). Against the nulls in C-v2,
  flywire becomes real < rewired (−23.0°) and real < sign_shuffled (−20.0°), and malecns has
  no evidence either way.
- By the registered reading, **coverage can explain BANC's sign.** The malecns/flywire vs BANC
  disagreement in v2 is what you would expect if the BANC subgraph were simply a sparser
  sample of the same circuit.
- Caveat on the method: matching works by dropping edges below a higher synapse count. That
  changes more than coverage, because it removes all weak edges and shrinks the subgraph to
  540–610 neurons. A coverage-matched subgraph of another kind (for example, sub-sampling
  edges) might behave differently. The registered rule was this one, and only this one was run.
- The bigger point: **the sign of the v2 effect depends on how the subgraph is cut**. At
  min_weight 5, malecns and flywire turn away. At 25 and 15, they turn toward. Under the shared
  scale (C-s) all subgraphs are again near 0° (−0.05°, −0.37°).

## Raw turn asymmetry (check 2b, descriptive)

The mean of raw turn_left − turn_right over each episode, as a ratio to the real brain's raw
S_turn (A, seeds 0–29). A left loom is away if the ratio is negative, a right loom if it is
positive.

- real: malecns −0.37 / +0.13 (both away); flywire −0.25 / +0.01 (left away, right
  neutral); BANC −0.10 / −0.65 (left slightly away, right **toward**). These match the v2
  per-side reading.
- rewired: a same-sign bias on both sides (malecns −2.7 / −2.0), which cancels in turn-away.
  sign_shuffled: about 0.
- lat_rewired: ±25–31 (malecns, flywire) and ±150–240 (BANC), with the toward sign on both
  sides.
- In absolute units, the real asymmetry is about 5e-4 rate units or less: 5e-4 to 2e-2 of
  `lat_rewired`'s, depending on dataset and side.

## Exploratory seeds 0–9 (condition A)

These are the v2 seeds, re-run with the v3 code and the 100,000-resample m = 30 interval.
real − rewired: malecns +29.1, flywire +15.4, BANC −31.2. real − sign_shuffled: +26.7, +17.2,
−7.8. These reproduce v2 to the reported precision, so the v3 code did not change v2's
pipeline. real − lat_rewired: +88.5, +80.9, +21.0.

## What can and cannot be said (Level A)

- **Can:**
  - Under the v2 per-brain normalisation, the v2 results replicate on 20 held-out seeds.
  - The real subgraphs' raw left/right turn signal has a stable direction per dataset.
  - That signal is far smaller and far less ipsilateral than a laterality-preserving random
    graph's.
  - At matched readout coverage, malecns and flywire steer toward the loom, as BANC does.
- **Cannot:**
  - That the real wiring steers the fly away from looms by tens of degrees. Under a shared
    scale the effect is under 1°, and only malecns vs sign_shuffled survives.
  - That malecns/flywire and BANC differ in circuitry. Matched coverage removes the
    difference.
  - Anything about real flies. This is Level A: a hand-written decoder, a silent rest of the
    brain, and an amplified signal of about 1e-3.
- **Overall:** the v2 headline should be read as "the sign of a tiny DNa02 asymmetry in a
  k = 3 subgraph, amplified by construction". That sign depends on the subgraph's edge
  threshold.

## Possible next steps (each needs a registration and the owner's go-ahead)

1. Option B: a wider subgraph, or the whole-brain input to DNa02 (not run; needs the owner).
   v3 suggests the 3-hop cut is the main limit, not the metric.
2. A second coverage-matching method (edge sub-sampling instead of a higher threshold), to
   separate coverage from edge strength.
3. A readout that does not depend on a scale: for example, the sign of the raw asymmetry per
   episode, tested against the nulls' sign distribution.
