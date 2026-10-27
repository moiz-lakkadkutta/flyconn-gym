# Reading spread-v2 (hand-written; numbers from the generated reports)

Registration: `docs/metrics.md` (spread-v2, committed before the code and before any run).
Generated reports: `spread-v2.md` (headline) and `spread-v2-<dataset>.md`. Each dataset took
about 930 s (4 parallel single-threaded processes). In every dataset the replay check is
exact (no_brain vs its replay: 0.0°).

## Registered headline (Bonferroni 99.17 % intervals, degrees of turn-away)

| dataset | real − rewired | real − sign_shuffled |
|---|---|---|
| malecns@1.0 | **+29.1** (21.7 to 35.2), real > null | **+26.7** (10.0 to 47.0), real > null |
| flywire@783 | **+15.4** (8.4 to 23.2), real > null | **+17.2** (2.7 to 41.3), real > null |
| banc@888 | **−31.2** (−42.5 to −19.9), real < null | −7.8 (−16.4 to 0.4), no evidence |

The toy pipeline check (not counted) gives no evidence against either null (+1.6 and −0.9,
wide intervals), as it should.

Brain effect of the real wiring alone: malecns +30.1° (SD 2.0), flywire +14.2° (SD 2.3),
banc −31.2° (SD 3.1). The real brains are very consistent across seeds. The nulls average
near zero with SD 8–24°.

Against the registered prediction: malecns and flywire real turn-away > 0 ✓. BANC
"little signal" ✗: BANC turns *toward* the loom, strongly.

## What drives it (exploratory, not registered)

- **The scale fix, not the replay, made the difference.** The v1-style E (against no_brain)
  under the v2 decoder is almost identical to the replay-based E (malecns 30.4 vs 30.1). The
  replay removes v1's drift × slowing confound, but in v2 that confound is small next to the
  steering signal. What changed from v1 is normalisation: the real subgraphs' raw turn
  readouts are about 1e-3 (malecns), 1e-4 (BANC) and 1e-3 (flywire) of saturation, and v2
  divides them by a scale of that size. **The v2 result is about the left/right *pattern* of
  a very small signal**, as the registration disclosed. Whether downstream circuits read
  DNa02 that way is a modelling assumption.
- **By side:**
  - malecns: a loom on the left drives turn_right more than turn_left, and a loom on the
    right the reverse. The fly turns away on both sides (+44.7°, +15.6°).
  - flywire: away for left looms (+28.5°), neutral for right looms (−0.2°).
  - BANC: the left DNa02 barely responds to any stimulus (0.000–0.001 normalised). A loom on
    the right drives the right DNa02, which turns the fly right, toward the loom (−72.9°).
- **Not a side-label flip.** In all three datasets LPLC2 and LC4 synapse only onto the
  giant fibre on their own side (ipsilateral fraction 1.00). So `side` uses the same body
  frame for optic-lobe neurons and DNs in each dataset.
- **No direct path.** No LPLC2 or LC4 neuron synapses directly onto DNa02 in any dataset.
  All looming → steering goes through 2–3-hop paths.
- **BANC coverage differs.** BANC's DNa02s get 24 % of their input from inside the subgraph
  (malecns 68–70 %, flywire 56 %). Their total input is about 7k synapses, against about 24k
  in malecns. BANC's LPLC2 → giant-fibre synapses number 473 (malecns 4,862) and are
  lopsided (left 160, right 313). This is consistent with less complete or asymmetric
  optic-lobe reconstruction in BANC, which would make its k=3 subgraph a poorer brain. It
  does not show that BANC flies would steer differently.

## What can and cannot be said

- **Can:** under the registered Level A model, the normalised DNa02 readouts of the real
  malecns and flywire looming subgraphs steer the fly away from a looming object, more than
  degree-preserving and sign-shuffled nulls do. The BANC subgraph steers it toward the loom.
- **Cannot:** that flies use DNa02 to steer away from looms. The steering is executed by a
  hand-written decoder and controller (Level A). The cited papers tie looming escape to the
  giant fibre and take-off, not to DNa02 steering. The rest of the brain is silent. And the
  effect rests on amplifying a raw signal of about 1e-3.
- The dataset disagreement (malecns/flywire vs BANC) is the most informative result here.
  It is a warning that subgraph brains inherit each dataset's reconstruction coverage.

## Suggested next steps (each needs its own registration)

1. Robustness of v2's sign per dataset: the calibration reference amplitude (0.5, 2.0) and a
   held-out seed range (10–29). Same frozen everything else.
2. A coverage-matched comparison: restrict all three datasets to neurons and edges present
   in comparable proportion (for example, require a minimum in-subgraph input fraction for the
   readout DNs). Or report BANC separately as low-coverage.
3. Option B (wider subgraph, measured with v2), after an owner go-ahead.
