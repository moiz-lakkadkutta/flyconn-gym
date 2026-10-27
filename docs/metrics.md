# Pre-registered metrics

Committed before any run that uses them. If something changes after results exist, add a
new version below; never edit an old one.

## spread-v1 (2026-10-25): looming turn-away, Level A

**Question.** Under a looming stimulus, does a wired brain turn the fly away from the
stimulus more than the controller does on its own, and more than its null wirings?

**Episode.** `LoomingWalkEnv` defaults: sphere of radius 1 mm starts 12 mm to the fly's left
or right and approaches at 15 mm/s; 800 control steps of 1 ms (0.8 s); physics at 0.1 ms.
Episode seed `s` sets the controller's initial CPG phases. For the null variants, the null
draw (rewire or sign shuffle) also uses seed `s`.

**Design.** Seeds `s = 0..9` × side ∈ {left, right} × variant ∈ {real, rewired,
sign_shuffled, no_brain}. One connectome per run (toy now; real datasets later).

**Primary metric: turn-away angle.** `θ_away = −σ · Δheading`, where Δheading (degrees,
counter-clockwise positive) is the change from reset to episode end, and σ = +1 for a loom on
the left, −1 on the right. Positive means the fly turned away.

**Drift correction.** The controller turns even under symmetric drive. For variant `v` and seed `s`:
`E_v(s) = mean over sides of [θ_away(v, s, side) − θ_away(no_brain, s, side)]`.
Averaging the two sides also cancels any constant drift.

**Comparisons.**
1. Brain effect: `E_v(s)` for v ∈ {real, rewired, sign_shuffled}.
2. Wiring effect: paired `E_real(s) − E_null(s)` for null ∈ {rewired, sign_shuffled}.

Report: mean, SD and 95 % percentile-bootstrap CI over seeds (10,000 resamples, bootstrap
seed 0). A claim ("real beats rewired") needs the 95 % CI of comparison 2 to exclude 0.
Otherwise it is reported as "no evidence". n = 10 seeds of one model, not animals.

**Secondary (descriptive only, no claims).** Minimum fly-sphere distance (mm), path length (mm),
and the fraction of episodes ending in a hit or a flip.

**Expected for the toy connectome:** no evidence. Its I/O map is arbitrary by design, so a
"win" there would point to a pipeline bug, not biology.

## spread-v1 on real-connectome subgraphs (registered 2026-10-26)

Same metric, design and claim rule as spread-v1 above; only the connectome changes. This
section fixes every free parameter before the run. Changing any of them after seeing results
needs a new, separately registered run.

- **Datasets:** malecns@1.0, banc@888, flywire@783, each with its `data/iomaps/looming@1` map.
- **Brain:** `flyconn_gym.subgraph.load_subgraph(k=3, min_weight=5)`. That is every neuron on
  an input-to-readout path of at most 3 hops over edges of at least 5 synapses, plus all I/O
  neurons. Weights are divided by whole-dataset input totals, and the rest of the brain is
  silent. `RateBrain` defaults (tau 20 ms, dt 1 ms, gain 2), `input_gain` 0.5, `DNDecoder`
  defaults. The null variants are drawn on the subgraph.
- **Disclosure:** before writing this section, an open-loop probe (constant input 1.0 to one
  eye for 300 ms, seed 0) showed that on the real subgraphs the turn readouts (DNa02) reach
  about 1e-3, the forward readout (DNp09) about 1e-3 and the escape readout (DNp01) 0.01–0.14.
  The rewired nulls reach about 1e-2 on every readout. The decoder was not tuned after this.
- **Prediction:** no evidence for a wiring effect on turn-away in any dataset, because the
  turn readouts are too weak to move the drive. A brain effect, if any, should come mostly
  through escape → slowing (secondary: path length), not through turning.
- **Extra secondary (descriptive):** the mean of each DN readout over the episode, per variant.

## spread-v2 (registered 2026-10-27): looming turn-away, steering isolated

**Why v2.** On the real subgraphs, v1's one nominal win (BANC) came from the escape readout:
it slowed the fly by different amounts on the left and right, which scaled the controller's
own drift unevenly. And v1 used raw readout rates, whose scale was set by accident (the rate
gain and how much of each DN's input lies inside the subgraph). v2 fixes both before any v2
run. v1 results stay as they are.

**Episode, design, connectomes.** Same as v1 (seeds 0..9 × left/right × variants, 800 ms,
`LoomingWalkEnv` defaults). Connectomes: the registered k=3 / min_weight=5 subgraphs of
malecns@1.0, banc@888 and flywire@783 (`load_subgraph`), plus the toy as a pipeline check
(not counted). Brain: `RateBrain` defaults, `input_gain` 0.5. No wider subgraphs in v2.

**Fix 1: replay baseline (removes drift × slowing).** For each wired episode (variant v,
seed s, side), record the drive `(L_t, R_t)` at every step. Then replay on the same seed with
the symmetric drive `((L_t + R_t)/2, (L_t + R_t)/2)`. The replay has the same speed profile,
including every escape or backward slowing, but no left/right difference. Primary per-episode
quantity: `Δθ = θ_away(episode) − θ_away(replay)`, with θ_away as in v1 (positive = turned
away from the loom). `E_v(s)` = mean of Δθ over the two sides.
`no_brain` is symmetric by construction (Δθ ≡ 0). It is run once per seed and side only to
check that the replay is exact (|Δθ| < 1e-6°; otherwise the run is invalid and reported as
such).

**Fix 2: fixed readout scale.** Before each episode, each brain (variant and seed, so each
null draw is scaled on its own wiring) is calibrated open-loop. Feed input 1.0 to `loom_left`
only for 300 ms, reset, then the same for `loom_right` only (scaled by `input_gain`, as in
the loop). Record each readout's steady rate. Scale groups:
`turn = {turn_left, turn_right}` (one shared scale, so the left/right asymmetry survives),
and `forward`, `backward`, `escape` each on their own. Group scale `S_g` = the maximum rate
of any set in the group over both reference stimuli. In the loop, readout k in group g is
`clip(r_k / S_g, 0, 1)`. If `S_g < 1e-9`, the group is "unreachable": its readouts are 0
and this is reported. Decoder on the normalised readouts: tonic 1.0, forward_gain 0.25,
turn_gain 0.5, backward_gain 0.5, escape_gain 0.5, clip [-1, 1.5]. (By
`docs/results/dn-direction.md`, a normalised turn difference of 1 gives about 49° over
0.3 s.)

**Comparisons.** Brain effect `E_v(s)` for v ∈ {real, rewired, sign_shuffled}. Wiring effect:
paired `E_real(s) − E_null(s)`, null ∈ {rewired, sign_shuffled}.

**Headline and claim rule.** m = 6 wiring comparisons (3 datasets × 2 nulls). The headline is
the Bonferroni percentile-bootstrap CI at level 1 − 0.05/6 (99.17 %; 10,000 resamples,
bootstrap seed 0). A claim ("real > null" on a dataset) needs that interval to exclude 0.
The 95 % CI is reported beside it, descriptively. The toy is not one of the 6.

**Secondary (descriptive).** v1's no_brain-referenced E (to compare with v1), path length,
min distance, hits and flips, mean normalised readouts, the calibration scales `S_g`, and
which groups were unreachable.

**Disclosure.** Before this registration, the committed open-loop probe
(`docs/results/open-loop-looming.md`, seed 0, raw rates) showed small left/right asymmetries
in the real turn readouts. For example, malecns with a loom on the left: turn_left 0.0002,
turn_right 0.0011. After normalisation these become large. The rewired null at seed 0 was
nearly symmetric. So v2 could show real > null in malecns or flywire. If it does, that is a
statement about the *pattern* of a signal that is about 1e-3 of saturation in raw units.
"Only the relative pattern of DN activity matters" is a modelling assumption of v2, not a
finding.

**Prediction.** malecns and flywire: real turn-away > 0, direction uncertain against the
nulls. BANC: little signal (its raw turn readouts round to 0).

## spread-v3 (registered 2026-10-27): robustness of spread-v2

**Why.** The v2 calibration divides each brain's turn readouts by that brain's own maximum,
so an asymmetry of about 1e-3 of saturation becomes full-scale steering. "Real > null" may
therefore only mean "the real subgraph is lateralised and rewiring averages that out", and
the size of about 29° is a product of normalisation. Seeds 0–9 were partly seen before the
v2 registration. v3 tests the v2 claims under four owner-approved checks. Everything not
named here is as in spread-v2: episode, replay baseline (Δθ, E_v(s)), the replay-exactness
check, decoder gains, input gain, `RateBrain` defaults, calibration reference 1.0 for
300 ms, Level A.

**Variants.** real, rewired, sign_shuffled, **lat_rewired** (new, check 3), no_brain. Null
draws for seed s use seed s, as before.

**Check 1: held-out seeds.** Seeds 10–29 were never run or inspected before this
registration. They are the confirmatory set (n = 20 per comparison). Seeds 0–9 are run
only in condition A, below, and reported beside it, labelled exploratory.

**Check 2: scale independence.**
- (a) Shared scale. For each dataset and readout group g, one scale is used for every brain
  (real, all null draws and seeds) in a run: `S_g(d) = max` over the real brain and all null
  draws in that run (seeds 10–29, all three null types) of that brain's own v2 calibration
  scale for g. So no brain's readouts are clipped at the reference stimulus, and no brain is
  amplified relative to another. (The real brain's own scale was rejected as the shared
  scale. Null raw rates are 10–100× the real ones, so they would clip at 1 on both sides
  and be forced symmetric: a bias toward real.) Calibration is open-loop on the brains and
  uses no episode outcomes. A group whose shared scale is below 1e-9 is unreachable, as in v2.
- (b) Descriptive, every condition: per episode, the mean raw (unnormalised) turn asymmetry
  `turn_left − turn_right` and its ratio to the real brain's raw S_turn. Reported per variant
  and side next to the normalised results.

**Check 3: laterality-preserving null (`lat_rewired`).** Edges are split into blocks by
(pre side, post side), with side ∈ {left, right, other/unknown}. Within each block, edge
destinations are permuted and duplicate edges and self-loops are resolved by swaps within
the block. This is flyconn's `degree_preserving_rewire` algorithm applied per block. flyconn
0.4 has no block option, so it is implemented in flyconn-gym and logged in
`docs/upstream-notes.md`. It preserves each edge's source, weight, and (pre side, post side)
relation; each neuron's out-degree into each side; and each neuron's in-degree from each
side. If a block cannot be resolved in 1,000 rounds, it stays unshuffled and the number of
edges affected is reported. **Pre-stated interpretation:** if real > lat_rewired, the v2
effect is more than lateralisation. If not, the report says the v2 effect is explained by,
or not distinguishable from, lateralisation of the subgraph.

**Check 4: coverage matching.** Coverage of a subgraph = the mean, over the neurons in the
`turn_left`, `turn_right` and `escape` readout sets, of (input synapses from subgraph
neurons on simulated edges, i.e. weight ≥ min_weight) / (the neuron's whole-dataset input
total, the same totals used for normalisation). The target is T = the minimum coverage over
the three datasets at the registered k=3, min_weight=5. For each dataset, k stays 3 and
min_weight is the value in {5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50} whose coverage is
closest to T (ties go to the smaller value). The dataset that defines T keeps min_weight 5.
This matches downward only: no subgraph grows beyond its v2 size, so no widening. The
chosen min_weight and achieved coverage are reported before the outcomes.
**Pre-stated interpretation:** if malecns and flywire still turn away at BANC-level coverage
and BANC still turns toward, the dataset disagreement is not explained by readout-input
coverage. If malecns or flywire lose or flip their effect at matched coverage, coverage can
explain BANC's sign.

**Conditions (all seeds 10–29 unless stated).**
- **A:** v2 per-brain scale, v2 subgraphs. Seeds 0–29; 0–9 are exploratory.
- **B:** shared scale (2a), v2 subgraphs.
- **C-v2:** per-brain scale, coverage-matched subgraphs.
- **C-s:** shared scale, coverage-matched subgraphs.
For any dataset whose matched min_weight is 5, its C conditions are the A and B runs; they
are not run or counted twice.

**Comparisons and headline.** Wiring effect: paired E_real − E_null for null ∈ {rewired,
sign_shuffled, lat_rewired}, per dataset and condition, on seeds 10–29. m = 9 (A) + 9 (B)
+ 3 × (number of datasets whose matched min_weight ≠ 5) × 2 (C-v2, C-s). If one dataset
keeps min_weight 5, m = 9 + 9 + 12 = 30. Headline: the Bonferroni percentile-bootstrap
interval at level 1 − 0.05/m, with **100,000** resamples (bootstrap seed 0), because the
tails are extreme. The 95 % CI is descriptive.

**Survival rule for the v2 claims.** The v2 claims are: malecns real > rewired and
real > sign_shuffled; flywire real > rewired and real > sign_shuffled; banc
real < rewired. A claim **survives** only if the same verdict holds at the Bonferroni level
in both A (held-out replication) and B (shared scale). A claim that holds in A but not B is
reported as "normalisation-dependent". The laterality reading uses real vs lat_rewired in B
(and A, descriptively). The coverage reading uses C-v2 and C-s.

**Budget.** About 5.8 s per episode; condition A has 3 datasets × 30 seeds and B and C have
20 seeds each, with 5 variants plus replays: about 7.5 CPU-hours, run as at most 4 parallel
single-threaded processes on the shared laptop.
