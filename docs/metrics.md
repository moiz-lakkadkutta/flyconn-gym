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
