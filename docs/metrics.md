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
