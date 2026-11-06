# External ML expert feedback — 6 Nov 2026 (summary)

Received on `docs/expert_review_brief.md`. Summarised faithfully; our responses are in
`docs/process-log.html` (decisions D41+).

## Critique ("where the picture could be misleading")
1. **Primary scoreboard = 3 matches with labels we derived ourselves.** Metrica's per-frame labels are
   our interpretation of its events, and HMM costs, β and stoppage-linked blackouts were selected
   against that interpretation. 0.601 partly measures how well we reproduce our own rule; the
   0.537 → 0.601 jump is the model learning Metrica's label style.
2. **Noise band vs. deltas.** We decide on differences of 0.001–0.024 while quoting a ~0.04 band.
   Report **per-match paired deltas with a sign test or bootstrap CI**, not pooled means. The
   Metrica-in-training gain may itself be indistinguishable from zero.
3. **Synthetic blackouts are independent of game state.** Real blackouts cluster at aerials, duels,
   long balls, scrambles — where possession changes. Bundesliga bench can't teach "blackout = contested
   moment"; its interception results are optimistic (no clustering) and pessimistic (no signal) at once.
   Metrica (organic gaps) is the better bench.
4. **Missing diagnostic — a ceiling split:** (a) decoder with perfect emissions; (b) real emissions with
   the true boundaries fixed and only labels free. Decides frame judge vs. referee. Guess: boundaries
   are the bottleneck (transF1@25 = 0.48).

## Answers to our questions
1. **Duration-aware decoding:** keep GBT emissions; replace decoder with a **hidden semi-Markov model**:
   non-parametric duration log-priors from external labels (especially D: 0.8 s stoppage ≈ impossible,
   12 s common); for H/A a ~1 s minimum duration with absorb-into-neighbour. O(T·D_max).
2. **Interceptions under blackouts:** (i) **collective velocity reversal** — signed x-velocity of each
   team centroid, its derivative, frame where the two teams' velocities cross sign (gives a boundary
   *inside* the blackout); (ii) **convergence attractor** — latent ball = least-squares intersection of
   player velocity rays; nearest home/away to the inferred ball; (iii) **pressing collapse** — defenders
   within 5 m of inferred ball and its 1 s change, with attacking width dropping. Team aggregates →
   robust to jitter.
3. **Label shift:** train on converted (control-style) DFL labels and accept the DFL drop; add "time
   since last touch-flip" as a feature. **Last touch before the ball goes out** belongs in
   post-processing: at each D-run starting with a line crossing, re-assign the final 0.5–1 s of the
   preceding possession to the team nearest at the crossing.
4. **Domain adaptation:** no importance weighting (variance). Do **output-level self-consistency** —
   compare predicted durations, possessions/5 min, dead share, D-run durations on test vs. bench; use
   Smart Tagging agreement as a diagnostic, never a tuning target.
5. **Optimising spellF1:** **minimum-Bayes-risk decoding** (sample ~200 segmentations by
   forward-backward, pick highest expected spellF1 using the official scorer). 80/20 alternative:
   **boundary refinement** — a dedicated boundary classifier on ±1 s windows, search ±25 frames around
   each decoded boundary.
6. **SkillCorner:** skip for this deadline (camera-driven missingness, undocumented possession field);
   at most a held-out transfer check.
7. **Calibration/ensembling:** calibration absorbed by β/switch tuning; bagging 5 seeds ≤ 0.01, a
   stability measure only. Low priority.

## Smaller notes
- 778 vs 871 possessions with recall < precision → slight under-segmentation; HSMM + min-duration is
  the way to lower the switch cost without flicker.
- Public-window alignment score (0.58) hints test labels may be *finer* than Metrica's. Don't react;
  note in the technical doc.
- Check whether slot swaps in our tracking ever **cross team boundaries** (bench never simulates it).
- Byte-reproducibility: pin `OMP_NUM_THREADS` or verify HGB output is thread-count invariant.

## Suggested 11-day plan
1. Freeze current submission as fallback (today).
2. Two oracle experiments (½ day) → frame judge vs. decoder.
3. Switch DFL training labels to control style (1 h).
4. HSMM durations for D + min-duration for H/A (1–2 days), decided on paired per-match deltas.
5. Centroid-velocity-crossing features (1 day).
6. Boundary refinement or MBR decoding, per the oracle split (2 days).
7. Last three days: nothing new — re-run, confirm reproducibility, write the doc incl. per-match variance table.
