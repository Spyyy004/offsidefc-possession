# Expert review brief — Team OffsideFC, More than a Hack 2027 (Phase 1: Ball Possession)

*Prepared 6 Nov 2026 for an external ML review. Deadline for the competition: **17 Nov 2026**.*
*Everything here is aggregate description; no match data is included. Please do not paste this
document's tables into AI services alongside any match data (competition terms).*

---

## 1. The problem in one paragraph

Given one professional football match (FC Barcelona vs Athletic Club, LaLiga 2024-25), output **one
label per video frame** (25 fps, 199,319 frames; 158,857 graded): `H` (home in possession, ball in
play), `A` (away in possession), `D` (dead ball), or `X` (abstain). It is a **sequence segmentation
problem on noisy multi-agent tracking**, scored against a private frame-by-frame ground truth.
Ranking is fully automatic (no jury).

## 2. Inputs available for the test match

| Asset | Content | Quality issues |
|---|---|---|
| **Automatic tracking (ATD)** | x,y,speed per frame for 46 track slots (team known, no identities) + ball x,y | Ball missing on **41%** of graded frames, mostly in long blackouts (median 10.7 s; 85% of missing frames in gaps > 10 s). Slots swap identity (~11 new track segments/min). ~20.5 of 22 players visible. Ball speed field missing 92% of the time. Position wobble ~1 m. |
| Smart Tagging | 638 automatic tactical phases/set pieces with team tags | Automatic, incomplete (e.g. only 5 free kicks, 26 throw-ins tagged; throw-ins have no team) |
| Homography | per-frame pitch↔pixel matrix | not used so far |
| Tactical + broadcast video (13 GB) | — | Not used (broadcast is unsynchronised) |
| **Public ground truth** | first 5 minutes only (17 possessions) | For format/alignment checks only — **may not be used for training, tuning or calibration** |

**How the ground truth was made** (per organisers): mechanically from *corrected* tracking — `D`
where the corrected tracking marks the ball dead, otherwise the team it names as owning the ball.
The written definition is control-based: possession changes only when the opponent gains
**controlled** possession; touches, deflections and won aerial duels without control do not count;
a loose ball stays with the last team in control.

## 3. Hard constraints (organiser rules — they shape every design choice)

- **No hand labels** on the test match, in any amount, for any purpose.
- **No training, fitting, calibration or tuning on the ground truth (incl. the public 5 min) or on the
  provided video.** Any tuned parameter must be tuned on **external data** only.
- No checking our output against the video. No human-produced data about this match (event data etc.).
- No uploading the provided data to AI services.
- Must submit a public repo that regenerates the file **byte-for-byte**, plus a technical document.
- Allowed (confirmed by organisers): any external dataset we have rights to; Metrica's open sample data;
  making external data resemble our tracking using statistics of the provided tracking.

## 4. The metrics

- **spellF1 (ranking).** Possessions = maximal runs of `H` or `A` (`D` runs are stoppages). Predicted
  and true possessions of the same team are matched one-to-one (greedy by temporal IoU); F1 of that
  matching is averaged over IoU thresholds 0.3, 0.4, 0.5, 0.6, 0.7. Each possession counts once
  regardless of length. **`X` is forward-filled with the previous label before spellF1**, so
  abstaining inside a possession is free and abstaining across a change carries the old label over.
  Full-match noise band ≈ 0.04.
- **macroF1 (tie-break).** Unweighted mean of per-frame F1 for H, A, D. `X` counts as wrong.
- Diagnostics reported by the official scorer: accuracy, kappa, alive/dead F1, team accuracy,
  transition F1 within ±12/±25 frames, possession-share error, fragmentation, counts and durations.
- **Published floor:** organisers' baseline (nearest player to ball + ±2 s majority vote) =
  **spellF1 0.2433** on the full match. There is **no leaderboard and no feedback** before results.

Key property: per-frame accuracy and spellF1 disagree strongly. A frame-level classifier with 87% team
accuracy still scored badly because one wrong frame inside a possession splits it into three.

## 5. Our validation setup (the core methodological choice)

Because we cannot tune on the test match, we built an **external test bench**:

| Data | Matches | Labels | Licence |
|---|---|---|---|
| IDSSE / DFL Bundesliga 2022-23 (Bassek et al. 2025, *Scientific Data*) | 7 (~668 min) | per-frame `BallPossession` + `BallStatus` from operator-corrected tracking | CC-BY 4.0 |
| Metrica Sports open sample data | 3 | **we derive** per-frame labels from Metrica's own events (control-based: team of latest controlled action; dead from ball-out/foul/goal to next set piece) | open, use confirmed by organisers |

**Making the clean external data look like our automatic tracking** (only tracking statistics or
external data used — never our ground truth):
1. **Ball blackouts.** Bundesliga ball positions removed so that the ball is missing on ~79% of dead
   frames and ~16% of live frames — the rates measured on Metrica's (realistic) tracking. Gap lengths
   drawn from our match's empirical gap distribution. Result: 44% missing overall (our match: 41%);
   blackout length quartiles 6.3/12.1/22.5 s (ours 6.1/10.7/21.1 s).
2. **Un-freezing the ball.** DFL parks the ball during stoppages (99% of dead frames still; 2% in real
   tracking). Replaced by "nearest player carries it, then it moves to the restart spot".
3. **Player dropouts** to ~20.5 visible players, and **1 m smooth position jitter** (matches our
   tracking's close-control statistics).
4. Metrica's tracking already has realistic ball gaps (37% missing, not frozen): only dropouts + jitter.

**Evaluation protocol:** leave-one-match-out over all 10 matches; any setting is chosen by **nested**
leave-one-match-out (chosen on the other matches only). Every change is scored with the **official
scoring code** (`possession_metrics.evaluate`). Before trusting a feature we check its **unconditional
distribution matches between bench and our match** (no labels needed) — this caught several artefacts.

**Metrica is our primary scoreboard**: same provider as our ground truth, and its label style is close
to the public 5 minutes (14.6–16.7 possessions / 5 min vs 17; median possession 7.1–7.2 s vs 7.6 s;
13–17% under 2 s vs 12%). The Bundesliga labels are touch-based and much finer (22.9 / 5 min, median
3.7 s, 30% under 2 s).

## 6. Our solution (three layers)

```
Layer 1  Features per frame (tracking only; past AND future windows — offline labelling)
Layer 2  Two gradient-boosted tree classifiers (sklearn HistGradientBoosting, 300 iters, lr 0.1)
           - dead-ball model:  P(dead)          trained on all frames
           - team model:       P(home | alive)  trained on live frames
Layer 3  4-state HMM decoded with Viterbi: H, A, D-after-H, D-after-A
           emissions  log(1-p_dead)+β·log(p_home) etc.;  transition costs (log units):
           switch H<->A = 10, stop = 160, restart (either team) = 5, β = 2
Rule     "Goal rule": a ball blackout > 15 s ending on the centre spot is forced dead (goal celebration)
```

**Dead-ball features (≈25):** players' mean speed (now, 1/3/6 s windows, past 3 s, future 3 s), share of
players walking, ball distance outside the lines, ball out in last 3 s, distances to touchline / goal
line / corner / centre spot, in-6-yard-box flag, blackout length / time since / until ball seen, ball
seen share over 10 s, players visible, gap between team x-centroids (kick-offs).
*Dropped as data artefacts:* ball speed and nearest-player distance (DFL frozen ball).

**Team features (≈30 + ≈20 turnover):** nearest home vs away player distance to ball (raw, rolling
medians 0.5/1/3 s, past/future 1/3 s, last-seen and next-seen values bridging blackouts), players of
each team within 5 m, ball position in home attacking direction (direction inferred per half from team
centroids), team width/length (std of y/x), block position and its 1 s movement, team speeds and
difference; **turnover clues**: signed close control (player within 1.5 m), its past/future means,
control change, last/next controller and flip, reception (ball decelerating next to a player), shape
flip (future minus past width and block over 3 s).
Top features by permutation importance: team width difference (3 s), block moving forward, nearest-player
difference over the last 3 s, speed difference.

Final models are trained on all 10 external matches. Pipeline runtime ≈ 1 minute; output is
byte-reproducible.

## 7. Results

### 7a. Progression (spellF1, honest — each match predicted by models that never saw it)

| Step | Bundesliga bench | Metrica |
|---|---|---|
| Organisers' baseline (smooth ±2 s) | 0.206 clean / 0.155 with jitter | 0.265 |
| + dead-ball model | 0.290 (clean) | — |
| + team model (frame judge) | 0.429 (clean) | — |
| + HMM referee | 0.449 (clean) | — |
| + turnover clues | 0.489 clean / **0.451 with jitter** (gain halves under noise) | — |
| trained on Bundesliga only, tested on Metrica (transfer) | — | 0.537 |
| + Metrica in training (10 matches) | 0.441 | 0.577 |
| **+ stoppage-linked blackouts (current)** | 0.491* | **0.601** |

\*Bundesliga bench changed with the new blackouts, so not directly comparable with earlier rows.

### 7b. Full scorecard of the current system (exact; leave-one-match-out over 10 matches)

| Metric | Metrica: baseline | **Metrica: ours** | Bundesliga: ours | Perfect |
|---|---|---|---|---|
| **spellF1** | 0.265 | **0.601** | 0.491 | 1 |
| spellF1 @ IoU 0.3 / 0.5 / 0.7 | .38 / .26 / .14 | **.71 / .62 / .46** | .63 / .50 / .33 | 1 |
| possessions found (recall @0.5) | 0.46 | **0.59** | 0.42 | 1 |
| our possessions correct (precision @0.5) | 0.19 | **0.66** | 0.63 | 1 |
| **macroF1** | 0.679 | **0.785** | 0.856 | 1 |
| F1 H / A / D | .68 / .68 / .67 | **.797 / .802 / .756** | .826 / .821 / .921 | 1 |
| accuracy | 0.679 | **0.784** | 0.867 | 1 |
| kappa | 0.518 | **0.675** | 0.795 | 1 |
| alive/dead F1 | 0.848 | **0.860** | 0.936 | 1 |
| team accuracy (ball alive) | 0.805 | **0.929** | 0.880 | 1 |
| changes caught within 0.5 s / 1 s (transF1@12/@25) | .08 / .15 | **.30 / .48** | .31 / .46 | 1 |
| possession-share error | 0.004 | 0.014 | 0.001 | 0 |
| fragmentation (pred/true segments) | 2.25 | **0.96** | 0.71 | 1 |
| possessions predicted / true | 2151 / 871 | **778 / 871** | 2039 / 3061 | equal |
| mean possession length (s) pred / true | 5.6 / 13.0 | **13.2 / 13.0** | 10.8 / 7.2 | equal |

On the test match the current file has 274 possessions (mean 13.1 s), home 62% of live time,
35% of graded frames dead. Alignment check on the public 5 minutes (format/alignment only, never used
for decisions): spellF1 0.58, macroF1 0.90 vs baseline 0.33 / 0.62 — note these 5 minutes are the
best-tracked of the match (ball visible 88% vs 60% typical).

### 7c. Diagnosis of remaining errors (Bundesliga bench, before Metrica was added)

| True possession length | share | found @IoU 0.5 |
|---|---|---|
| < 2 s | 30% | 9% |
| 2–5 s | 31% | 34% |
| 5–10 s | 19% | 64% |
| > 10 s | 21% | 82% |

- Of missed 5–10 s possessions: 18% are **merged** with a neighbour (in 93% of those, our long
  possession swallowed a real short possession of the other team — a missed interception), 7% wrong
  team, 7% shifted boundaries, 4% called dead, 1% split.
- Of missed short possessions swallowed by neighbours, **39% have the ball completely invisible**
  (vs 24% of short possessions we find).
- Timing: median offset of our changes is +1 frame (no systematic lag).

## 8. What we tried that did NOT help (all measured with nested LOMO)

| Idea | Result |
|---|---|
| Ball "last seen / next seen" location features for dead balls | +0.001 |
| Per-frame "change moment" detector lowering the HMM switch cost locally | 0.488 vs 0.489 |
| Touch-event features (debounced touches, flips, pass streaks) | 0.449 vs 0.450 |
| Bigger trees / more iterations | 0.448 vs 0.450 |
| Shifting labels in time | +0.002 (noise) |
| Training on DFL labels converted to control style (merge sub-3 s opponent spells without 0.4 s control) | +0.008 on control-style labels, −0.027 on DFL labels |
| Abstaining (`X`) inside possessions / on doubtful short pieces | macroF1 −0.02 / +0.0015 spellF1 |
| Restart prior "restart goes to other team" in the HMM | tuned to equal costs (no effect) |

Learning curve (team model, 2 / 4 / 6 training matches): 0.427 / 0.435 / 0.450 — still rising.

## 9. Known limitations / open problems

1. **Short possessions (< 5 s) and interceptions** — the dominant error. Many happen while the ball is
   invisible; the HMM with one global switch cost trades flicker against merging.
2. **Possession-definition mismatch** between training labels (DFL touch-based; Metrica events
   control-based, our derivation) and the test ground truth (corrected-tracking "owner"). Evidence
   suggests the test truth is control-based in open play but last-touch just before the ball goes out.
3. **Domain gap**: synthetic degradation is our best approximation of automatic tracking; we can only
   check unconditional statistics on the test match, never conditional behaviour.
4. **Small external pool**: 10 matches; learning curve not saturated.
5. Unused assets: Smart Tagging (set pieces/kick-offs; effect not measurable on the bench because the
   external matches lack it), homography + video (a pretrained ball detector would be allowed if not
   trained on this video).
6. HMM has no explicit duration modelling (dead balls in reality last 5–20 s; we only have constant
   per-frame costs).

## 10. Questions for the expert

1. **Sequence model:** would a duration-aware model (explicit-duration / semi-Markov HMM, or a CRF /
   temporal CNN / BiLSTM-CRF on top of the frame features) likely beat our HMM with constant switch
   costs on short possessions? How would you train it with only 10 external matches?
2. **Interceptions under ball blackouts:** any modelling trick to infer a brief change of possession
   from player movement alone (pressing collapse, counter-movement) that is robust to tracking noise?
3. **Label-definition shift:** best practice for training on touch-based labels when the test is
   control-based (label smoothing, weak-label models, learning a mapping on Metrica)?
4. **Domain adaptation:** beyond matching unconditional feature distributions, is there a principled way
   (e.g. importance weighting, adversarial feature alignment) to reduce the bench→test gap without
   test labels?
5. **Directly optimising spellF1:** is there a decoding objective closer to IoU-matched possession F1
   than per-frame Viterbi likelihood (e.g. segment-level decoding / minimum-duration constraints)?
6. **Data:** worth adding SkillCorner open data (10 matches, 10 fps broadcast tracking, MIT) given its
   different frame rate and possession definition?
7. **Calibration / ensembling:** would probability calibration of the two classifiers or an ensemble
   materially change the Viterbi output?
8. **Anything in our validation design that you think gives a misleading picture?**

## 11. Reproducibility snapshot

- Python 3.14.3; numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1 (pinned in `requirements-ours.txt`).
- One command: `python make_submission.py` → `submissions/OffsideFC.csv` (≈1 min, deterministic,
  SHA-1 33b8facee8501d242cb0ae279198fd1be6e39ea2).
- Experiment scripts: `analysis/01_…` to `analysis/17_…`; full decision log (D1–D40) in
  `docs/process-log.html`; possession rules in `docs/possession_rules.md`.
