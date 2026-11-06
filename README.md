# OffsideFC — More than a Hack 2027, Phase 1: Ball Possession

Frame-level ball possession for **FC Barcelona vs Athletic Club** (LaLiga 2024-25): one label per
tactical-video frame — `H` (home in possession), `A` (away in possession), `D` (dead ball) — from
the provided automatic tracking only. Team **OffsideFC**.

## Reproduce the submission

```bash
python3 -m venv .venv && source .venv/bin/activate     # tested with Python 3.14.3
pip install -r requirements.txt                          # pinned versions

# 1. Provided match data (gated; not in this repository): unpack the organisers' bundle into data/
#    (data/tactical/BAR-ATH_tactical_atd.xml, ..._atd.txt, ...) or point MTAH_DATA_DIR at it.
# 2. External training data (open; downloaded and checksum-verified, ~2.8 GB):
python fetch_external.py
# 3. Build the submission (~1-2 minutes, deterministic):
python make_submission.py                                # -> submissions/OffsideFC.csv
python evaluation/validate_submission.py submissions/OffsideFC.csv
```

Expected output: `submissions/OffsideFC.csv`, 199,319 rows,
SHA-1 `7fe033d5ae8253c0c07ad736b95d0c4d1edb12e9`. The result is byte-identical across runs and
across processor thread counts (checked with `OMP_NUM_THREADS=1` vs 10 threads).

## Method in one screen

```
Layer 1  Football features per frame from tracking only (past and future windows):
           players' speeds and walking share, ball outside the lines, ball blackouts,
           nearest home vs away player (sustained), team width / block / movement,
           close control, receptions, controller changes, shape flips
Layer 2  Two gradient-boosted tree models (scikit-learn HistGradientBoosting):
           P(dead ball) and P(home | ball alive)
Layer 3  Match-flow referee: 4-state hidden Markov model (H, A, dead-after-H, dead-after-A)
           decoded with Viterbi -- switching team is costly, stoppages last seconds
Rule     Goal rule: a ball blackout > 15 s ending on the centre spot is a goal stoppage
```

**Trained only on external data** (competition rules forbid training or tuning on the provided game):
7 Bundesliga matches (IDSSE/DFL, per-frame possession labels) and 3 Metrica sample games (labels
derived from Metrica's events). The clean external tracking is degraded to look like the provided
automatic tracking: ball blackouts linked to stoppages at real-tracking rates, an "un-frozen" ball
during stoppages, player dropouts, 1 m position jitter and wrong-team tracking episodes — all
calibrated from tracking statistics or external data only.

**Validation:** leave-one-match-out over the 10 external matches, settings chosen by nested
leave-one-match-out, every change judged per match with the official scoring code
(`evaluation/possession_metrics.py`). Current system: spellF1 **0.595** on the Metrica games
(organisers' baseline 0.265) and **0.483** on the Bundesliga matches (baseline 0.155), on the
realistic bench with wrong-team episodes.

Full details: `docs/possession_rules.md` (the football rules), `docs/process-log.html` (every step,
decision D1–D48 and result), `docs/GLOSSARY.md`, and the technical document.

## Repository layout

```
make_submission.py     the one command that builds the submission
fetch_external.py      downloads + verifies + parses the external training data
analysis/              features, models, referee, bench, parsers; numbered experiment scripts 01-22
docs/                  process log, possession rules, glossary, expert review brief and feedback
games.py, evaluation/, starter/baseline.py   from the organisers' repository (MIT, see LICENSE-organisers)
data/                  put the provided match data here (git-ignored)
external/              external datasets land here (git-ignored)
submissions/           generated files land here (git-ignored)
```

## Compliance with the competition rules

- No hand labels or manual corrections of the provided game; every label comes from the pipeline.
- No training, fitting, calibration or tuning on the ground truth or the provided video. The public
  5-minute ground truth was used only to check frame alignment and label meaning.
- The provided video was not used as input.
- No match data, derived frame-level data or submission files are in this repository.

## Data credits and licences

- **IDSSE / DFL open data**, CC-BY 4.0: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025).
  *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data.
  figshare DOI 10.6084/m9.figshare.28196177. Data provided with authorisation of the Deutsche
  Fußball Liga (DFL).
- **Metrica Sports open sample data**, github.com/metrica-sports/sample-data — source acknowledged
  as requested; use confirmed by the organisers.
- Files from the organisers' repository (`games.py`, `evaluation/`, `starter/baseline.py`) are MIT
  licensed; see `LICENSE-organisers`. The provided match data is not covered by any licence here.
- Our code: MIT, see `LICENSE`.
