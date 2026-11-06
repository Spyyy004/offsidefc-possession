"""Step 8b: the match-flow referee (HMM + Viterbi), judged on the honest test bench.

Uses the out-of-fold Layer 2 probabilities saved by Steps 7 and 8a, so every match is decoded from
probabilities made by models that never saw it. Referee settings are chosen by NESTED
leave-one-match-out (picked on the other six matches only).

    python analysis/08_referee.py
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, score  # noqa: E402
from analysis.dead_ball import dead_mask  # noqa: E402
from analysis.referee import viterbi  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
D25 = dict(smooth_s=0.5, threshold=0.4, min_dead_s=4.0, fill_alive_s=1.0)

# Round 3: round 1 (switch 10-80, stop 10-40, beta 0.5-1) and round 2 (switch 2-10, stop 40-160,
# beta 1-2) both picked switch=10 (best from both sides), but stop and beta kept landing on the
# upper edge, so the range is widened there once more.
GRID = list(itertools.product(
    [7, 10, 14],           # switch: change of possession in open play
    [160, 320, 640],       # stop: play stops
    [5],                   # restart_other
    [5],                   # restart_same
    [2.0, 3.0, 4.0],       # beta: trust in the team model
))
DEFAULT = (20, 20, 2, 5, 1.0)

halves, _ = bench()
matches = sorted({h["match"] for h in halves})
p_dead = list(np.load(os.path.join(DERIVED, "bench_p_dead.npz"), allow_pickle=True)["p"])
p_home = list(np.load(os.path.join(DERIVED, "bench_p_home.npz"), allow_pickle=True)["p"])
by_match = {m: [k for k, h in enumerate(halves) if h["match"] == m] for m in matches}


def decode(params):
    s, st, ro, rs, b = params
    return [viterbi(p_dead[k], p_home[k], s, st, ro, rs, b) for k in range(len(halves))]


t0 = time.time()
table = {}
for i, params in enumerate(GRID):
    preds = decode(params)
    table[params] = {m: score([halves[k] for k in ks], [preds[k] for k in ks])["spell_f1_mean"]
                     for m, ks in by_match.items()}
    if i == 0:
        print("one setting takes {:.0f} s; {} settings".format(time.time() - t0, len(GRID)), flush=True)

nested_preds, chosen = [None] * len(halves), {}
for m, ks in by_match.items():
    best = max(GRID, key=lambda g: np.mean([table[g][o] for o in matches if o != m]))
    chosen[m] = best
    preds = decode(best) if best not in ("cache",) else None
    for k in ks:
        nested_preds[k] = preds[k]

previous = [np.where(dead_mask(p, **D25), "D",
                     np.where(pd.Series(q).rolling(50, center=True, min_periods=1).mean() > 0.5, "H", "A"))
            for p, q in zip(p_dead, p_home)]

print("\nsettings chosen per test match (switch, stop, restart_other, restart_same, beta):")
for m in matches:
    print("   ", m, chosen[m])

print("\n{:<34} {:>8} {:>8} {:>6} {:>7} {:>7}".format("on the test bench", "macroF1", "spellF1", "poss", "rec@.5", "prec@.5"))
for name, preds in [("Step 8a best (2 s averaging)", previous),
                    ("referee, default settings", decode(DEFAULT)),
                    ("referee, tuned (nested)", nested_preds)]:
    r = score(halves, preds)
    s5 = r["spell_f1"][0.5]
    print("{:<34} {:>8.4f} {:>8.4f} {:>6} {:>7.0%} {:>7.0%}".format(
        name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], s5["recall"], s5["precision"]))
print("{:<34} {:>24}".format("true possessions", r["n_gt_spells"]))
print("spellF1 by strictness (tuned):", "  ".join("{}: {:.2f}".format(k, v["f1"]) for k, v in r["spell_f1"].items()))
