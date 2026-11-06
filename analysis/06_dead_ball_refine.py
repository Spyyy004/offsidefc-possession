"""Step 7 refinements, judged on the honest test bench.

  #2  new clues: where the ball was last seen / is next seen (blackout_location_features)
  #1  post-processing settings (smoothing, cut-off, minimum stoppage, filling tiny alive gaps),
      chosen by NESTED leave-one-match-out: the settings used on each test match are picked on the
      other six only, so the reported score is honest.

    python analysis/06_dead_ball_refine.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, score  # noqa: E402
from analysis.features import dead_ball_features, blackout_location_features  # noqa: E402
from analysis.dead_ball import (FROZEN_BALL_ARTEFACTS, leave_one_match_out, baseline_team,  # noqa: E402
                                dead_mask, nearest_codes)
from starter.baseline import smooth  # noqa: E402

GRID = list(itertools.product([0.0, 0.5, 1.0, 2.0],        # smooth_s
                              [0.3, 0.4, 0.5, 0.6],        # threshold
                              [1.0, 2.0, 3.0, 4.0],        # min_dead_s
                              [0.0, 1.0]))                 # fill_alive_s
DEFAULT = (0.0, 0.5, 2.0, 0.0)

halves, _ = bench()
matches = sorted({h["match"] for h in halves})
y = [h["label"] == "D" for h in halves]
teams = [baseline_team(h) for h in halves]
base_feats = [dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in halves]
loc_feats = [blackout_location_features(h) for h in halves]


def labels(p, k, params):
    return np.where(dead_mask(p, *params), "D", teams[k])


def per_match_scores(p):
    """spellF1 of every grid setting on every match: {params: {match: score}}."""
    out = {}
    for params in GRID:
        out[params] = {}
        for m in matches:
            ks = [k for k, h in enumerate(halves) if h["match"] == m]
            out[params][m] = score([halves[k] for k in ks], [labels(p[k], k, params) for k in ks])["spell_f1_mean"]
    return out


def nested(p, table):
    """Each test match uses the setting that did best on the other six."""
    preds, chosen = [None] * len(halves), {}
    for m in matches:
        best = max(GRID, key=lambda g: np.mean([table[g][o] for o in matches if o != m]))
        chosen[m] = best
        for k, h in enumerate(halves):
            if h["match"] == m:
                preds[k] = labels(p[k], k, best)
    return preds, chosen


rows = []
def report(name, preds):
    r = score(halves, preds)
    rows.append((name, r["macro_f1"], r["spell_f1_mean"], r["alive_dead_f1"], r["n_pred_spells"]))

report("baseline_smooth50", [smooth(nearest_codes(h), 50) for h in halves])
for name, X in [("detector", base_feats),
                ("detector + location clues", [pd.concat([a, b], axis=1) for a, b in zip(base_feats, loc_feats)])]:
    p = leave_one_match_out(halves, X, y)
    report(name + " (default settings)", [labels(p[k], k, DEFAULT) for k in range(len(halves))])
    table = per_match_scores(p)
    preds, chosen = nested(p, table)
    report(name + " (tuned, nested)", preds)
    print("{}: settings chosen per test match (smooth s, cut-off, min dead s, fill alive s):".format(name))
    print("   ", sorted(set(chosen.values()), key=lambda g: -list(chosen.values()).count(g)))
report("perfect dead-ball (ceiling)", [np.where(h["label"] == "D", "D", t) for h, t in zip(halves, teams)])

print("\n{:<44} {:>8} {:>8} {:>8} {:>6}".format("on the test bench", "macroF1", "spellF1", "deadF1", "poss"))
for name, mf, sf, df, n in rows:
    print("{:<44} {:>8.4f} {:>8.4f} {:>8.4f} {:>6}".format(name, mf, sf, df, n))
