"""Step 9: catch quick open-play turnovers, judged on the honest test bench.

  a) team judge v2      = team clues + turnover clues               -> p_home2
  b) change detector    = "is possession changing hands within 0.5 s?" -> p_change
  c) referee v2         = switch cost varies per frame: cheap where p_change is high

All models leave-one-match-out; referee settings by NESTED leave-one-match-out.

    python analysis/09_turnovers.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from analysis.bench import bench, score  # noqa: E402
from analysis.features import team_features, turnover_features  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
FPS = 25

halves, _ = bench()
matches = sorted({h["match"] for h in halves})
by_match = {m: [k for k, h in enumerate(halves) if h["match"] == m] for m in matches}
p_dead = list(np.load(os.path.join(DERIVED, "bench_p_dead.npz"), allow_pickle=True)["p"])
p_home1 = list(np.load(os.path.join(DERIVED, "bench_p_home.npz"), allow_pickle=True)["p"])

T = [team_features(h) for h in halves]
O = [turnover_features(h) for h in halves]
X2 = [pd.concat([t, o], axis=1) for t, o in zip(T, O)]


def change_target(label, window=12):
    """True within +/- window frames of an open-play change H<->A."""
    lab = np.asarray(label)
    flip = np.zeros(len(lab), dtype=bool)
    idx = np.flatnonzero((lab[1:] != lab[:-1]) & (lab[1:] != "D") & (lab[:-1] != "D")) + 1
    for i in idx:
        flip[max(0, i - window):i + window] = True
    return flip


def lomo(X, target_fn, live_only):
    out = [None] * len(halves)
    for m in matches:
        tr = [k for k in range(len(halves)) if halves[k]["match"] != m]
        rows = [(halves[k]["label"] != "D") if live_only else np.ones(len(halves[k]["label"]), bool) for k in tr]
        Xtr = np.vstack([X[k].to_numpy()[r][::TRAIN_EVERY] for k, r in zip(tr, rows)])
        ytr = np.concatenate([target_fn(halves[k]["label"])[r][::TRAIN_EVERY] for k, r in zip(tr, rows)])
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
        for k in by_match[m]:
            out[k] = model.predict_proba(X[k].to_numpy())[:, 1]
        print("  fold", m, "done", flush=True)
    return out


print("team judge v2 ...", flush=True)
p_home2 = lomo(X2, lambda lab: lab == "H", live_only=True)
print("change detector ...", flush=True)
p_change = lomo(X2, change_target, live_only=True)
np.savez(os.path.join(DERIVED, "bench_p_home2.npz"), p=np.array(p_home2, dtype=object))
np.savez(os.path.join(DERIVED, "bench_p_change.npz"), p=np.array(p_change, dtype=object))

S = REFEREE_SETTINGS


def decode(p_home, params, change):
    hi, lo, beta = params
    preds = []
    for k in range(len(halves)):
        sw = hi if not change else hi - (hi - lo) * p_change[k]
        preds.append(viterbi(p_dead[k], p_home[k], sw, S["stop"], S["restart_other"], S["restart_same"], beta))
    return preds


def nested(p_home, grid, change):
    table = {}
    for g in grid:
        preds = decode(p_home, g, change)
        table[g] = {m: score([halves[k] for k in ks], [preds[k] for k in ks])["spell_f1_mean"] for m, ks in by_match.items()}
    out, chosen = [None] * len(halves), {}
    for m, ks in by_match.items():
        best = max(grid, key=lambda g: np.mean([table[g][o] for o in matches if o != m]))
        chosen[m] = best
        preds = decode(p_home, best, change)
        for k in ks:
            out[k] = preds[k]
    return out, chosen


def breakdown(preds):
    tot, hit = np.zeros(4), np.zeros(4)
    edges = [0, 2, 5, 10, 1e9]
    for h, p in zip(halves, preds):
        g, q = pm.label_codes(h["label"]), pm.label_codes(p)
        gs, qs = pm.possession_spells(g, [(0, len(g))]), pm.possession_spells(q, [(0, len(q))])
        flags = pm.matched_gt_flags(pm.spell_overlaps(gs, qs), len(gs), 0.5)
        for (a, b, _), fl in zip(gs, flags):
            i = np.searchsorted(edges, (b - a) / FPS, side="right") - 1
            tot[i] += 1
            hit[i] += fl
    return " ".join("{}:{:.0%}".format(n, x) for n, x in zip(["<2s", "2-5s", "5-10s", ">10s"], hit / tot))


rows = []
const = [(c, c, b) for c, b in itertools.product([7, 10, 14], [1.0, 2.0, 3.0])]
varying = list(itertools.product([20, 40, 80], [1, 3, 5], [1.0, 2.0, 3.0]))

v1 = decode(p_home1, (S["switch"], S["switch"], S["beta"]), False)
rows.append(("Step 8b best", v1))
v2, c2 = nested(p_home2, const, False)
rows.append(("+ turnover clues in team judge", v2))
v3, c3 = nested(p_home2, varying, True)
rows.append(("+ change moments in referee", v3))
print("\nchosen (constant switch, -, beta):", sorted(set(c2.values())))
print("chosen (switch high, switch low, beta):", sorted(set(c3.values())))

print("\n{:<34} {:>8} {:>8} {:>6} {:>7} {:>7}   found by length (overlap 0.5)".format(
    "on the test bench", "macroF1", "spellF1", "poss", "rec@.5", "prec@.5"))
for name, preds in rows:
    r = score(halves, preds)
    s5 = r["spell_f1"][0.5]
    print("{:<34} {:>8.4f} {:>8.4f} {:>6} {:>7.0%} {:>7.0%}   {}".format(
        name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], s5["recall"], s5["precision"], breakdown(preds)))
print("{:<34} {:>24}".format("true possessions", r["n_gt_spells"]))
