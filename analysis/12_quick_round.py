"""Quick round on the realistic bench (1 m jitter), referee D27, dead-ball outputs from Step 9c:

  A  timing    : do our possession changes land early/late? Test shifting labels (nested LOMO).
  B  strength  : team judge with more / bigger trees.
  C  learning  : team judge trained on 2, 4, 6 matches -- is the score still rising?

    python analysis/12_quick_round.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER  # noqa: E402
from analysis.features import team_features, turnover_features  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER)
matches = sorted({h["match"] for h in halves})
by_match = {m: [k for k, h in enumerate(halves) if h["match"] == m] for m in matches}
p_dead = list(np.load(os.path.join(DERIVED, "bench_j1_p_dead.npz"), allow_pickle=True)["p"])
p_v2 = list(np.load(os.path.join(DERIVED, "bench_j1_p_home_v2.npz"), allow_pickle=True)["p"])
X = [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in halves]


def decode(ph):
    return [viterbi(p_dead[k], ph[k], **REFEREE_SETTINGS) for k in range(len(halves))]


def lomo(params, n_train=None):
    out = [None] * len(halves)
    for i, m in enumerate(matches):
        others = [o for o in matches if o != m]
        if n_train:   # rotate which matches are used so every size sees varied data
            others = [others[(i + j) % len(others)] for j in range(n_train)]
        tr = [k for k, h in enumerate(halves) if h["match"] in others]
        live = [halves[k]["label"] != "D" for k in tr]
        mdl = HistGradientBoostingClassifier(random_state=0, **params).fit(
            np.vstack([X[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
            np.concatenate([(halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
        for k in by_match[m]:
            out[k] = mdl.predict_proba(X[k].to_numpy())[:, 1]
    return out


def line(name, preds):
    r = score(halves, preds)
    print("   {:<40} macroF1 {:.4f}  spellF1 {:.4f}  poss {}".format(name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"]), flush=True)
    return r


base_preds = decode(p_v2)

# ---- A: timing ------------------------------------------------------------------------------
print("A. TIMING")
lat = []
for h, p in zip(halves, base_preds):
    g, q = pm.label_codes(h["label"]), pm.label_codes(p)
    gt_t, pr_t = pm.transitions(g, [(0, len(g))]), pm.transitions(q, [(0, len(q))])   # (frame, from, to)
    # nearest predicted change within 2 s
    gts = np.array([t[0] for t in gt_t]) if len(gt_t) else np.zeros(0)
    prs = np.array([t[0] for t in pr_t]) if len(pr_t) else np.zeros(0)
    for t in gts:
        if len(prs):
            d = prs - t
            j = np.argmin(np.abs(d))
            if abs(d[j]) <= 50:
                lat.append(d[j])
lat = np.array(lat)
print("   changes matched within 2 s: {}  | our change minus true change, frames: median {:+.0f}, mean {:+.1f}  (+ = we are late)".format(
    len(lat), np.median(lat), lat.mean()))


def shifted(preds, s):
    """Move every label s frames earlier (s > 0) within each half."""
    out = []
    for p in preds:
        if s > 0:
            out.append(np.concatenate([p[s:], np.repeat(p[-1], s)]))
        elif s < 0:
            out.append(np.concatenate([np.repeat(p[0], -s), p[:s]]))
        else:
            out.append(p)
    return out


shifts = [-10, -5, 0, 5, 10, 15, 20]
table = {s: {m: score([halves[k] for k in ks], [shifted(base_preds, s)[k] for k in ks])["spell_f1_mean"]
             for m, ks in by_match.items()} for s in shifts}
nested, chosen = [None] * len(halves), {}
for m, ks in by_match.items():
    best = max(shifts, key=lambda s: np.mean([table[s][o] for o in matches if o != m]))
    chosen[m] = best
    sp = shifted(base_preds, best)
    for k in ks:
        nested[k] = sp[k]
print("   shift chosen per test match (frames earlier):", sorted(chosen.values()))
line("v2 as is", base_preds)
line("v2 + timing shift (nested)", nested)

# ---- B: model strength -----------------------------------------------------------------------
print("B. MODEL STRENGTH (team judge)")
for name, params in [("600 trees, learning rate 0.05", dict(max_iter=600, learning_rate=0.05)),
                     ("600 trees, lr 0.05, bigger trees", dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=63))]:
    line(name, decode(lomo(params)))
print("   (current: 300 trees, learning rate 0.1 = 'v2 as is' above)")

# ---- C: learning curve -----------------------------------------------------------------------
print("C. LEARNING CURVE (team judge trained on n matches)")
for n in (2, 4, 6):
    line("{} training matches".format(n), decode(lomo(dict(max_iter=300, learning_rate=0.1), n_train=n)))
