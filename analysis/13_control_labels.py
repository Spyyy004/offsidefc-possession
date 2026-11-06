"""Step 9d: retrain on control-style labels (D35), realistic bench (1 m jitter).

The degraded bench is identical to before (labels do not affect the degradation), so the Step 9c
dead-ball outputs are reused. The team judge is retrained on converted labels and the referee's
switch cost / team trust re-tuned by nested leave-one-match-out against the converted labels.
Every variant is scored against BOTH label styles.

    python analysis/13_control_labels.py
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER  # noqa: E402
from analysis.features import team_features, turnover_features  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
FPS = 25
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, labels="control")
matches = sorted({h["match"] for h in halves})
by_match = {m: [k for k, h in enumerate(halves) if h["match"] == m] for m in matches}
p_dead = list(np.load(os.path.join(DERIVED, "bench_j1_p_dead.npz"), allow_pickle=True)["p"])
p_old = list(np.load(os.path.join(DERIVED, "bench_j1_p_home_v2.npz"), allow_pickle=True)["p"])
X = [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in halves]

p_new = [None] * len(halves)
for m in matches:
    tr = [k for k, h in enumerate(halves) if h["match"] != m]
    live = [halves[k]["label"] != "D" for k in tr]
    mdl = new_model().fit(np.vstack([X[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
                          np.concatenate([(halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
    for k in by_match[m]:
        p_new[k] = mdl.predict_proba(X[k].to_numpy())[:, 1]
np.savez(os.path.join(DERIVED, "bench_j1_p_home_ctrl.npz"), p=np.array(p_new, dtype=object))
print("team judge retrained on control-style labels", flush=True)

S = REFEREE_SETTINGS


def decode(ph, sw, beta):
    return [viterbi(p_dead[k], ph[k], sw, S["stop"], S["restart_other"], S["restart_same"], beta) for k in range(len(halves))]


grid = list(itertools.product([7, 10, 14, 20, 28], [1.0, 2.0, 3.0]))
table = {}
for g in grid:
    pr = decode(p_new, *g)
    table[g] = {m: score([halves[k] for k in ks], [pr[k] for k in ks])["spell_f1_mean"] for m, ks in by_match.items()}
nested, chosen = [None] * len(halves), {}
for m, ks in by_match.items():
    best = max(grid, key=lambda g: np.mean([table[g][o] for o in matches if o != m]))
    chosen[m] = best
    pr = decode(p_new, *best)
    for k in ks:
        nested[k] = pr[k]
print("referee (switch, beta) chosen per test match:", sorted(chosen.values()), flush=True)


def with_labels(field):
    return [dict(h, label=h[field]) for h in halves]


def breakdown(hs, preds):
    tot, hit, edges = np.zeros(4), np.zeros(4), [0, 2, 5, 10, 1e9]
    for h, p in zip(hs, preds):
        g, q = pm.label_codes(h["label"]), pm.label_codes(p)
        gs, qs = pm.possession_spells(g, [(0, len(g))]), pm.possession_spells(q, [(0, len(q))])
        for (a, b, _), fl in zip(gs, pm.matched_gt_flags(pm.spell_overlaps(gs, qs), len(gs), 0.5)):
            i = np.searchsorted(edges, (b - a) / FPS, side="right") - 1
            tot[i] += 1
            hit[i] += fl
    return " ".join("{}:{:.0%}".format(n, x) for n, x in zip(["<2s", "2-5s", "5-10s", ">10s"], hit / tot))


variants = [("old: trained on DFL labels (D27)", decode(p_old, S["switch"], S["beta"])),
            ("new: trained on control labels, tuned", nested)]
for target, field in [("CONTROL-style labels (new target)", "label"), ("DFL labels as published (reference)", "label_dfl")]:
    hs = with_labels(field)
    print("\nscored against {}".format(target))
    for name, preds in variants:
        r = score(hs, preds)
        print("   {:<40} macroF1 {:.4f}  spellF1 {:.4f}  poss {} / true {}   {}".format(
            name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"], breakdown(hs, preds)))
