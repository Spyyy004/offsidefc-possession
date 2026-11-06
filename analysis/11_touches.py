"""Step 9c: touch-event clues for interceptions, on the REALISTIC bench (1 m jitter, D31).

Fair comparison: dead-ball detector, team judge v2 (team + turnover clues) and v3 (+ touch clues)
are all retrained leave-one-match-out on the same jittered bench, decoded with the D27 referee.
Also re-runs the 5-10 s diagnosis to see whether 'merged' failures drop.

    python analysis/11_touches.py
"""
import os
import sys

import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER  # noqa: E402
from analysis.features import dead_ball_features, team_features, turnover_features, touch_features  # noqa: E402
from analysis.dead_ball import FROZEN_BALL_ARTEFACTS, leave_one_match_out  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY, new_model  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
FPS = 25

halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER)
matches = sorted({h["match"] for h in halves})

cache = os.path.join(DERIVED, "bench_j1_p_dead.npz")
if os.path.exists(cache):
    p_dead = list(np.load(cache, allow_pickle=True)["p"])
else:
    p_dead = leave_one_match_out(halves, [dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in halves],
                                 [h["label"] == "D" for h in halves])
    np.savez(cache, p=np.array(p_dead, dtype=object))
print("dead-ball detector ready", flush=True)


def lomo_team(X):
    out = [None] * len(halves)
    for m in matches:
        tr = [k for k, h in enumerate(halves) if h["match"] != m]
        live = [halves[k]["label"] != "D" for k in tr]
        mdl = new_model().fit(np.vstack([X[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
                              np.concatenate([(halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
        for k, h in enumerate(halves):
            if h["match"] == m:
                out[k] = mdl.predict_proba(X[k].to_numpy())[:, 1]
    return out


def diagnose(preds):
    """Share of real 5-10 s possessions found / merged (ours swallowed an opponent possession)."""
    n = found = merged = 0
    for h, p in zip(halves, preds):
        g, q = pm.label_codes(h["label"]), pm.label_codes(p)
        gs, qs = pm.possession_spells(g, [(0, len(g))]), pm.possession_spells(q, [(0, len(q))])
        flags = pm.matched_gt_flags(pm.spell_overlaps(gs, qs), len(gs), 0.5)
        for (a, b, k), fl in zip(gs, flags):
            if not 5 * FPS <= b - a < 10 * FPS:
                continue
            n += 1
            found += fl
            if not fl:
                ov = [(pa, pb) for pa, pb, pk in qs if pk == k and pb > a and pa < b]
                if ov:
                    pa, pb = max(ov, key=lambda s: min(s[1], b) - max(s[0], a))
                    if (min(pb, b) - max(pa, a)) >= 0.8 * (b - a) and (pb - pa) > 1.6 * (b - a):
                        merged += 1
    return found / n, merged / n


T = [team_features(h) for h in halves]
O = [turnover_features(h) for h in halves]
C = [touch_features(h) for h in halves]
variants = {
    "v2: team + turnover clues": [pd.concat([t, o], axis=1) for t, o in zip(T, O)],
    "v3: + touch-event clues": [pd.concat([t, o, c], axis=1) for t, o, c in zip(T, O, C)],
}
print("\n{:<30} {:>8} {:>8} {:>6} {:>7} {:>7}  {:>9} {:>8}".format(
    "realistic bench (1 m jitter)", "macroF1", "spellF1", "poss", "rec@.5", "prec@.5", "5-10s fnd", "merged"))
for name, X in variants.items():
    cached = os.path.join(DERIVED, "bench_j1_p_home_{}.npz".format(name[:2]))
    # v2 does not depend on touch clues, so its saved result is reused; v3 is always retrained
    ph = list(np.load(cached, allow_pickle=True)["p"]) if name.startswith("v2") and os.path.exists(cached) else lomo_team(X)
    preds = [viterbi(p_dead[k], ph[k], **REFEREE_SETTINGS) for k in range(len(halves))]
    r = score(halves, preds)
    s5 = r["spell_f1"][0.5]
    f, mg = diagnose(preds)
    print("{:<30} {:>8.4f} {:>8.4f} {:>6} {:>7.0%} {:>7.0%}  {:>9.0%} {:>8.0%}".format(
        name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], s5["recall"], s5["precision"], f, mg), flush=True)
    np.savez(os.path.join(DERIVED, "bench_j1_p_home_{}.npz".format(name[:2])), p=np.array(ph, dtype=object))
print("{:<30} {:>24}".format("true possessions", r["n_gt_spells"]))
