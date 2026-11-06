"""Phase 2 #1: robustness to wrong-team tracking (D45).

Both versions are SCORED on the bench WITH wrong-team episodes (what our match looks like):
  A  models trained on the bench without episodes (current)
  B  models trained on the bench with episodes
Leave-one-match-out over 10 matches; paired per-match comparison (D41).

    python analysis/19_wrong_team.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, BENCH_JITTER  # noqa: E402
from analysis.features import dead_ball_features, team_features, turnover_features  # noqa: E402
from analysis.dead_ball import FROZEN_BALL_ARTEFACTS, TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
profile = our_mess_profile()
clean, _ = bench(profile, jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True)
mixed, _ = bench(profile, jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in mixed})


def feats(hs):
    return ([dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in hs],
            [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in hs])


XDc, XTc = feats(clean)
XDm, XTm = feats(mixed)
print("features ready", flush=True)


def lomo(train_halves, XD, XT):
    """Train per fold on train_halves' features; predict the held-out match on the MIXED bench."""
    out = [None] * len(mixed)
    for m in matches:
        tr = [k for k, h in enumerate(train_halves) if h["match"] != m]
        dm = new_model().fit(np.vstack([XD[k].to_numpy()[::TRAIN_EVERY] for k in tr]),
                             np.concatenate([(train_halves[k]["label"] == "D")[::TRAIN_EVERY] for k in tr]))
        live = [train_halves[k]["label"] != "D" for k in tr]
        tm = new_model().fit(np.vstack([XT[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
                             np.concatenate([(train_halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
        for k, h in enumerate(mixed):
            if h["match"] == m:
                out[k] = viterbi(dm.predict_proba(XDm[k].to_numpy())[:, 1],
                                 tm.predict_proba(XTm[k].to_numpy())[:, 1], **REFEREE_SETTINGS)
        print("  fold", m, "done", flush=True)
    return out


print("A: trained WITHOUT mix-ups", flush=True)
pred_a = lomo(clean, XDc, XTc)
print("B: trained WITH mix-ups", flush=True)
pred_b = lomo(mixed, XDm, XTm)
np.savez(os.path.join(DERIVED, "bench_wrongteam_preds.npz"), a=np.array(pred_a, dtype=object), b=np.array(pred_b, dtype=object))

print("\nScored on the bench WITH wrong-team episodes:")
compare(mixed, pred_a, pred_b, "A (no mix-ups in training)", "B (mix-ups in training)")
compare(mixed, pred_a, pred_b, "A", "B", metric="macro_f1")
