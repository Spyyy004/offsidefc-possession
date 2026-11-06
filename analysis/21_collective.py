"""Phase 2 #3: collective-movement clues in the team judge, realistic bench (with mix-ups).

  current : team + turnover clues (D29), probabilities cached by 20_hsmm.py
  (a)     : + all collective clues
  (b)     : + only collective clues whose distribution matches our match (D23): forward speeds and
            their difference, inferred-ball distance and crowding (no crossing timing / flips, which
            occur ~2.2x more often on the bench than on our match)
Dead-ball outputs and referee (D27) unchanged. Paired per-match comparison (D41).

    python analysis/21_collective.py
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
from analysis.features import team_features, turnover_features, collective_features  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
MATCHED = ["home_fwd_speed", "away_fwd_speed", "fwd_speed_diff", "infer_near_diff",
           "infer_within5_diff", "infer_crowd", "infer_crowd_change_1s",
           "home_width_change_1s", "away_width_change_1s"]
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in halves})
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
p_dead, p_cur = list(z["d"]), list(z["h"])

base = [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in halves]
coll = [collective_features(h) for h in halves]
print("features ready", flush=True)


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


def decode(ph):
    return [viterbi(p_dead[k], ph[k], **REFEREE_SETTINGS) for k in range(len(halves))]


cur = decode(p_cur)
p_a = lomo_team([pd.concat([b, c], axis=1) for b, c in zip(base, coll)])
print("(a) trained", flush=True)
p_b = lomo_team([pd.concat([b, c[MATCHED]], axis=1) for b, c in zip(base, coll)])
print("(b) trained", flush=True)
np.savez(os.path.join(DERIVED, "bench_mixed_p_home_collective.npz"), a=np.array(p_a, dtype=object), b=np.array(p_b, dtype=object))
pred_a, pred_b = decode(p_a), decode(p_b)
for name, pr in (("(a) all collective clues", pred_a), ("(b) matched collective clues only", pred_b)):
    print("\n" + name)
    compare(halves, cur, pr, "current", name[:3])
    compare(halves, cur, pr, "current", name[:3], metric="macro_f1", verbose=False)
    r = compare(halves, cur, pr, "current", name[:3], metric="macro_f1", verbose=False)
    print("   macroF1: metrica mean {:+.4f} ({}/3) | all {:+.4f} ({}/10)".format(
        r["metrica"]["mean"], r["metrica"]["wins"], r["all"]["mean"], r["all"]["wins"]))
