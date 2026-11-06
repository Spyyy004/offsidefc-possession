"""Step 9 robustness check: does the turnover-clue gain survive realistic position noise?

Our real tracking is noisier than the bench (close control 24.8% of frames vs 39.7%; controller
flips 29.8% vs 19.4%). Jitter of ~1.0 m matches the control share, ~1.5-2 m the flip rate, so both
1.0 and 1.5 m are tested. Team judge v1 (team clues) and v2 (team + turnover clues) are retrained on
each jittered bench and decoded with the same referee settings (D27) and dead-ball outputs.

    python analysis/10_jitter_check.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score  # noqa: E402
from analysis.features import team_features, turnover_features  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
p_dead = list(np.load(os.path.join(DERIVED, "bench_p_dead.npz"), allow_pickle=True)["p"])
profile = our_mess_profile()


def lomo_team(halves, X):
    matches = sorted({h["match"] for h in halves})
    out = [None] * len(halves)
    for m in matches:
        tr = [k for k, h in enumerate(halves) if h["match"] != m]
        live = [halves[k]["label"] != "D" for k in tr]
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(
            np.vstack([X[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
            np.concatenate([(halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
        for k, h in enumerate(halves):
            if h["match"] == m:
                out[k] = model.predict_proba(X[k].to_numpy())[:, 1]
    return out


print("{:<8} {:<34} {:>8} {:>8} {:>6}".format("jitter", "team judge", "macroF1", "spellF1", "poss"))
for jitter in (1.0, 1.5):
    halves, _ = bench(profile, jitter=jitter)
    T = [team_features(h) for h in halves]
    O = [turnover_features(h) for h in halves]
    for name, X in [("v1: team clues", T), ("v2: team + turnover clues", [pd.concat([t, o], axis=1) for t, o in zip(T, O)])]:
        ph = lomo_team(halves, X)
        preds = [viterbi(p_dead[k], ph[k], **REFEREE_SETTINGS) for k in range(len(halves))]
        r = score(halves, preds)
        print("{:<8} {:<34} {:>8.4f} {:>8.4f} {:>6}".format("{} m".format(jitter), name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"]), flush=True)
