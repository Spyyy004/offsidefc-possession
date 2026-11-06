"""Step 8a: the team frame judge (Layer 2), judged on the honest test bench.

For every frame: probability that the HOME team has the ball, from team_features(). Trained on
live frames of six matches, predicting the seventh (leave-one-match-out). Dead balls come from
the Step 7 detector with the D25 settings in every variant, so only the team part differs.

    python analysis/07_team_judge.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, score  # noqa: E402
from analysis.features import dead_ball_features, team_features  # noqa: E402
from analysis.dead_ball import (FROZEN_BALL_ARTEFACTS, leave_one_match_out, baseline_team,  # noqa: E402
                                dead_mask, TRAIN_EVERY)
from starter.baseline import smooth  # noqa: E402

D25 = dict(smooth_s=0.5, threshold=0.4, min_dead_s=4.0, fill_alive_s=1.0)
CACHE = os.path.join(ROOT, "external", "idsse", "derived", "bench_p_dead.npz")

halves, _ = bench()
matches = sorted({h["match"] for h in halves})

# Dead-ball probabilities from Step 7 (cached: they do not change in this step)
if os.path.exists(CACHE):
    p_dead = list(np.load(CACHE, allow_pickle=True)["p"])
else:
    Xd = [dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in halves]
    p_dead = leave_one_match_out(halves, Xd, [h["label"] == "D" for h in halves])
    np.savez(CACHE, p=np.array(p_dead, dtype=object))
dead = [dead_mask(p, **D25) for p in p_dead]

# Team judge, leave-one-match-out, trained on truly live frames only
Xt = [team_features(h) for h in halves]
names = list(Xt[0].columns)
p_home = [None] * len(halves)
for m in matches:
    tr = [k for k, h in enumerate(halves) if h["match"] != m]
    Xtr = np.vstack([Xt[k].to_numpy()[halves[k]["label"] != "D"][::TRAIN_EVERY] for k in tr])
    ytr = np.concatenate([(halves[k]["label"][halves[k]["label"] != "D"] == "H")[::TRAIN_EVERY] for k in tr])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
    for k, h in enumerate(halves):
        if h["match"] == m:
            p_home[k] = model.predict_proba(Xt[k].to_numpy())[:, 1]
    print("fold", m, "done", flush=True)


def with_dead(team_labels, k):
    return np.where(dead[k], "D", team_labels)


def majority(p, window):
    codes = np.where(p > 0.5, 0, 1).astype(np.int8)            # 0 home, 1 away (baseline codes)
    return smooth(codes, window)


variants = {
    "baseline team (current best)": [with_dead(baseline_team(h), k) for k, h in enumerate(halves)],
    "judge, raw per frame": [with_dead(np.where(p > 0.5, "H", "A"), k) for k, p in enumerate(p_home)],
    "judge + majority vote +/-2 s": [with_dead(majority(p, 50), k) for k, p in enumerate(p_home)],
    "judge, confidence averaged 2 s": [with_dead(np.where(pd.Series(p).rolling(50, center=True, min_periods=1).mean() > 0.5, "H", "A"), k)
                                       for k, p in enumerate(p_home)],
    "perfect team + our dead balls (ceiling)": [with_dead(np.where(h["label"] == "A", "A", "H"), k) for k, h in enumerate(halves)],
}

print("\n{:<42} {:>8} {:>8} {:>9} {:>6}".format("on the test bench", "macroF1", "spellF1", "team acc", "poss"))
for name, preds in variants.items():
    r = score(halves, preds)
    print("{:<42} {:>8.4f} {:>8.4f} {:>9.3f} {:>6}".format(
        name, r["macro_f1"], r["spell_f1_mean"], r["team_acc_both_alive"], r["n_pred_spells"]))
print("{:<42} {:>34}".format("true possessions", r["n_gt_spells"]))

k = [i for i, h in enumerate(halves) if h["match"] == m][0]
live = np.flatnonzero(halves[k]["label"] != "D")
idx = np.random.default_rng(0).choice(live, size=min(30000, len(live)), replace=False)
imp = permutation_importance(model, Xt[k].to_numpy()[idx], halves[k]["label"][idx] == "H",
                             n_repeats=3, random_state=0, scoring="accuracy")
print("\nmost useful clues for 'which team' (drop in accuracy when shuffled):")
for j in np.argsort(-imp.importances_mean)[:10]:
    print("   {:<26} {:.3f}".format(names[j], imp.importances_mean[j]))
np.savez(os.path.join(ROOT, "external", "idsse", "derived", "bench_p_home.npz"), p=np.array(p_home, dtype=object))
