"""Step 7: dead-ball detector, judged on the Bundesliga test bench (leave-one-match-out).

    python analysis/05_dead_ball.py

Compares, with the official metrics:
  baseline_smooth50          the organisers' floor
  + our dead-ball detector   dead/alive from our model, team from the baseline
  + perfect dead-ball        TRUE dead/alive (bench labels), team from the baseline: the ceiling
                             of what fixing dead balls alone can give
"""
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, score  # noqa: E402
from analysis.features import dead_ball_features  # noqa: E402
from starter.baseline import smooth, NOBODY  # noqa: E402

FPS = 25
MIN_DEAD = 2 * FPS        # rule R2: stoppages are never brief
TRAIN_EVERY = 5           # train on every 5th frame (neighbouring frames are near-duplicates)


def nearest_codes(h):
    d2 = (h["px"] - h["bx"][:, None]) ** 2 + (h["py"] - h["by"][:, None]) ** 2
    d2 = np.where(np.isnan(d2), np.inf, d2)
    c = h["team"][np.argmin(d2, axis=1)].astype(np.int8)
    c[~np.isfinite(d2.min(axis=1))] = NOBODY
    return c


def team_labels(h):
    """Baseline team guess on every frame: smooth50, gaps filled with the last team seen."""
    lab = smooth(nearest_codes(h), 50).astype(object)
    team = np.where(lab == "D", None, lab)
    last = None
    for i in range(len(team)):          # forward fill
        if team[i] is None:
            team[i] = last
        else:
            last = team[i]
    first = next((t for t in team if t is not None), "H")
    return np.array([t if t is not None else first for t in team])


def min_dead(dead, n):
    """Drop dead runs shorter than n frames (rule R2)."""
    dead = dead.copy()
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < n:
            dead[a:b] = False
    return dead


# Clues that only work because the Bundesliga data FREEZES the ball during stoppages (99% of dead
# frames still vs 2% in our tracking) -- they would not transfer to our match, so they are dropped.
FROZEN_BALL_ARTEFACTS = ["ball_speed", "ball_speed_med1s", "ball_speed_med3s",
                         "nearest_player_dist", "nearest_player_dist_med1s"]

halves, _ = bench()
X = [dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in halves]
y = [h["label"] == "D" for h in halves]
matches = sorted({h["match"] for h in halves})
names = list(X[0].columns)

p_dead = [None] * len(halves)
for m in matches:
    train = [k for k, h in enumerate(halves) if h["match"] != m]
    test = [k for k, h in enumerate(halves) if h["match"] == m]
    Xtr = np.vstack([X[k].to_numpy()[::TRAIN_EVERY] for k in train])
    ytr = np.concatenate([y[k][::TRAIN_EVERY] for k in train])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
    model.fit(Xtr, ytr)
    for k in test:
        p_dead[k] = model.predict_proba(X[k].to_numpy())[:, 1]
    print("fold {} done".format(m), flush=True)

teams = [team_labels(h) for h in halves]
base = [smooth(nearest_codes(h), 50) for h in halves]
ours = [np.where(min_dead(p > 0.5, MIN_DEAD), "D", t) for p, t in zip(p_dead, teams)]
perfect = [np.where(h["label"] == "D", "D", t) for h, t in zip(halves, teams)]

print("\n{:<28} {:>8} {:>8} {:>9} {:>8} {:>8} {:>7}".format(
    "on the test bench", "macroF1", "spellF1", "dead F1", "D prec", "D rec", "poss"))
for name, preds in [("baseline_smooth50", base), ("+ our dead-ball detector", ours),
                    ("+ perfect dead-ball (ceiling)", perfect)]:
    r = score(halves, preds)
    d = r["per_class"]["D"]
    print("{:<28} {:>8.4f} {:>8.4f} {:>9.4f} {:>8.3f} {:>8.3f} {:>7}".format(
        name, r["macro_f1"], r["spell_f1_mean"], r["alive_dead_f1"], d["precision"], d["recall"], r["n_pred_spells"]))
print("{:<28} {:>53}".format("true possessions", r["n_gt_spells"]))

# Which clues does the detector rely on? (permutation importance, last fold, 30k frames)
k = test[0]
idx = np.random.default_rng(0).choice(len(y[k]), size=min(30000, len(y[k])), replace=False)
imp = permutation_importance(model, X[k].to_numpy()[idx], y[k][idx], n_repeats=3, random_state=0, scoring="f1")
order = np.argsort(-imp.importances_mean)
print("\nmost useful clues (drop in dead-ball F1 when shuffled):")
for j in order[:10]:
    print("   {:<26} {:.3f}".format(names[j], imp.importances_mean[j]))
