"""Reusable pieces of the dead-ball detector (Step 7), shared by the experiments and the final run."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from starter.baseline import smooth, NOBODY

FPS = 25
TRAIN_EVERY = 5

# Clues that only work because the Bundesliga data freezes the ball during stoppages (D22, D23).
FROZEN_BALL_ARTEFACTS = ["ball_speed", "ball_speed_med1s", "ball_speed_med3s",
                         "nearest_player_dist", "nearest_player_dist_med1s"]


def new_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)


def leave_one_match_out(halves, X, y):
    """Out-of-fold dead probabilities: each match predicted by a model trained on the others."""
    p = [None] * len(halves)
    for m in sorted({h["match"] for h in halves}):
        train = [k for k, h in enumerate(halves) if h["match"] != m]
        model = new_model().fit(np.vstack([X[k].to_numpy()[::TRAIN_EVERY] for k in train]),
                                np.concatenate([y[k][::TRAIN_EVERY] for k in train]))
        for k, h in enumerate(halves):
            if h["match"] == m:
                p[k] = model.predict_proba(X[k].to_numpy())[:, 1]
    return p


def nearest_codes(h):
    d2 = (h["px"] - h["bx"][:, None]) ** 2 + (h["py"] - h["by"][:, None]) ** 2
    d2 = np.where(np.isnan(d2), np.inf, d2)
    c = h["team"][np.argmin(d2, axis=1)].astype(np.int8)
    c[~np.isfinite(d2.min(axis=1))] = NOBODY
    return c


def baseline_team(h):
    """Baseline team guess on every frame: smooth50, gaps filled with the last team seen."""
    lab = pd.Series(smooth(nearest_codes(h), 50)).replace("D", np.nan).ffill().bfill()
    return lab.fillna("H").to_numpy()


def _drop_short_runs(mask, n):
    mask = mask.copy()
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < n:
            mask[a:b] = False
    return mask


def dead_mask(p, smooth_s=0.0, threshold=0.5, min_dead_s=2.0, fill_alive_s=0.0):
    """Turn per-frame dead probabilities into a dead/alive decision (refinement #1)."""
    if smooth_s > 0:
        p = pd.Series(p).rolling(max(1, int(smooth_s * FPS)), center=True, min_periods=1).mean().to_numpy()
    dead = p > threshold
    if fill_alive_s > 0:   # tiny 'alive' blips between two stoppages are noise
        dead = ~_drop_short_runs(~dead, int(fill_alive_s * FPS))
    return _drop_short_runs(dead, int(min_dead_s * FPS))
