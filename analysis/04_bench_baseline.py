"""Step 6: check the test bench is realistic by running the organisers' baseline on it.

The baseline's published full-match scores on OUR match (EVALUATION.md) are a reference for
"does the bench look like our match?". Nothing is adjusted to hit them.

    python analysis/04_bench_baseline.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, score  # noqa: E402
from starter.baseline import smooth, HOME, AWAY, NOBODY  # noqa: E402  (the organisers' smoother)

PUBLISHED = {0: (0.6532, 0.0486), 25: (0.6752, 0.1996), 50: (0.6724, 0.2433), 125: (0.6356, 0.2242)}


def nearest_player_codes(h):
    """The baseline's step 1: team of the nearest visible player to the visible ball."""
    d2 = (h["px"] - h["bx"][:, None]) ** 2 + (h["py"] - h["by"][:, None]) ** 2
    d2 = np.where(np.isnan(d2), np.inf, d2)
    nearest = np.argmin(d2, axis=1)
    codes = h["team"][nearest].astype(np.int8)
    codes[~np.isfinite(d2.min(axis=1))] = NOBODY
    return codes


halves, profile = bench()
print("our mess profile: ball missing {:.1%} of graded frames, {} gaps, {:.1f} players visible".format(
    profile["target_missing"], len(profile["gap_lengths"]), profile["players_visible"]))
print("bench: {} halves from {} matches, ball missing {:.1%}, players visible {:.1f}\n".format(
    len(halves), len({h['match'] for h in halves}), np.mean([h["ball_missing"] for h in halves]),
    np.mean([(~np.isnan(h["px"])).sum(1).mean() for h in halves])))

codes = [nearest_player_codes(h) for h in halves]
print("{:<20} {:>8} {:>8} {:>7} {:>7}   {:>18}".format("baseline on bench", "macroF1", "spellF1", "poss", "true", "published on ours"))
for window in (0, 25, 50, 125):
    preds = [smooth(c, window) for c in codes]
    r = score(halves, preds)
    name = "baseline_raw" if window == 0 else "baseline_smooth{}".format(window)
    pub = PUBLISHED[window]
    print("{:<20} {:>8.4f} {:>8.4f} {:>7} {:>7}   {:>8.4f} {:>8.4f}".format(
        name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"], *pub))

# Per-match spread for the published floor (smooth50), to see how much matches differ.
print("\nsmooth50 per match (spellF1):")
for m in sorted({h["match"] for h in halves}):
    sel = [k for k, h in enumerate(halves) if h["match"] == m]
    r = score([halves[k] for k in sel], [smooth(codes[k], 50) for k in sel])
    print("   {}  {:.4f}".format(m, r["spell_f1_mean"]))
