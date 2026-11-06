"""Step 3: health check of the tactical tracking. Prints AGGREGATES ONLY -- no raw rows.

Section H splits a few signals by the public ground-truth label (first 5 minutes). That is for
understanding what a dead ball looks like in the tracking; no threshold or parameter is taken from
it (organisers' rules: no fitting or calibrating on the ground truth).

    python analysis/02_tracking_health.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402
from analysis.atd_arrays import load_arrays  # noqa: E402

FPS = 25
t = load_arrays()
L, W = t["pitch"]
n = len(t["bx"])

graded = np.zeros(n, dtype=bool)
half_of = np.zeros(n, dtype=np.int8)
for half, first, last in half_video_blocks("BAR-ATH"):
    graded[first:last + 1] = True
    half_of[first:last + 1] = half


def pct(a):
    return "{:5.1f}%".format(100.0 * np.mean(a)) if len(a) else "  n/a"


def runs(mask):
    """(start, length) of True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


seen = ~np.isnan(t["bx"])
print("A. BALL VISIBILITY (graded frames)")
print("   visible", pct(seen[graded]), "| 1st half", pct(seen[half_of == 1]), "| 2nd half", pct(seen[half_of == 2]))

print("\nB. BALL GAPS (runs of missing ball inside graded play)")
lens = []
for half in (1, 2):
    m = (half_of == half)
    _, l = runs(~seen & m)
    lens.extend(l.tolist())
lens = np.array(lens)
print("   gaps: {}  | length secs: median {:.2f}  p75 {:.2f}  p90 {:.2f}  p99 {:.1f}  max {:.1f}".format(
    len(lens), *(np.percentile(lens, [50, 75, 90, 99]) / FPS), lens.max() / FPS))
total = lens.sum()
for lo, hi, name in [(0, 25, "<= 1 s"), (25, 75, "1-3 s"), (75, 250, "3-10 s"), (250, 10**9, "> 10 s")]:
    sel = (lens > lo) & (lens <= hi)
    print("   gaps {:7}: {:5d} gaps, {:5.1f}% of all missing frames".format(name, sel.sum(), 100.0 * lens[sel].sum() / total))

print("\nC. BALL POSITION (graded, ball visible)")
gv = graded & seen
bx, by = t["bx"][gv], t["by"][gv]
out = (bx < 0) | (bx > L) | (by < 0) | (by > W)
print("   outside the pitch lines", pct(out), "| > 1 m outside", pct((bx < -1) | (bx > L + 1) | (by < -1) | (by > W + 1)))

print("\nD. BALL SPEED (graded, ball visible)")
bs = t["bs"][gv]
print("   speed field present on", pct(~np.isnan(bs)), "of visible frames -> unreliable, derive from positions")
# speed from positions over a 5-frame (0.2 s) span, m/s
SPAN = 5
ball_speed = np.full(n, np.nan)
ball_speed[:-SPAN] = np.hypot(t["bx"][SPAN:] - t["bx"][:-SPAN], t["by"][SPAN:] - t["by"][:-SPAN]) / (SPAN / FPS)
v = ball_speed[graded & ~np.isnan(ball_speed)]
print("   speed from positions m/s: median {:.1f}  p90 {:.1f}  p99 {:.1f}".format(*np.percentile(v, [50, 90, 99])))
print("   nearly still (< 0.5 m/s)", pct(v < 0.5), "| slow (< 2 m/s)", pct(v < 2))

print("\nD2. HOW REPRESENTATIVE IS THE PUBLIC WINDOW? ball visibility per 5-minute block")
blocks = [seen[s:s + 7500].mean() * 100 for h, a, b in half_video_blocks("BAR-ATH") for s in range(a, b + 1 - 7500, 7500)]
print("   public window (first 5 min): {:.1f}% | all 5-min blocks: min {:.0f}%  median {:.0f}%  max {:.0f}%".format(
    seen[11185:18685].mean() * 100, min(blocks), np.median(blocks), max(blocks)))

print("\nE. BALL TELEPORTS (consecutive visible graded frames)")
both = gv[1:] & gv[:-1]
step = np.hypot(np.diff(t["bx"]), np.diff(t["by"]))[both]
print("   moves per frame m: median {:.2f}  p99 {:.2f}".format(*np.percentile(step, [50, 99])))
print("   jumps > 1.5 m in one frame (> 135 km/h):", pct(step > 1.5), "=", int((step > 1.5).sum()), "frames")

print("\nF. PLAYERS VISIBLE (graded)")
present = ~np.isnan(t["px"])
home = present[:, t["team"] == 0].sum(1)
away = present[:, t["team"] == 1].sum(1)
tot = home + away
g = graded
print("   per frame: mean {:.1f} | home {:.1f} | away {:.1f}".format(tot[g].mean(), home[g].mean(), away[g].mean()))
print("   frames with < 18 visible", pct(tot[g] < 18), "| 18-22", pct((tot[g] >= 18) & (tot[g] <= 22)), "| > 22", pct(tot[g] > 22))
print("   frames where one team shows > 11", pct((home[g] > 11) | (away[g] > 11)))
starts = (present[1:] & ~present[:-1])[g[1:]].sum()
print("   new track segments starting: {:.0f} per minute (identity churn)".format(starts / (g.sum() / FPS / 60)))

print("\nG. PLAYER SPEED (graded) -- mean over visible players, m/s")
with np.errstate(all="ignore"):
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    mean_speed = np.nanmean(np.where(present, t["ps"], np.nan), axis=1)
ms = mean_speed[g & ~np.isnan(mean_speed)]
print("   p10 {:.2f}  median {:.2f}  p90 {:.2f}".format(*np.percentile(ms, [10, 50, 90])))
print("   frames with everyone walking (mean < 1.5 m/s)", pct(ms < 1.5))
long_gap = np.zeros(n, dtype=bool)
for half in (1, 2):
    st, ln = runs(~seen & (half_of == half))
    for a, l in zip(st, ln):
        if l > 10 * FPS:
            long_gap[a:a + l] = True
print("   players' median speed: ball visible {:.2f} | during ball gaps > 10 s {:.2f} m/s".format(
    np.nanmedian(mean_speed[graded & seen]), np.nanmedian(mean_speed[long_gap])))

print("\nH. PUBLIC 5 MINUTES, SPLIT BY TRUE LABEL (understanding only, no thresholds taken)")
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].to_numpy()
inside = (t["bx"] >= 0) & (t["bx"] <= L) & (t["by"] >= 0) & (t["by"] <= W)
print("   {:10} {:>7} {:>9} {:>10} {:>11} {:>12}".format("label", "frames", "ball seen", "ball <2m/s", "ball outside", "players m/s"))
for lab, name in [("H", "H alive"), ("A", "A alive"), ("D", "D dead")]:
    m = gt == lab
    s = seen[m]
    still = (ball_speed[m] < 2)[s & ~np.isnan(ball_speed[m])]
    outside = (~inside[m])[s]
    print("   {:10} {:>7} {:>9} {:>10} {:>11} {:>12.2f}".format(
        name, m.sum(), pct(s), pct(still), pct(outside), np.nanmedian(mean_speed[m])))
