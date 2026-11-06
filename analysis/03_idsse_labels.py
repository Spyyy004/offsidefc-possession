"""Step 4b: do the IDSSE/DFL possession labels behave like our ground truth?

Compares the STYLE of the labels (how possessions and stoppages are cut), not any score. Our
public 5 minutes are shown alongside for reference only; nothing is tuned here.

    python analysis/03_idsse_labels.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25


def segments(labels):
    """[(label, length_in_frames)] for one continuous half."""
    labels = np.asarray(labels)
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    return [(labels[a], b - a) for a, b in zip(bounds[:-1], bounds[1:])]


def describe(halves):
    segs_all, trans, same, other, frames = [], [], 0, 0, 0
    for labels in halves:
        segs = segments(labels)
        frames += len(labels)
        segs_all += segs
        labs = [s[0] for s in segs]
        trans += ["{}{}".format(a, b) for a, b in zip(labs, labs[1:])]
        for i in range(1, len(labs) - 1):
            if labs[i] == "D" and labs[i - 1] != "D" and labs[i + 1] != "D":
                same += labs[i - 1] == labs[i + 1]
                other += labs[i - 1] != labs[i + 1]
    pos = np.array([n for l, n in segs_all if l != "D"]) / FPS
    dead = np.array([n for l, n in segs_all if l == "D"]) / FPS
    direct = sum(t in ("HA", "AH") for t in trans)
    changes = direct + sum(t[0] == "D" for t in trans)
    mins = frames / FPS / 60
    return {
        "minutes": round(mins, 1),
        "poss per 5 min": round(len(pos) / mins * 5, 1),
        "poss median s": round(np.median(pos), 1),
        "poss mean s": round(pos.mean(), 1),
        "poss < 1 s": "{:.0%}".format(np.mean(pos < 1)),
        "poss < 2 s": "{:.0%}".format(np.mean(pos < 2)),
        "dead per 5 min": round(len(dead) / mins * 5, 1),
        "dead median s": round(np.median(dead), 1),
        "dead < 2 s": "{:.0%}".format(np.mean(dead < 2)),
        "dead share": "{:.0%}".format(sum(dead) * FPS / frames),
        "changes in open play": "{:.0%}".format(direct / max(changes, 1)),
        "restart to other team": "{:.0%}".format(other / max(same + other, 1)),
    }


rows = {}
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].to_numpy()
rows["OUR public 5 min"] = describe([gt[gt != "X"]])

all_halves = []
for path in sorted(glob.glob(os.path.join(ROOT, "external", "idsse", "cache", "*.npz"))):
    d = np.load(path)
    halves = [d["h1_label"], d["h2_label"]]
    all_halves += halves
    title = str(d["title"]).replace("FC ", "").replace("1. ", "")
    rows[title[:28]] = describe(halves)
if all_halves:
    rows["ALL IDSSE matches"] = describe(all_halves)

pd.set_option("display.width", 250)
print(pd.DataFrame(rows).T.to_string())

# How unusual are OUR public 5 minutes, in DFL style? Count possessions in every 5-minute window
# (stepping 1 minute) of every IDSSE half, starting at kickoff of each half like ours.
WIN, STEP = 5 * 60 * FPS, 60 * FPS
counts, dead_shares = [], []
for labels in all_halves:
    for start in range(0, len(labels) - WIN + 1, STEP):
        w = labels[start:start + WIN]
        counts.append(sum(l != "D" for l, _ in segments(w)))
        dead_shares.append(np.mean(w == "D"))
counts, dead_shares = np.array(counts), np.array(dead_shares)
if len(counts):
    print("\nIDSSE 5-minute windows: {}  | possessions per window: p10 {:.0f}  median {:.0f}  p90 {:.0f}".format(
        len(counts), *np.percentile(counts, [10, 50, 90])))
    print("windows with <= 17 possessions: {:.1%}   | with dead share <= 20%: {:.1%}".format(
        np.mean(counts <= 17), np.mean(dead_shares <= 0.20)))
    calm = dead_shares <= 0.25
    print("among fluid windows (dead share <= 25%): median {:.0f} possessions, {:.1%} have <= 17".format(
        np.median(counts[calm]), np.mean(counts[calm] <= 17)))
