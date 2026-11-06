"""Step 1: what does the public ground truth look like?

Reads only evaluation/gt/BAR-ATH_possession_gt_public.csv (public, ships with the repo).
Purpose is understanding, not tuning: nothing here feeds a parameter.

    python analysis/01_public_gt.py
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = 25
KICKOFF = 11185  # first graded frame of the 1st half


def clock(frame):
    """Match clock (mm:ss) since the first-half kickoff."""
    s = (frame - KICKOFF) / FPS
    return "{:02d}:{:04.1f}".format(int(s // 60), s % 60)


gt = pd.read_csv(GT)
graded = gt[gt["label"] != "X"].reset_index(drop=True)
print("graded frames:", len(graded), " = {:.1f} min".format(len(graded) / FPS / 60))
print("label counts:", graded["label"].value_counts().to_dict())
print()

# Cut into segments: unbroken runs of one label
change = graded["label"].ne(graded["label"].shift())
graded["seg"] = change.cumsum()
segs = graded.groupby("seg").agg(label=("label", "first"), start=("frame", "first"),
                                 end=("frame", "last"), frames=("frame", "size"))
segs["secs"] = segs["frames"] / FPS

print("{:>3}  {:5} {:>8} {:>8} {:>7}  {}".format("#", "label", "start", "end", "secs", "what comes next"))
for i, (_, r) in enumerate(segs.iterrows(), 1):
    nxt = segs["label"].iloc[i] if i < len(segs) else "-"
    print("{:>3}  {:5} {:>8} {:>8} {:>7.1f}  -> {}".format(
        i, r.label, clock(r.start), clock(r.end + 1), r.secs, nxt))
print()

pos = segs[segs["label"].isin(["H", "A"])]
dead = segs[segs["label"] == "D"]
print("possessions: {}  (H {}, A {})".format(len(pos), (pos.label == "H").sum(), (pos.label == "A").sum()))
print("  duration secs: min {:.1f}  median {:.1f}  mean {:.1f}  max {:.1f}".format(
    pos.secs.min(), pos.secs.median(), pos.secs.mean(), pos.secs.max()))
print("dead-ball stretches: {}".format(len(dead)))
print("  duration secs: min {:.1f}  median {:.1f}  mean {:.1f}  max {:.1f}".format(
    dead.secs.min(), dead.secs.median(), dead.secs.mean(), dead.secs.max()))
print()

# How does possession end? Directly to the other team, or via a stoppage?
pairs = list(zip(segs["label"], segs["label"].iloc[1:]))
kinds = pd.Series(["{}->{}".format(a, b) for a, b in pairs]).value_counts()
print("transitions between segments:")
for k, n in kinds.items():
    print("  {:5} {}".format(k, n))

# After a stoppage, is the restart taken by the team that had the ball before it?
same = other = 0
labels = list(segs["label"])
for i in range(1, len(labels) - 1):
    if labels[i] == "D" and labels[i - 1] in "HA" and labels[i + 1] in "HA":
        if labels[i - 1] == labels[i + 1]:
            same += 1
        else:
            other += 1
print("\nrestarts after a stoppage: same team as before {}, other team {}".format(same, other))
