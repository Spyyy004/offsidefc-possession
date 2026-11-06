"""Phase 2 #4a: last-touch-before-out rule, paired per match on the realistic bench (with mix-ups).
Caveat: Metrica labels were derived control-style (a deflection out does not change possession), so
this rule is expected to look worse there by construction; Bundesliga labels are last-touch style.

    python analysis/22_last_touch.py
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
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.last_touch import apply_last_touch  # noqa: E402
from analysis.paired import compare  # noqa: E402
from games import half_video_blocks  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
cur = [viterbi(z["d"][k], z["h"][k], **REFEREE_SETTINGS) for k in range(len(halves))]
new, n_changed = [], 0
for h, p in zip(halves, cur):
    q, c = apply_last_touch(p, h)
    new.append(q)
    n_changed += c
print("rule fired {} times on the bench".format(n_changed))
compare(halves, cur, new, "current", "last-touch rule")
r = compare(halves, cur, new, "current", "last-touch rule", metric="macro_f1", verbose=False)
print("   macroF1: metrica {:+.4f} ({}/3) | bundesliga {:+.4f} ({}/7)".format(
    r["metrica"]["mean"], r["metrica"]["wins"], r["bundesliga"]["mean"], r["bundesliga"]["wins"]))


def restart_other(preds):
    s = o = 0
    for p in preds:
        b = np.flatnonzero(p[1:] != p[:-1]) + 1
        segs = [p[x] for x in np.concatenate(([0], b))]
        for i in range(1, len(segs) - 1):
            if segs[i] == "D" and segs[i - 1] != "D" and segs[i + 1] != "D":
                s += segs[i - 1] == segs[i + 1]
                o += segs[i - 1] != segs[i + 1]
    return o / max(1, s + o)


for src in ("dfl", "metrica"):
    ks = [k for k, h in enumerate(halves) if h["source"] == src]
    print("   restart to other team, {}: truth {:.0%} | current {:.0%} | with rule {:.0%}".format(
        src, restart_other([halves[k]["label"] for k in ks]), restart_other([cur[k] for k in ks]),
        restart_other([new[k] for k in ks])))
