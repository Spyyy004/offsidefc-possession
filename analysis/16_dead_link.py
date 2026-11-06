"""Lever 1: train on Bundesliga blackouts linked to stoppages like real tracking (Metrica's rates).

The Metrica games are identical in both versions, so the Metrica scoreboard compares exactly.
10-match leave-one-match-out, realistic jitter, referee D27.

    python analysis/16_dead_link.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER  # noqa: E402
from analysis.features import dead_ball_features, team_features, turnover_features  # noqa: E402
from analysis.dead_ball import FROZEN_BALL_ARTEFACTS, TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
profile = our_mess_profile()
halves, _ = bench(profile, jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True)
old_halves, _ = bench(profile, jitter=BENCH_JITTER, sources=("dfl", "metrica"))
matches = sorted({h["match"] for h in halves})

# D23: blackout lengths now vs our match
def gaps(hs):
    out = []
    for h in hs:
        m = np.isnan(h["bx"])
        e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        out += ((np.flatnonzero(e == -1) - np.flatnonzero(e == 1)) / 25).tolist()
    return np.array(out)
g_new = gaps([h for h in halves if h["source"] == "dfl"])
print("blackout length s (p25/p50/p75): bench now {:.1f}/{:.1f}/{:.1f} | ours {:.1f}/{:.1f}/{:.1f}".format(
    *np.percentile(g_new, [25, 50, 75]), *np.percentile(profile["gap_lengths"] / 25, [25, 50, 75])), flush=True)

XD = [dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in halves]
XT = [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in halves]
new = [None] * len(halves)
for m in matches:
    tr = [k for k, h in enumerate(halves) if h["match"] != m]
    dm = new_model().fit(np.vstack([XD[k].to_numpy()[::TRAIN_EVERY] for k in tr]),
                         np.concatenate([(halves[k]["label"] == "D")[::TRAIN_EVERY] for k in tr]))
    live = [halves[k]["label"] != "D" for k in tr]
    tm = new_model().fit(np.vstack([XT[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
                         np.concatenate([(halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
    for k, h in enumerate(halves):
        if h["match"] == m:
            new[k] = viterbi(dm.predict_proba(XD[k].to_numpy())[:, 1], tm.predict_proba(XT[k].to_numpy())[:, 1], **REFEREE_SETTINGS)
    print("fold", m, "done", flush=True)
np.savez(os.path.join(DERIVED, "bench_10match_deadlink_preds.npz"), p=np.array(new, dtype=object))
old = list(np.load(os.path.join(DERIVED, "bench_10match_preds.npz"), allow_pickle=True)["p"])


def row(name, hs, preds, source):
    ks = [k for k, h in enumerate(hs) if h["source"] == source]
    r = score([hs[k] for k in ks], [preds[k] for k in ks])
    print("   {:<36} spellF1 {:.4f}  macroF1 {:.4f}  alive/dead F1 {:.3f}  D F1 {:.3f}  changes<=1s {:.3f}  poss {}/{}".format(
        name, r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["per_class"]["D"]["f1"],
        r["transitions"][25]["f1"], r["n_pred_spells"], r["n_gt_spells"]))


print("\nMETRICA scoreboard (identical games in both versions)")
row("before (random-ish blackouts)", old_halves, old, "metrica")
row("after (stoppage-linked blackouts)", halves, new, "metrica")
print("BUNDESLIGA scoreboard (each on its own bench; not directly comparable)")
row("before", old_halves, old, "dfl")
row("after", halves, new, "dfl")
