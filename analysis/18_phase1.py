"""Phase 1 (expert plan): oracle ceiling split + re-test control-style labels with paired deltas.

Current system = D38 + D39 (10-match leave-one-match-out probabilities, cached by 17_x_versions.py).

  Oracle (a) "decoder ceiling": true labels as near-certain emissions through our referee.
             What the decoder alone loses (its costs forbid some short segments).
  Oracle (b) "identity ceiling": our real emissions, TRUE segment boundaries fixed, each segment gets
             the label with the highest summed log-probability. What is lost to boundaries vs identity.
  Control labels: team model retrained on control-style DFL labels (Metrica labels unchanged),
             same dead-ball outputs and referee settings; paired per-match comparison.

    python analysis/18_phase1.py
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
from analysis.features import team_features, turnover_features  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
EPS = 1e-4
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True)
z = np.load(os.path.join(DERIVED, "bench_10match_deadlink_probs.npz"), allow_pickle=True)
p_dead, p_home = list(z["d"]), list(z["h"])
current = [viterbi(p_dead[k], p_home[k], **REFEREE_SETTINGS) for k in range(len(halves))]


def show(name, preds):
    for src in ("metrica", "dfl"):
        ks = [k for k, h in enumerate(halves) if h["source"] == src]
        r = score([halves[k] for k in ks], [preds[k] for k in ks])
        print("   {:<46} {:<8} spellF1 {:.4f}  macroF1 {:.4f}  @IoU.3/.5/.7 {:.2f}/{:.2f}/{:.2f}  poss {}/{}".format(
            name, src, r["spell_f1_mean"], r["macro_f1"], r["spell_f1"][0.3]["f1"], r["spell_f1"][0.5]["f1"],
            r["spell_f1"][0.7]["f1"], r["n_pred_spells"], r["n_gt_spells"]), flush=True)


# ---- Oracle (a): perfect emissions through our decoder --------------------------------------
oracle_a = []
for h in halves:
    lab = h["label"]
    pd_ = np.where(lab == "D", 1 - EPS, EPS)
    ph = np.where(lab == "H", 1 - EPS, np.where(lab == "A", EPS, 0.5))
    oracle_a.append(viterbi(pd_, ph, **REFEREE_SETTINGS))

# ---- Oracle (b): true boundaries, our emissions choose each segment's label ------------------
oracle_b = []
for k, h in enumerate(halves):
    lab = h["label"]
    pd_, ph = np.clip(p_dead[k], EPS, 1 - EPS), np.clip(p_home[k], EPS, 1 - EPS)
    ll = {"D": np.log(pd_), "H": np.log1p(-pd_) + np.log(ph), "A": np.log1p(-pd_) + np.log1p(-ph)}
    out = lab.copy()
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    for a, e in zip(np.concatenate(([0], b)), np.concatenate((b, [len(lab)]))):
        out[a:e] = max(ll, key=lambda s: ll[s][a:e].sum())
    oracle_b.append(out)

print("CEILING SPLIT")
show("current system", current)
show("(a) perfect emissions, our decoder", oracle_a)
show("(b) our emissions, TRUE boundaries fixed", oracle_b)

# ---- Control-style labels, paired ------------------------------------------------------------
print("\nCONTROL-STYLE TRAINING LABELS (paired per match)", flush=True)
hc, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, labels="control")
X = [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in hc]
p_ctrl = [None] * len(hc)
for m in sorted({h["match"] for h in hc}):
    tr = [k for k, h in enumerate(hc) if h["match"] != m]
    live = [hc[k]["label"] != "D" for k in tr]
    mdl = new_model().fit(np.vstack([X[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
                          np.concatenate([(hc[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
    for k, h in enumerate(hc):
        if h["match"] == m:
            p_ctrl[k] = mdl.predict_proba(X[k].to_numpy())[:, 1]
np.savez(os.path.join(DERIVED, "bench_10match_deadlink_p_home_ctrl.npz"), p=np.array(p_ctrl, dtype=object))
ctrl = [viterbi(p_dead[k], p_ctrl[k], **REFEREE_SETTINGS) for k in range(len(halves))]
print(" scored against DFL labels as published + Metrica:")
compare(halves, current, ctrl, "current", "control-trained")
print(" scored against control-style DFL labels + Metrica:")
compare(hc, current, ctrl, "current", "control-trained")
