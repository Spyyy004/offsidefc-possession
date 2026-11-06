"""Phase 2 #2: duration-aware decoder (HSMM) vs the current HMM, same emissions.

Realistic bench (stoppage-linked blackouts, 1 m jitter, wrong-team episodes); models trained with
mix-ups (D45), leave-one-match-out. Duration priors and transitions are learned per fold from the
TRAINING matches only. Settings by nested leave-one-match-out; decision by paired deltas (D41).

    python analysis/20_hsmm.py
"""
import itertools
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
from analysis.hsmm import decode, learn_priors  # noqa: E402
from analysis.paired import compare  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
MAX_BLOCKS = 1500                     # 300 s
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in halves})
by_match = {m: [k for k, h in enumerate(halves) if h["match"] == m] for m in matches}

cache = os.path.join(DERIVED, "bench_mixed_probs.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    p_dead, p_home = list(z["d"]), list(z["h"])
else:
    XD = [dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in halves]
    XT = [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in halves]
    p_dead, p_home = [None] * len(halves), [None] * len(halves)
    for m in matches:
        tr = [k for k, h in enumerate(halves) if h["match"] != m]
        dm = new_model().fit(np.vstack([XD[k].to_numpy()[::TRAIN_EVERY] for k in tr]),
                             np.concatenate([(halves[k]["label"] == "D")[::TRAIN_EVERY] for k in tr]))
        live = [halves[k]["label"] != "D" for k in tr]
        tm = new_model().fit(np.vstack([XT[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
                             np.concatenate([(halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
        for k in by_match[m]:
            p_dead[k] = dm.predict_proba(XD[k].to_numpy())[:, 1]
            p_home[k] = tm.predict_proba(XT[k].to_numpy())[:, 1]
        print("  fold", m, "done", flush=True)
    np.savez(cache, d=np.array(p_dead, dtype=object), h=np.array(p_home, dtype=object))
print("emissions ready", flush=True)

hmm = [viterbi(p_dead[k], p_home[k], **REFEREE_SETTINGS) for k in range(len(halves))]

# priors per held-out match, from training matches only
priors = {}
for m in matches:
    for src in ("pooled", "metrica"):
        labs = [h["label"] for h in halves if h["match"] != m and (src == "pooled" or h["source"] == "metrica")]
        priors[(m, src)] = learn_priors(labs, MAX_BLOCKS)

grid = list(itertools.product([0.3, 0.6, 1.0], [0.5, 1.0, 2.0], [1.0, 2.0], ["pooled", "metrica"]))


def run(params, ks):
    gamma, lam, beta, src = params
    return {k: decode(p_dead[k], p_home[k], priors[(halves[k]["match"], src)], beta=beta, gamma=gamma, lam=lam)
            for k in ks}


table = {}
for i, g in enumerate(grid):
    pr = run(g, range(len(halves)))
    table[g] = {m: score([halves[k] for k in ks], [pr[k] for k in ks])["spell_f1_mean"] for m, ks in by_match.items()}
    if i % 6 == 0:
        print("  grid {}/{}".format(i + 1, len(grid)), flush=True)

nested, chosen = [None] * len(halves), {}
for m, ks in by_match.items():
    best = max(grid, key=lambda g: np.mean([table[g][o] for o in matches if o != m]))
    chosen[m] = best
    pr = run(best, ks)
    for k in ks:
        nested[k] = pr[k]
np.savez(os.path.join(DERIVED, "bench_mixed_hsmm_preds.npz"), p=np.array(nested, dtype=object))
print("\nsettings chosen per test match (gamma, lam, beta, prior source):")
for m in matches:
    print("   ", m, chosen[m])

for src in ("metrica", "dfl"):
    ks = [k for k, h in enumerate(halves) if h["source"] == src]
    for name, pr in (("HMM (current)", hmm), ("HSMM (nested)", nested)):
        r = score([halves[k] for k in ks], [pr[k] for k in ks])
        print("   {:<8} {:<15} spellF1 {:.4f}  macroF1 {:.4f}  @IoU.3/.5/.7 {:.2f}/{:.2f}/{:.2f}  changes<=1s {:.3f}  poss {}/{}".format(
            src, name, r["spell_f1_mean"], r["macro_f1"], r["spell_f1"][0.3]["f1"], r["spell_f1"][0.5]["f1"],
            r["spell_f1"][0.7]["f1"], r["transitions"][25]["f1"], r["n_pred_spells"], r["n_gt_spells"]))
print()
compare(halves, hmm, nested, "HMM", "HSMM")
compare(halves, hmm, nested, "HMM", "HSMM", metric="macro_f1")
