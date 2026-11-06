"""Step 9e: Metrica sample games as (a) an outside transfer test and (b) extra training data.

  A  transfer : train on the 7 Bundesliga matches only, score on the 3 Metrica games
                (different provider, label style close to our ground truth, realistic ball gaps)
  B  more data: leave-one-match-out over all 10 matches; score Bundesliga and Metrica separately,
                compared with Bundesliga-only training

Realistic bench (1 m jitter). Referee D27.

    python analysis/14_metrica.py
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
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"))
src = np.array([h["source"] for h in halves])
matches = sorted({h["match"] for h in halves})
print("halves: {} Bundesliga, {} Metrica | Metrica ball missing {:.0%}".format(
    (src == "dfl").sum(), (src == "metrica").sum(),
    np.mean([h["ball_missing"] for h in halves if h["source"] == "metrica"])), flush=True)

XD = [dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in halves]
XT = [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in halves]


def fit(train_idx, X, target, live_only):
    rows = [(halves[k]["label"] != "D") if live_only else slice(None) for k in train_idx]
    return new_model().fit(np.vstack([X[k].to_numpy()[r][::TRAIN_EVERY] for k, r in zip(train_idx, rows)]),
                           np.concatenate([target(halves[k]["label"])[r][::TRAIN_EVERY] for k, r in zip(train_idx, rows)]))


def predict(train_idx, test_idx):
    dm = fit(train_idx, XD, lambda l: l == "D", False)
    tm = fit(train_idx, XT, lambda l: l == "H", True)
    return {k: viterbi(dm.predict_proba(XD[k].to_numpy())[:, 1], tm.predict_proba(XT[k].to_numpy())[:, 1],
                       **REFEREE_SETTINGS) for k in test_idx}


def report(name, preds):
    for source in ("dfl", "metrica"):
        ks = [k for k in preds if halves[k]["source"] == source]
        if ks:
            r = score([halves[k] for k in ks], [preds[k] for k in ks])
            print("   {:<44} on {:<8} macroF1 {:.4f}  spellF1 {:.4f}  poss {} / true {}".format(
                name, source, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]), flush=True)


dfl_idx = [k for k in range(len(halves)) if src[k] == "dfl"]
met_idx = [k for k in range(len(halves)) if src[k] == "metrica"]

print("\nA. TRANSFER TEST (train on 7 Bundesliga matches only)")
report("Bundesliga-trained model", predict(dfl_idx, met_idx))

print("\nB. MORE DATA: leave-one-match-out over all 10 matches")
preds = {}
for m in matches:
    test = [k for k, h in enumerate(halves) if h["match"] == m]
    preds.update(predict([k for k in range(len(halves)) if halves[k]["match"] != m], test))
    print("   fold", m, "done", flush=True)
report("trained on the other 9 matches", preds)
p_dead_old = list(np.load(os.path.join(DERIVED, "bench_j1_p_dead.npz"), allow_pickle=True)["p"])
p_home_old = list(np.load(os.path.join(DERIVED, "bench_j1_p_home_v2.npz"), allow_pickle=True)["p"])
report("reference: Bundesliga-only LOMO (6 matches)",
       {k: viterbi(p_dead_old[k], p_home_old[k], **REFEREE_SETTINGS) for k in dfl_idx})
