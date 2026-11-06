"""Full scorecard: EVERY metric the official scorer reports, for the current system (D38: trained on
all 10 external matches, leave-one-match-out) and the organisers' baseline, on both scoreboards
(Bundesliga, Metrica), next to the perfect value.

    python analysis/15_full_report.py
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
from analysis.dead_ball import FROZEN_BALL_ARTEFACTS, TRAIN_EVERY, new_model, nearest_codes  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from starter.baseline import smooth  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"))
matches = sorted({h["match"] for h in halves})
cache = os.path.join(DERIVED, "bench_10match_preds.npz")

if os.path.exists(cache):
    ours = list(np.load(cache, allow_pickle=True)["p"])
else:
    XD = [dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS) for h in halves]
    XT = [pd.concat([team_features(h), turnover_features(h)], axis=1) for h in halves]
    ours = [None] * len(halves)
    for m in matches:
        tr = [k for k, h in enumerate(halves) if h["match"] != m]
        dm = new_model().fit(np.vstack([XD[k].to_numpy()[::TRAIN_EVERY] for k in tr]),
                             np.concatenate([(halves[k]["label"] == "D")[::TRAIN_EVERY] for k in tr]))
        live = [halves[k]["label"] != "D" for k in tr]
        tm = new_model().fit(np.vstack([XT[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)]),
                             np.concatenate([(halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)]))
        for k, h in enumerate(halves):
            if h["match"] == m:
                ours[k] = viterbi(dm.predict_proba(XD[k].to_numpy())[:, 1],
                                  tm.predict_proba(XT[k].to_numpy())[:, 1], **REFEREE_SETTINGS)
        print("fold", m, "done", flush=True)
    np.savez(cache, p=np.array(ours, dtype=object))
base = [smooth(nearest_codes(h), 50) for h in halves]


def card(r):
    t12, t25 = r["transitions"][12], r["transitions"][25]
    pc = r["per_class"]
    rows = [
        ("RANKING  spellF1 (mean over IoU 0.3-0.7)", r["spell_f1_mean"], 1.0),
        ("  spellF1 @ IoU 0.3 (loose)", r["spell_f1"][0.3]["f1"], 1.0),
        ("  spellF1 @ IoU 0.5", r["spell_f1"][0.5]["f1"], 1.0),
        ("  spellF1 @ IoU 0.7 (strict)", r["spell_f1"][0.7]["f1"], 1.0),
        ("  possessions found (recall @0.5)", r["spell_f1"][0.5]["recall"], 1.0),
        ("  our possessions correct (precision @0.5)", r["spell_f1"][0.5]["precision"], 1.0),
        ("TIE-BREAK  macroF1", r["macro_f1"], 1.0),
        ("  F1 Barca/home (H)", pc["H"]["f1"], 1.0),
        ("  F1 Athletic/away (A)", pc["A"]["f1"], 1.0),
        ("  F1 dead ball (D)", pc["D"]["f1"], 1.0),
        ("accuracy (share of frames right)", r["accuracy"], 1.0),
        ("kappa (accuracy beyond chance)", r["kappa"], 1.0),
        ("alive/dead F1", r["alive_dead_f1"], 1.0),
        ("team accuracy when ball alive", r["team_acc_both_alive"], 1.0),
        ("possession changes caught within 0.5 s (transF1)", t12["f1"], 1.0),
        ("possession changes caught within 1 s (transF1)", t25["f1"], 1.0),
        ("possession-share error (lower is better)", r["share_err"], 0.0),
        ("fragmentation: segments ours / true", r["fragmentation"], 1.0),
        ("possessions: ours", r["n_pred_spells"], r["n_gt_spells"]),
        ("mean possession length s: ours", r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"]),
    ]
    return rows


for source, name in [("dfl", "BUNDESLIGA scoreboard (7 matches, touch-style labels)"),
                     ("metrica", "METRICA scoreboard (3 games, same company/style as our ground truth)")]:
    ks = [k for k, h in enumerate(halves) if h["source"] == source]
    rb = score([halves[k] for k in ks], [base[k] for k in ks])
    ro = score([halves[k] for k in ks], [ours[k] for k in ks])
    print("\n" + name)
    print("   {:<50} {:>9} {:>9} {:>9}".format("metric", "baseline", "OURS", "perfect"))
    for (label, b, perfect), (_, o, _) in zip(card(rb), card(ro)):
        fmt = "{:>9.0f}" if isinstance(perfect, (int, np.integer)) and perfect > 1 else "{:>9.3f}"
        print("   {:<50} ".format(label) + fmt.format(b) + " " + fmt.format(o) + " " + fmt.format(perfect))
