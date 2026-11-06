"""D18: "X when unsure" vs "always guess", on the current system (D38 + D39).

The scorer forward-fills X with our previous label before computing spellF1, so X can only help
spellF1 by removing a doubtful short piece (it gets absorbed into the previous label); X inside a
possession leaves spellF1 unchanged and costs macroF1.

  always guess            : current output
  X inside possessions    : X on the least confident 10% of frames that do not touch a change
  X on doubtful pieces    : predicted pieces shorter than L seconds with mean confidence below C
                            become X (L, C chosen by nested leave-one-match-out)

    python analysis/17_x_versions.py
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

FPS = 25
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True)
matches = sorted({h["match"] for h in halves})
by_match = {m: [k for k, h in enumerate(halves) if h["match"] == m] for m in matches}
cache = os.path.join(DERIVED, "bench_10match_deadlink_probs.npz")

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
        print("fold", m, "done", flush=True)
    np.savez(cache, d=np.array(p_dead, dtype=object), h=np.array(p_home, dtype=object))

guess = [viterbi(p_dead[k], p_home[k], **REFEREE_SETTINGS) for k in range(len(halves))]


def confidence(k, lab):
    """Per-frame confidence in the label we gave (0..1)."""
    pd_, ph = p_dead[k], p_home[k]
    return np.where(lab == "D", pd_, np.where(lab == "H", (1 - pd_) * ph, (1 - pd_) * (1 - ph)))


def x_inside(k, lab, share=0.10):
    conf = confidence(k, lab)
    change = np.zeros(len(lab), bool)
    idx = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    for i in idx:
        change[max(0, i - FPS):i + FPS] = True
    cut = np.quantile(conf[~change], share) if (~change).any() else -1
    out = lab.copy()
    out[(conf <= cut) & ~change] = "X"
    return out


def x_pieces(k, lab, max_s, max_conf):
    conf = confidence(k, lab)
    out = lab.copy()
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], b, [len(lab)]))
    for a, z in zip(bounds[:-1], bounds[1:]):
        if a > 0 and (z - a) < max_s * FPS and conf[a:z].mean() < max_conf:
            out[a:z] = "X"
    return out


grid = list(itertools.product([1.0, 2.0, 3.0, 5.0], [0.5, 0.6, 0.7]))
table = {g: {m: score([halves[k] for k in ks], [x_pieces(k, guess[k], *g) for k in ks])["spell_f1_mean"]
             for m, ks in by_match.items()} for g in grid}
nested, chosen = [None] * len(halves), {}
for m, ks in by_match.items():
    best = max(grid, key=lambda g: np.mean([table[g][o] for o in matches if o != m]))
    chosen[m] = best
    for k in ks:
        nested[k] = x_pieces(k, guess[k], *best)
print("doubtful-piece settings chosen (max length s, max confidence):", sorted(set(chosen.values())))

variants = [("always guess (current)", guess),
            ("X inside possessions (10% least sure)", [x_inside(k, guess[k]) for k in range(len(halves))]),
            ("X on doubtful short pieces (nested)", nested)]
for source in ("metrica", "dfl"):
    ks = [k for k, h in enumerate(halves) if h["source"] == source]
    print("\n{} scoreboard".format(source.upper()))
    for name, preds in variants:
        r = score([halves[k] for k in ks], [preds[k] for k in ks])
        print("   {:<40} spellF1 {:.4f}  macroF1 {:.4f}  X on graded frames {:5.1%}".format(
            name, r["spell_f1_mean"], r["macro_f1"], r["n_abstain_on_graded"] / r["n_graded"]))
