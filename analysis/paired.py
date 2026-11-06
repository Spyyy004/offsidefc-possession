"""Per-match paired comparison of two systems (D41) -- decide on paired deltas, not pooled means.

For each match: score A and B with the official metric, take the delta. Report wins/losses, a
two-sided sign test, and a bootstrap 90% confidence interval of the mean delta (matches resampled
with replacement, fixed seed). Metrica (primary) and Bundesliga are reported separately and together.
"""
import math

import numpy as np

from analysis.bench import score


def per_match(halves, preds, metric="spell_f1_mean"):
    out = {}
    for m in sorted({h["match"] for h in halves}):
        ks = [k for k, h in enumerate(halves) if h["match"] == m]
        out[m] = score([halves[k] for k in ks], [preds[k] for k in ks])[metric]
    return out


def sign_test(wins, losses):
    """Two-sided exact sign test p-value (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_ci(deltas, n_draws=10000, seed=0, level=0.90):
    d = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_draws, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def compare(halves, preds_a, preds_b, name_a="A", name_b="B", metric="spell_f1_mean", verbose=True):
    a, b = per_match(halves, preds_a, metric), per_match(halves, preds_b, metric)
    src = {h["match"]: h.get("source", "dfl") for h in halves}
    rows = {}
    for group, keep in [("metrica", lambda m: src[m] == "metrica"), ("bundesliga", lambda m: src[m] == "dfl"),
                        ("all", lambda m: True)]:
        ms = [m for m in a if keep(m)]
        if not ms:
            continue
        d = np.array([b[m] - a[m] for m in ms])
        wins, losses = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        lo, hi = bootstrap_ci(d)
        rows[group] = dict(n=len(ms), mean=float(d.mean()), wins=wins, losses=losses,
                           p=sign_test(wins, losses), lo=lo, hi=hi)
    if verbose:
        print("   {} vs {} on {}:".format(name_b, name_a, metric))
        for m in a:
            print("      {:<16} {:.4f} -> {:.4f}  ({:+.4f})".format(m, a[m], b[m], b[m] - a[m]))
        for g, r in rows.items():
            print("      {:<11} mean {:+.4f}  wins {}/{}  sign-test p={:.2f}  90% CI [{:+.4f}, {:+.4f}]".format(
                g, r["mean"], r["wins"], r["n"], r["p"], r["lo"], r["hi"]))
    return rows
