"""The test bench: Bundesliga (IDSSE/DFL) matches made to look like our match, scored with the
official metric code. Every later step is judged here, never on the public 5 minutes (D14).

1. load_dfl_halves()  -- each half in OUR shape: metres from the corner, speeds in m/s,
                         team 0/1 per player column, labels H/A/D per frame.
2. degrade()          -- make the clean tracking messy like ours (D13):
                         * ball blackouts with OUR gap-length distribution, placed more often
                           when players are slow (the pattern measured in Step 3)
                         * player dropouts so ~20.5 of 22 are visible on average
                         The pattern comes from our TRACKING only -- never the ground truth or video.
3. score()            -- the official evaluation.possession_metrics.evaluate() on any prediction.

Deterministic: fixed seeds, so the bench is identical on every run (D8).
"""
import glob
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (the official metric library)
from games import half_video_blocks  # noqa: E402
from analysis.atd_arrays import load_arrays  # noqa: E402

FPS = 25
DFL_CACHE = os.path.join(ROOT, "external", "idsse", "cache")
DFL_LENGTH, DFL_WIDTH = 105.0, 68.0
SEED = 2027
# From Step 9 on, every experiment uses 1 m of position jitter (D31): it matches the close-control
# share of our real tracking, and it halved an apparently clean +0.040 gain to an honest +0.020.
BENCH_JITTER = 1.0
# Lever 1 (D39): in realistic tracking the ball goes missing far more during stoppages. Measured on
# the Metrica sample games (external): missing on 78.6% of dead frames vs 15.6% of live frames (5x);
# our first Bundesliga blackouts only reached 1.2x. DFL blackouts are placed to match Metrica.
METRICA_MISSING = (0.786, 0.156)   # (dead, live)
DEAD_LINK_WEIGHT = 25.0
# Only ~half of wrong-team episodes are DETECTABLE (a team visibly at 12+) because ~1.5 players are
# missing per frame anyway; our 10.8% is the detectable share. The simulated share is scaled so the
# bench shows the same DETECTABLE share as our tracking (calibrated on tracking statistics only).
WRONG_TEAM_DETECTION_FACTOR = 2.2             # how much more likely a blackout starts on a dead frame


def _mean_player_speed(ps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(ps, axis=1)


# ---------------------------------------------------------------------------------------------
# 1. Bundesliga halves in our shape
# ---------------------------------------------------------------------------------------------

METRICA_CACHE = os.path.join(ROOT, "external", "metrica", "cache")


def load_metrica_halves():
    """Metrica sample games (organisers confirmed use), already in our shape by metrica_parse.py."""
    halves = []
    for path in sorted(glob.glob(os.path.join(METRICA_CACHE, "METRICA-G*.npz"))):
        d = np.load(path)
        for h in (1, 2):
            p = "h{}_".format(h)
            halves.append(dict(match=os.path.basename(path)[:-4], half=h, source="metrica",
                               **{k: d[p + k] for k in ("px", "py", "ps", "team", "bx", "by", "label")}))
    return halves


def load_dfl_halves():
    """[{match, half, px, py, ps, team, bx, by, label}] for every IDSSE half."""
    halves = []
    for path in sorted(glob.glob(os.path.join(DFL_CACHE, "DFL-MAT-*.npz"))):
        d = np.load(path)
        for h in (1, 2):
            p = "h{}_".format(h)
            halves.append(dict(
                match=os.path.basename(path)[:-4], half=h, source="dfl",
                px=d[p + "px"] + DFL_LENGTH / 2, py=d[p + "py"] + DFL_WIDTH / 2,
                ps=d[p + "ps"] / 3.6,                       # km/h -> m/s
                team=d[p + "team"],
                bx=d[p + "bx"] + DFL_LENGTH / 2, by=d[p + "by"] + DFL_WIDTH / 2,
                label=d[p + "label"],
            ))
    return halves


# ---------------------------------------------------------------------------------------------
# 2. Our match's mess, measured from our tracking only
# ---------------------------------------------------------------------------------------------

def our_mess_profile():
    """Gap lengths, missing rate by players' speed, and player visibility -- from OUR tracking."""
    t = load_arrays()
    seen = ~np.isnan(t["bx"])
    speed = _mean_player_speed(np.where(~np.isnan(t["px"]), t["ps"], np.nan))
    gaps, graded = [], np.zeros(len(seen), dtype=bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
        miss = ~seen[first:last + 1]
        edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        gaps += (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist()
    ok = graded & ~np.isnan(speed)
    edges = np.quantile(speed[ok], np.linspace(0, 1, 11)[1:-1])     # speed deciles
    bins = np.digitize(speed[ok], edges)
    rate = np.array([np.mean(~seen[ok][bins == b]) for b in range(10)])
    n_visible = (~np.isnan(t["px"][graded])).sum(axis=1).mean()
    # Wrong-team tracking (Phase 1 finding): one team shows >11 while the other is short.
    wt_lengths, wt_frames = [], 0
    for _, first, last in half_video_blocks("BAR-ATH"):
        pres = ~np.isnan(t["px"][first:last + 1])
        home, away = pres[:, t["team"] == 0].sum(1), pres[:, t["team"] == 1].sum(1)
        m = ((home > 11) & (away < 11)) | ((away > 11) & (home < 11))
        wt_frames += m.sum()
        e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        wt_lengths += (np.flatnonzero(e == -1) - np.flatnonzero(e == 1)).tolist()
    return dict(gap_lengths=np.array(gaps), speed_edges=edges, missing_rate=rate,
                target_missing=float(np.mean(~seen[graded])), players_visible=float(n_visible),
                wrong_team_lengths=np.array(wt_lengths), wrong_team_share=float(wt_frames / graded.sum()))


def unfreeze_ball(half):
    """The DFL data parks the ball during stoppages (99% of dead frames have a still ball; 2% in
    real automatic tracking). Replace that with what happens on a pitch: the nearest player collects
    the ball and carries it, and it travels to the restart spot over the last 2 s.
    Uses the DFL's own labels (external data) to find its stoppages -- never our ground truth."""
    bx, by = half["bx"].copy(), half["by"].copy()
    px, py = half["px"], half["py"]
    dead = half["label"] == "D"
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        spot_x, spot_y = half["bx"][b - 1], half["by"][b - 1]        # where play restarts
        d = np.hypot(px[a] - bx[a], py[a] - by[a])
        if np.all(np.isnan(d)):
            continue
        j = int(np.nanargmin(d))                                    # player who collects it
        cx = pd_ffill(px[a:b, j], bx[a])
        cy = pd_ffill(py[a:b, j], by[a])
        blend = np.clip((np.arange(b - a) - (b - a - 2 * FPS)) / (2 * FPS), 0, 1)
        bx[a:b] = (1 - blend) * cx + blend * spot_x
        by[a:b] = (1 - blend) * cy + blend * spot_y
    out = dict(half)
    out.update(bx=bx, by=by)
    return out


def pd_ffill(x, first):
    """Forward-fill NaN in x, starting from `first`."""
    x = x.copy()
    last = first
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = last
        else:
            last = x[i]
    return x


def add_jitter(half, sigma, rng, corr_frames=12):
    """Smooth position wobble of about `sigma` metres on ball and players, like automatic tracking
    (Step 9 robustness check). Noise is low-pass filtered so it drifts rather than flickers."""
    if sigma <= 0:
        return half
    out = dict(half)
    kernel = np.ones(corr_frames) / corr_frames

    def wobble(shape):
        z = rng.standard_normal(shape)
        z = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, z)
        return z / (z.std() + 1e-9) * sigma

    n = len(half["label"])
    out["bx"] = half["bx"] + wobble(n)
    out["by"] = half["by"] + wobble(n)
    out["px"] = half["px"] + wobble(half["px"].shape)
    out["py"] = half["py"] + wobble(half["py"].shape)
    return out


def add_wrong_team(half, profile, rng):
    """Copy our tracking's wrong-team episodes (D45): for each episode, one visible player's positions
    move to a new column labelled with the OTHER team. One episode at a time; lengths and share of
    frames drawn from our match's tracking."""
    px, py, ps, team = half["px"], half["py"], half["ps"], half["team"]
    n = px.shape[0]
    busy = np.zeros(n, dtype=bool)
    new_x, new_y, new_s, new_t = [], [], [], []
    px, py, ps = px.copy(), py.copy(), ps.copy()
    attempts = 0
    while busy.mean() < profile["wrong_team_share"] * WRONG_TEAM_DETECTION_FACTOR and attempts < 20000:
        attempts += 1
        L = int(rng.choice(profile["wrong_team_lengths"]))
        a = int(rng.integers(0, max(1, n - L)))
        b = a + L
        if busy[max(0, a - 1):min(n, b + 1)].any():
            continue
        visible = np.flatnonzero(~np.isnan(px[a:b]).any(axis=0))
        if visible.size == 0:
            continue
        j = int(rng.choice(visible))
        cx, cy, cs = (np.full(n, np.nan) for _ in range(3))
        cx[a:b], cy[a:b], cs[a:b] = px[a:b, j], py[a:b, j], ps[a:b, j]
        px[a:b, j] = py[a:b, j] = ps[a:b, j] = np.nan
        new_x.append(cx); new_y.append(cy); new_s.append(cs); new_t.append(1 - team[j])
        busy[a:b] = True
    out = dict(half)
    if new_x:
        out.update(px=np.column_stack([px] + new_x), py=np.column_stack([py] + new_y),
                   ps=np.column_stack([ps] + new_s), team=np.concatenate([team, np.array(new_t, dtype=team.dtype)]))
    return out


def degrade(half, profile, seed, jitter=0.0, dead_link=False, wrong_team=False):
    """Copy of a half with an unfrozen ball, ball blackouts and player dropouts like our tracking."""
    rng = np.random.default_rng(seed)
    metrica = half.get("source") == "metrica"
    half = add_jitter(half if metrica else unfreeze_ball(half), jitter, np.random.default_rng(seed + 7))
    out = dict(half)
    n = len(half["label"])

    # Ball blackouts: draw gap lengths from our distribution, place starts more often where
    # players are slow (weights = our missing rate in that speed decile), until our rate is hit.
    speed = _mean_player_speed(half["ps"])
    bins = np.digitize(np.nan_to_num(speed, nan=0.0), profile["speed_edges"])
    weights = profile["missing_rate"][bins]
    target = profile["target_missing"]
    mask = np.zeros(n, dtype=bool)
    attempts = 0
    if dead_link and not metrica:
        # Stoppages: the ball is lost soon after play stops and found just before the restart,
        # covering ~78.6% of each dead stretch on average (Metrica's rate).
        dead_frames = half["label"] == "D"
        edges = np.diff(np.concatenate(([0], dead_frames.astype(np.int8), [0])))
        cut = 1 - METRICA_MISSING[0]                     # mean uncovered share, split over both ends
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            L = b - a
            s0 = int(rng.uniform(0, cut) * L)
            s1 = int(rng.uniform(0, cut) * L)
            mask[a + s0:b - s1] = True
        # Live play: separate gaps (our gap lengths), kept inside live stretches, up to 15.6%.
        live = ~dead_frames
        live_idx = np.flatnonzero(live)
        while live_idx.size and mask[live].mean() < METRICA_MISSING[1] and attempts < 100000:
            attempts += 1
            length = int(rng.choice(profile["gap_lengths"]))
            start = int(live_idx[rng.integers(live_idx.size)])
            stop = start
            while stop < n and live[stop] and stop - start < length:
                stop += 1
            if mask[max(0, start - 1):min(n, stop + 1)].any():
                continue
            mask[start:stop] = True
    else:
        cdf = np.cumsum(weights) / weights.sum()
        while not metrica and mask.mean() < target and attempts < 100000:
            attempts += 1
            length = int(rng.choice(profile["gap_lengths"]))
            start = int(np.searchsorted(cdf, rng.random()))
            stop = min(n, start + length)
            if mask[max(0, start - 1):min(n, stop + 1)].any():
                continue  # keep gaps separate, as in the real data
            mask[start:stop] = True
    out["bx"] = np.where(mask, np.nan, half["bx"])
    out["by"] = np.where(mask, np.nan, half["by"])

    # Player dropouts: remove short runs per player until ~our visible count is reached.
    px, py, ps = half["px"].copy(), half["py"].copy(), half["ps"].copy()
    present = ~np.isnan(px)
    target = max(0.0, 1 - profile["players_visible"] / max(present.sum(axis=1).mean(), 1e-9))
    drop = np.zeros_like(present)
    while (drop & present).sum() < target * present.sum():
        j = rng.integers(px.shape[1])
        start = int(rng.integers(n))
        drop[start:start + int(rng.exponential(2 * FPS)) + 1, j] = True
    px[drop], py[drop], ps[drop] = np.nan, np.nan, np.nan
    out.update(px=px, py=py, ps=ps, ball_missing=float(np.mean(np.isnan(out["bx"]))))
    if wrong_team:
        out = add_wrong_team(out, profile, np.random.default_rng(seed + 13))
    return out


CONTROL_STYLE = dict(max_s=3.0, control_dist=1.0, control_frames=10)


def control_style_labels(half, max_s=3.0, control_dist=1.0, control_frames=10):
    """Convert DFL's touch-style possession labels to control style (D35).

    An opponent spell sandwiched between two possessions of the same team, with no stoppage in
    between, is merged back when no player of that team had the ball under control during it
    (control = within control_dist of the ball for >= control_frames consecutive frames, judged on
    the CLEAN DFL positions). Deflections, 50/50s and ricochets merge; a quick interception
    followed by control survives. This is the Opta / challenge definition (CHALLENGE.md, D16).
    """
    lab = half["label"].copy()
    d = np.hypot(half["px"] - half["bx"][:, None], half["py"] - half["by"][:, None])
    breaks = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(lab)]))
    segs = [(lab[a], a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    merged = 0
    for i in range(1, len(segs) - 1):
        (pl, _, _), (l, a, b), (nl, _, _) = segs[i - 1], segs[i], segs[i + 1]
        if l == "D" or pl == "D" or pl != nl or pl == l or (b - a) > max_s * FPS:
            continue
        side = 0 if l == "H" else 1
        close = np.nan_to_num(d[a:b][:, half["team"] == side], nan=np.inf).min(axis=1) < control_dist
        run = best = 0
        for c in close:
            run = run + 1 if c else 0
            best = max(best, run)
        if best < control_frames:
            lab[a:b] = pl
            merged += 1
    return lab, merged


def bench(profile=None, jitter=0.0, labels="dfl", sources=("dfl",), dead_link=False, wrong_team=False):
    """All degraded IDSSE halves, deterministic. jitter = position wobble in metres (0 = none).
    labels = 'dfl' (as published, touch style) or 'control' (converted to control style, D35)."""
    profile = profile or our_mess_profile()
    out = []
    pool = (load_dfl_halves() if "dfl" in sources else []) + (load_metrica_halves() if "metrica" in sources else [])
    for k, h in enumerate(pool):
        if labels == "control" and h.get("source") == "dfl":
            h = dict(h)
            h["label_dfl"] = h["label"]
            h["label"], h["n_merged"] = control_style_labels(h, **CONTROL_STYLE)
        out.append(degrade(h, profile, SEED + 100 * k, jitter, dead_link, wrong_team))
    return out, profile


# ---------------------------------------------------------------------------------------------
# 3. Official scoring
# ---------------------------------------------------------------------------------------------

def score(halves, predictions):
    """Official metrics over a list of halves. Halves are joined with one ungraded X frame so no
    possession can straddle two halves -- exactly how the scorer treats halftime."""
    gt, pred = [], []
    for h, p in zip(halves, predictions):
        gt += list(h["label"]) + ["X"]
        pred += list(p) + ["X"]
    return pm.evaluate(pm.label_codes(np.array(gt)), pm.label_codes(np.array(pred)))
