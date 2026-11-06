"""Layer 1: football features per frame, computed the same way for any half.

A half is a dict with px, py, ps (players: metres from the corner, m/s, NaN when not seen),
team (0 home / 1 away per player column), bx, by (ball, NaN when not seen). Works identically on
the Bundesliga test bench and on our match.

Everything here comes from tracking only. Features may look both backwards and forwards in time
(we label a finished match, not a live feed).
"""
import warnings

import numpy as np
import pandas as pd

FPS = 25
LENGTH, WIDTH = 105.0, 68.0
WALKING = 1.5  # m/s


def _roll(x, seconds, how="mean", where="centre"):
    """Rolling statistic over a window, ignoring NaN. where = centre | past | future."""
    w = max(1, int(seconds * FPS))
    s = pd.Series(x)
    if where == "future":
        s = s[::-1]
    r = getattr(s.rolling(w, min_periods=1, center=(where == "centre")), how)()
    return (r[::-1] if where == "future" else r).to_numpy()


def _run_lengths(mask):
    """For each frame: length (frames) of the True-run it sits in, frames since its start and until its end."""
    n = len(mask)
    length, since, until = np.zeros(n), np.zeros(n), np.zeros(n)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        length[a:b] = b - a
        since[a:b] = np.arange(b - a)
        until[a:b] = np.arange(b - a)[::-1]
    return length, since, until


def _fill(x, backwards=False):
    s = pd.Series(x)
    return (s.bfill() if backwards else s.ffill()).to_numpy()


def blackout_location_features(h):
    """Where was the ball LAST seen and where is it NEXT seen? (refinement #2)

    A blackout that starts with the ball by a touchline and ends with it at the touchline is
    almost certainly a throw-in; same idea for corner flags, the 6-yard box and the centre spot.
    On visible frames, 'last' and 'next' are the frame itself.
    """
    bx, by = h["bx"], h["by"]
    f = {}
    for name, back in (("last", False), ("next", True)):
        x, y = _fill(bx, back), _fill(by, back)
        f[name + "_seen_to_touchline"] = np.minimum(y, WIDTH - y)
        f[name + "_seen_to_goalline"] = np.minimum(x, LENGTH - x)
        f[name + "_seen_to_corner"] = np.hypot(np.minimum(x, LENGTH - x), np.minimum(y, WIDTH - y))
        f[name + "_seen_to_centre"] = np.hypot(x - LENGTH / 2, y - WIDTH / 2)
        f[name + "_seen_out"] = np.hypot(np.maximum(0, np.maximum(-x, x - LENGTH)),
                                         np.maximum(0, np.maximum(-y, y - WIDTH)))
    lx, ly, nx, ny = _fill(bx), _fill(by), _fill(bx, True), _fill(by, True)
    f["blackout_ball_moved"] = np.hypot(nx - lx, ny - ly)
    return pd.DataFrame(f)


def attack_direction(h):
    """+1 if the home team attacks towards x = LENGTH in this half, else -1.

    Found by the algorithm: the team whose average position over the half is nearer x = 0 defends
    that goal. No frame or constant from the video (rule R7)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home = np.nanmean(np.where(h["team"] == 0, h["px"], np.nan))
        away = np.nanmean(np.where(h["team"] == 1, h["px"], np.nan))
    return 1.0 if home < away else -1.0


def team_features(h):
    """Clues for 'which team has the ball' (rules R3, R4), symmetric in the two teams.

    Positive values lean home, negative lean away, wherever a difference is taken."""
    px, py, ps, bx, by, team = h["px"], h["py"], h["ps"], h["bx"], h["by"], h["team"]
    home, away = team == 0, team == 1
    seen = ~np.isnan(bx)
    direction = attack_direction(h)
    f = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.hypot(px - bx[:, None], py - by[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        d_home = d[:, home].min(axis=1)
        d_away = d[:, away].min(axis=1)
        d_home[~seen] = np.nan
        d_away[~seen] = np.nan
        f["near_home"], f["near_away"] = d_home, d_away
        diff = d_away - d_home                                  # > 0: a home player is closer
        f["near_diff"] = diff
        for s in (0.5, 1, 3):
            f["near_diff_med{}s".format(s)] = _roll(diff, s, "median")
        for s in (1, 3):
            f["near_diff_past{}s".format(s)] = _roll(diff, s, "median", "past")
            f["near_diff_future{}s".format(s)] = _roll(diff, s, "median", "future")
        f["near_diff_last_seen"] = _fill(diff)                  # bridges blackouts
        f["near_diff_next_seen"] = _fill(diff, backwards=True)
        f["home_within5"] = (d[:, home] < 5).sum(axis=1).astype(float)
        f["away_within5"] = (d[:, away] < 5).sum(axis=1).astype(float)
        f["within5_diff"] = f["home_within5"] - f["away_within5"]
        f["within5_diff_med1s"] = _roll(np.where(seen, f["within5_diff"], np.nan), 1)

        # Ball position in the home team's attacking direction (+ = towards the away goal)
        ball_adv = (bx - LENGTH / 2) * direction
        f["ball_adv_home"] = ball_adv
        f["ball_adv_home_last_seen"] = _fill(ball_adv)
        step = np.full(len(bx), np.nan)
        step[5:] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) / (5 / FPS)
        f["ball_speed_med1s"] = _roll(step, 1, "median")       # alive-ball speeds match (D23)

        # Team shape: possession pushes a team up and spreads it; defending compacts it
        xrel = (px - LENGTH / 2) * direction                    # + = towards the away goal
        hx = np.nanmean(np.where(home, xrel, np.nan), axis=1)
        ax = np.nanmean(np.where(away, xrel, np.nan), axis=1)
        f["block_adv_home"] = np.nanmean(xrel, axis=1)          # whole block's position
        f["home_adv"], f["away_adv"] = hx, ax
        f["home_width"] = np.nanstd(np.where(home, py, np.nan), axis=1)
        f["away_width"] = np.nanstd(np.where(away, py, np.nan), axis=1)
        f["home_length"] = np.nanstd(np.where(home, px, np.nan), axis=1)
        f["away_length"] = np.nanstd(np.where(away, px, np.nan), axis=1)
        f["width_diff"] = f["home_width"] - f["away_width"]
        f["length_diff"] = f["home_length"] - f["away_length"]
        f["block_adv_home_3s"] = _roll(f["block_adv_home"], 3)
        f["width_diff_3s"] = _roll(f["width_diff"], 3)
        vel = np.full(len(bx), np.nan)
        vel[FPS:] = (f["block_adv_home"][FPS:] - f["block_adv_home"][:-FPS])
        f["block_moving_home"] = _roll(vel, 1)                  # block advancing towards away goal
        f["home_speed"] = np.nanmean(np.where(home, ps, np.nan), axis=1)
        f["away_speed"] = np.nanmean(np.where(away, ps, np.nan), axis=1)
        f["speed_diff_3s"] = _roll(f["home_speed"] - f["away_speed"], 3)

    length, since, until = _run_lengths(~seen)
    f["blackout_since_s"], f["blackout_until_s"] = since / FPS, until / FPS
    return pd.DataFrame(f)


CONTROL_DIST = 1.5  # m: a player this close to the ball is treated as on it
CROSS_THRESHOLD = 0.5  # m/s: a velocity crossing must pass this on the other side (hysteresis)


def turnover_features(h):
    """Clues for WHEN possession changes hands in open play (Step 9, rule R4).

    Signed so that + leans home and - leans away."""
    px, py, bx, by, team = h["px"], h["py"], h["bx"], h["by"], h["team"]
    seen = ~np.isnan(bx)
    f = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.hypot(px - bx[:, None], py - by[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        j = np.argmin(d, axis=1)
        dmin = d[np.arange(len(j)), j]
        sign = np.where(team[j] == 0, 1.0, -1.0)
        control = np.where(seen & (dmin < CONTROL_DIST), sign, 0.0)       # +1 home, -1 away, 0 none
        f["control"] = control
        for s in (0.5, 1, 2):
            f["control_past{}s".format(s)] = _roll(control, s, where="past")
            f["control_future{}s".format(s)] = _roll(control, s, where="future")
        f["control_change_1s"] = f["control_future1s"] - f["control_past1s"]
        f["control_change_2s"] = f["control_future2s"] - f["control_past2s"]
        f["control_change_abs_1s"] = np.abs(f["control_change_1s"])

        # Who last had close control, and who has it next (bridges passes and short blackouts)
        ctrl = np.where(control != 0, control, np.nan)
        last, nxt = _fill(ctrl), _fill(ctrl, backwards=True)
        f["last_controller"] = np.nan_to_num(last)
        f["next_controller"] = np.nan_to_num(nxt)
        f["controller_flip"] = np.nan_to_num(last * nxt)                 # -1: the ball changed team
        idx = np.arange(len(ctrl), dtype=float)
        last_t = _fill(np.where(control != 0, idx, np.nan))
        next_t = _fill(np.where(control != 0, idx, np.nan), backwards=True)
        f["since_control_s"] = (idx - last_t) / FPS
        f["until_control_s"] = (next_t - idx) / FPS

        # Ball speed and reception (ball slowing sharply next to a player); alive speeds match (D23)
        sp = np.full(len(bx), np.nan)
        sp[3:] = np.hypot(bx[3:] - bx[:-3], by[3:] - by[:-3]) / (3 / FPS)
        before = _roll(sp, 0.4, where="past")
        after = _roll(sp, 0.4, where="future")
        f["ball_speed_before"], f["ball_speed_after"] = before, after
        decel = before - after
        f["reception"] = np.where(seen & (dmin < CONTROL_DIST) & (decel > 3), sign, 0.0)
        f["reception_past1s"] = _roll(f["reception"], 1, "sum", "past")
        f["reception_future1s"] = _roll(f["reception"], 1, "sum", "future")

        # Shape flip: what changes over the next 3 s compared with the last 3 s
        direction = attack_direction(h)
        xrel = (px - LENGTH / 2) * direction
        width_diff = (np.nanstd(np.where(team == 0, py, np.nan), axis=1)
                      - np.nanstd(np.where(team == 1, py, np.nan), axis=1))
        block = np.nanmean(xrel, axis=1)
        f["width_flip_3s"] = _roll(width_diff, 3, where="future") - _roll(width_diff, 3, where="past")
        f["block_flip_3s"] = _roll(block, 3, where="future") - _roll(block, 3, where="past")
    return pd.DataFrame(f)


TOUCH_DIST = 2.5  # m: generous enough for ~1 m tracking wobble
DEBOUNCE_S = 0.6  # s: same-team touches closer than this are one touch


def touch_features(h):
    """Clues built from TOUCH EVENTS rather than single frames (Step 9c: interceptions).

    A touch = a run of frames with a player within TOUCH_DIST of the ball. Its team score is the
    average margin (nearest away distance minus nearest home distance) over the whole touch, so one
    wobbly frame cannot flip it. + leans home, - leans away."""
    px, py, bx, by, team = h["px"], h["py"], h["bx"], h["by"], h["team"]
    n = len(bx)
    seen = ~np.isnan(bx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.hypot(px - bx[:, None], py - by[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        dh, da = d[:, team == 0].min(axis=1), d[:, team == 1].min(axis=1)
    near = seen & (np.minimum(dh, da) < TOUCH_DIST)
    margin = np.clip(np.where(near, da - dh, 0.0), -5, 5)

    edges = np.diff(np.concatenate(([0], near.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    # Debounce: a wobbly ball hovering at the distance threshold chops one touch into many. Merge
    # touches by the same team separated by less than DEBOUNCE_S, as a human would count them.
    if len(starts) > 1:
        side = np.sign([margin[a:b].mean() for a, b in zip(starts, stops)])
        keep_s, keep_e = [starts[0]], [stops[0]]
        last_side = side[0]
        for a, b, s in zip(starts[1:], stops[1:], side[1:]):
            if s == last_side and a - keep_e[-1] < DEBOUNCE_S * FPS:
                keep_e[-1] = b
            else:
                keep_s.append(a)
                keep_e.append(b)
                last_side = s
        starts, stops = np.array(keep_s), np.array(keep_e)
    k = len(starts)
    score = np.array([margin[a:b].mean() for a, b in zip(starts, stops)]) if k else np.zeros(0)
    mid_x = np.array([np.nanmean(bx[a:b]) for a, b in zip(starts, stops)]) if k else np.zeros(0)
    mid_y = np.array([np.nanmean(by[a:b]) for a, b in zip(starts, stops)]) if k else np.zeros(0)
    sgn = np.sign(score)
    # streak: consecutive touches by the same team, counted forwards and backwards
    streak_f, streak_b = np.ones(k), np.ones(k)
    for i in range(1, k):
        if sgn[i] == sgn[i - 1] and sgn[i] != 0:
            streak_f[i] = streak_f[i - 1] + 1
    for i in range(k - 2, -1, -1):
        if sgn[i] == sgn[i + 1] and sgn[i] != 0:
            streak_b[i] = streak_b[i + 1] + 1

    # Map to frames: index of the previous touch (ended at or before t) and the next touch
    prev_i = np.searchsorted(stops, np.arange(n), side="right") - 1      # last touch whose stop <= t
    in_touch = np.zeros(n, dtype=int) - 1
    for i, (a, b) in enumerate(zip(starts, stops)):
        in_touch[a:b] = i
    next_i = np.searchsorted(starts, np.arange(n), side="right")         # first touch starting after t
    cur_or_prev = np.where(in_touch >= 0, in_touch, prev_i)
    cur_or_next = np.where(in_touch >= 0, in_touch, next_i)

    def take(arr, idx, default=0.0):
        out = np.full(n, default)
        ok = (idx >= 0) & (idx < k)
        out[ok] = arr[idx[ok]]
        return out

    f = {}
    f["touch_prev"] = take(score, cur_or_prev)
    f["touch_next"] = take(score, cur_or_next)
    f["touch_prev2"] = take(score, cur_or_prev - 1)
    f["touch_next2"] = take(score, cur_or_next + 1)
    f["touch_flip"] = -np.sign(f["touch_prev"]) * np.sign(f["touch_next"])          # +1 = changed team
    f["touch_flip_confirmed"] = f["touch_flip"] * (np.sign(f["touch_next"]) == np.sign(f["touch_next2"]))
    f["streak_before"] = take(streak_f, cur_or_prev) * np.sign(f["touch_prev"])
    f["streak_after"] = take(streak_b, cur_or_next) * np.sign(f["touch_next"])
    idx = np.arange(n, dtype=float)
    f["since_touch_s"] = (idx - take(stops.astype(float), cur_or_prev, np.nan)) / FPS
    f["until_touch_s"] = (take(starts.astype(float), cur_or_next, np.nan) - idx) / FPS
    # (pass length between touches was tried and dropped: median 4.2 m on the bench vs 7.9 m on our
    #  match -- it does not behave the same way, D23)
    touch_start = np.zeros(n)
    touch_start[starts] = 1.0                                        # debounced touches
    f["touches_10s"] = _roll(touch_start, 10, "sum")
    return pd.DataFrame(f)


def collective_features(h):
    """Team-level movement clues for turnovers inside ball blackouts (expert suggestion, Phase 2 #3).

    All are team aggregates, so robust to ~1 m jitter and to a single wrong-team player.
    + leans home, - leans away where a difference is taken."""
    px, py, team = h["px"], h["py"], h["team"]
    n = len(px)
    direction = attack_direction(h)
    home, away = team == 0, team == 1
    f = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # 1) team forward speed, each in its own attacking direction (m/s)
        hx = np.nanmean(np.where(home, px, np.nan), axis=1) * direction
        ax = -np.nanmean(np.where(away, px, np.nan), axis=1) * direction
        span = int(0.5 * FPS)
        vh, va = np.full(n, np.nan), np.full(n, np.nan)
        vh[span:] = (hx[span:] - hx[:-span]) / 0.5
        va[span:] = (ax[span:] - ax[:-span]) / 0.5
        vh, va = _roll(vh, 1), _roll(va, 1)
        vd = vh - va                                        # + : home surging relative to away
        f["home_fwd_speed"], f["away_fwd_speed"], f["fwd_speed_diff"] = vh, va, vd
        for s in (1, 2):
            f["fwd_flip_{}s".format(s)] = _roll(vd, s, where="future") - _roll(vd, s, where="past")
        # 2) velocity crossing with hysteresis: the difference must move clearly (past +/-0.5 m/s) to
        #    the other side -- a real surge, not jitter flicker around zero (D23 check)
        th = CROSS_THRESHOLD
        state = np.where(vd > th, 1.0, np.where(vd < -th, -1.0, np.nan))
        sgn = np.nan_to_num(_fill(state))
        cross = np.zeros(n, bool)
        cross[1:] = (sgn[1:] != sgn[:-1]) & (sgn[1:] != 0) & (sgn[:-1] != 0)
        idx = np.arange(n, dtype=float)
        last_c = _fill(np.where(cross, idx, np.nan))
        next_c = _fill(np.where(cross, idx, np.nan), backwards=True)
        f["since_cross_s"] = (idx - last_c) / FPS
        f["until_cross_s"] = (next_c - idx) / FPS
        f["cross_dir_last"] = _fill(np.where(cross, sgn, np.nan))
        f["cross_dir_next"] = _fill(np.where(cross, sgn, np.nan), backwards=True)

        # 3) inferred ball: least-squares intersection of player movement lines, weighted by speed
        vx = np.full(px.shape, np.nan)
        vy = np.full(px.shape, np.nan)
        vx[5:], vy[5:] = (px[5:] - px[:-5]) / 0.2, (py[5:] - py[:-5]) / 0.2
        sp = np.hypot(vx, vy)
        w = np.where((sp > 2.0) & ~np.isnan(sp), sp, 0.0)
        ux, uy = np.nan_to_num(vx / np.where(sp > 0, sp, 1)), np.nan_to_num(vy / np.where(sp > 0, sp, 1))
        pxn, pyn = np.nan_to_num(px), np.nan_to_num(py)
        # projector onto the normal of each line: P = I - u u^T
        a11, a12, a22 = 1 - ux * ux, -ux * uy, 1 - uy * uy
        lam = 0.5                                           # weak pull towards the players' centroid
        cx = np.nanmean(px, axis=1)
        cy = np.nanmean(py, axis=1)
        A11 = (w * a11).sum(1) + lam
        A12 = (w * a12).sum(1)
        A22 = (w * a22).sum(1) + lam
        B1 = (w * (a11 * pxn + a12 * pyn)).sum(1) + lam * np.nan_to_num(cx)
        B2 = (w * (a12 * pxn + a22 * pyn)).sum(1) + lam * np.nan_to_num(cy)
        det = A11 * A22 - A12 * A12
        ibx = np.clip((A22 * B1 - A12 * B2) / det, -5, LENGTH + 5)
        iby = np.clip((A11 * B2 - A12 * B1) / det, -5, WIDTH + 5)
        ibx, iby = _roll(ibx, 0.4, "median"), _roll(iby, 0.4, "median")
        d = np.hypot(px - ibx[:, None], py - iby[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        f["infer_near_diff"] = d[:, away].min(axis=1) - d[:, home].min(axis=1)
        # 4) pressing collapse around the inferred ball
        nh, na = (d[:, home] < 5).sum(1).astype(float), (d[:, away] < 5).sum(1).astype(float)
        f["infer_within5_diff"] = nh - na
        f["infer_crowd"] = nh + na
        f["infer_crowd_change_1s"] = _roll(nh + na, 1, where="future") - _roll(nh + na, 1, where="past")
        hw = np.nanstd(np.where(home, py, np.nan), axis=1)
        aw = np.nanstd(np.where(away, py, np.nan), axis=1)
        f["home_width_change_1s"] = _roll(hw, 1, where="future") - _roll(hw, 1, where="past")
        f["away_width_change_1s"] = _roll(aw, 1, where="future") - _roll(aw, 1, where="past")
    return pd.DataFrame(f)


def dead_ball_features(h):
    """DataFrame of dead-ball features (rules R2, R3), one row per frame."""
    px, py, ps, bx, by = h["px"], h["py"], h["ps"], h["bx"], h["by"]
    seen = ~np.isnan(bx)
    f = {}

    # Ball position relative to the lines and set-piece spots
    out_x = np.maximum(0, np.maximum(-bx, bx - LENGTH))
    out_y = np.maximum(0, np.maximum(-by, by - WIDTH))
    f["ball_seen"] = seen.astype(float)
    f["ball_out_dist"] = np.hypot(out_x, out_y)                        # 0 inside, NaN if unseen
    f["ball_to_touchline"] = np.minimum(by, WIDTH - by)
    f["ball_to_goalline"] = np.minimum(bx, LENGTH - bx)
    f["ball_to_corner"] = np.hypot(np.minimum(bx, LENGTH - bx), np.minimum(by, WIDTH - by))
    f["ball_to_centre"] = np.hypot(bx - LENGTH / 2, by - WIDTH / 2)
    f["ball_in_6yd_box"] = ((np.minimum(bx, LENGTH - bx) < 5.5) & (np.abs(by - WIDTH / 2) < 9.16)).astype(float)
    f["ball_in_6yd_box"][~seen] = np.nan
    f["ball_out_recent"] = _roll(f["ball_out_dist"] > 0.5, 3, "max", "past")   # was it out in the last 3 s?

    # Ball movement (D11: speed from positions)
    step = np.full(len(bx), np.nan)
    step[5:] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) / (5 / FPS)
    f["ball_speed"] = step
    f["ball_speed_med1s"] = _roll(step, 1, "median")
    f["ball_speed_med3s"] = _roll(step, 3, "median")

    # Ball blackouts
    length, since, until = _run_lengths(~seen)
    f["blackout_len_s"] = length / FPS
    f["blackout_since_s"] = since / FPS
    f["blackout_until_s"] = until / FPS
    f["ball_seen_share_10s"] = _roll(seen.astype(float), 10)

    # Players' movement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(ps, axis=1)
        walking = np.nanmean(np.where(np.isnan(ps), np.nan, ps < WALKING), axis=1)
    f["players_speed"] = mean_speed
    for s in (1, 3, 6):
        f["players_speed_{}s".format(s)] = _roll(mean_speed, s)
    f["players_speed_past3s"] = _roll(mean_speed, 3, where="past")
    f["players_speed_future3s"] = _roll(mean_speed, 3, where="future")
    f["players_walking_3s"] = _roll(walking, 3)
    f["players_visible"] = (~np.isnan(px)).sum(axis=1).astype(float)

    # Shape: kick-offs put each team in its own half; nearest player to the ball
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        home_x = np.nanmean(np.where(h["team"] == 0, px, np.nan), axis=1)
        away_x = np.nanmean(np.where(h["team"] == 1, px, np.nan), axis=1)
        f["team_x_gap"] = np.abs(home_x - away_x)
        d = np.hypot(px - bx[:, None], py - by[:, None])
        f["nearest_player_dist"] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    f["nearest_player_dist"][~seen] = np.nan
    f["nearest_player_dist_med1s"] = _roll(f["nearest_player_dist"], 1, "median")
    return pd.DataFrame(f)
