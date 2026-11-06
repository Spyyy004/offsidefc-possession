"""Post-processing rule: last touch before the ball goes out (expert suggestion; Phase 2 #4a).

At each predicted stoppage that starts with the ball crossing (or within 1 m of) a line, the team
nearest the ball over the last 0.5 s before it went out is credited with possession for up to the
final MAX_S seconds before the stoppage, if it differs from the predicted team and the margin is clear.
Tracking only; no frame constants.
"""
import warnings

import numpy as np

FPS = 25
LENGTH, WIDTH = 105.0, 68.0


def apply_last_touch(label, h, near_line_m=1.0, look_s=2.0, avg_s=0.5, margin_m=1.0, max_s=1.0):
    lab = np.asarray(label).copy()
    bx, by, px, py, team = h["bx"], h["by"], h["px"], h["py"], h["team"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.hypot(px - bx[:, None], py - by[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        dh, da = d[:, team == 0].min(axis=1), d[:, team == 1].min(axis=1)
    to_line = np.minimum(np.minimum(bx, LENGTH - bx), np.minimum(by, WIDTH - by))   # < 0 = outside
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    changed = 0
    for i in b:
        if lab[i] != "D" or lab[i - 1] == "D":
            continue
        lo = max(0, i - int(look_s * FPS))
        seen = np.flatnonzero(~np.isnan(bx[lo:i]))
        if not len(seen):
            continue
        j = lo + seen[-1]                                   # last frame the ball was seen
        if not (to_line[j] < near_line_m):
            continue                                        # stoppage did not start at a line
        a = max(0, j - int(avg_s * FPS))
        m = np.nanmean(np.where(np.isfinite(da[a:j + 1] - dh[a:j + 1]), da[a:j + 1] - dh[a:j + 1], np.nan))
        if not np.isfinite(m) or abs(m) < margin_m:
            continue
        toucher = "H" if m > 0 else "A"
        if toucher == lab[i - 1]:
            continue
        start = max(i - int(max_s * FPS), i - (i - j) - int(avg_s * FPS))
        start = max(start, 0)
        seg_start = i - 1
        while seg_start > 0 and lab[seg_start - 1] == lab[i - 1]:
            seg_start -= 1
        start = max(start, seg_start + 1)                   # never erase the whole previous possession
        lab[start:i] = toucher
        changed += 1
    return lab, changed
