"""Fast reader for the raw ATD tracking file, returning plain numpy arrays.

Raw line format: "<frame_id>:<46 x (x,y,speed) separated by ;>:<ball x,y,speed>".
Missing observations are NaN. frame_id counts from 1; the arrays are indexed by VIDEO frame
(frame_id - 1), i.e. by submission row.

The parsed arrays are cached under data/cache/ (git-ignored with the rest of data/), so the
196 MB text file is parsed once.

    from analysis.atd_arrays import load_arrays
    t = load_arrays()            # dict of numpy arrays, positions in METRES
"""
import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

N_SLOTS = 46
CACHE_DIR = os.path.join(ROOT, "data", "cache")


def slot_teams(meta_path):
    """0 = home (first team in the metadata, Barcelona), 1 = away, per track slot."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def _parse(raw_path):
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN"], engine="c").to_numpy()
    expected = 1 + 3 * N_SLOTS + 3
    if table.shape[1] != expected:
        raise ValueError("expected {} columns per line, got {}".format(expected, table.shape[1]))
    return table


def load_arrays(game=GAME, source="tactical", data_dir=None, use_cache=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    cache = os.path.join(CACHE_DIR, "{}_{}_atd.npz".format(game, source))
    if use_cache and os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        return dict(np.load(cache))

    table = _parse(raw_path)
    length, width = GAMES[game]["pitch"]
    frame_id = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * N_SLOTS].reshape(-1, N_SLOTS, 3)
    ball = table[:, 1 + 3 * N_SLOTS:]

    n = int(frame_id.max())
    def blank(shape):
        return np.full(shape, np.nan)
    px, py, ps = blank((n, N_SLOTS)), blank((n, N_SLOTS)), blank((n, N_SLOTS))
    bx, by, bs = blank(n), blank(n), blank(n)
    row = frame_id - 1                      # video frame = submission row
    px[row], py[row], ps[row] = players[:, :, 0] * length, players[:, :, 1] * width, players[:, :, 2]
    bx[row], by[row], bs[row] = ball[:, 0] * length, ball[:, 1] * width, ball[:, 2]

    out = dict(px=px, py=py, ps=ps, bx=bx, by=by, bs=bs, team=slot_teams(meta_path),
               pitch=np.array([length, width]))
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **out)
    return out
