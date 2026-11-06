"""Parse IDSSE/DFL open positional data (CC-BY 4.0) into compact numpy arrays.

Source: Bassek et al. 2025, Scientific Data, figshare DOI 10.6084/m9.figshare.28196177.
Downloaded into external/idsse/ (git-ignored).

Each positions XML holds one <FrameSet> per person (and one for the ball) per half, each with
one <Frame N=... X=... Y=... S=.../> per 25 Hz frame. The ball frames also carry
  BallPossession  1 = home, 2 = away
  BallStatus      0 = dead,  1 = alive
which is the same recipe our ground truth uses (dead where the tracking says dead, otherwise the
team it says owns the ball).

Output per match: external/idsse/cache/<MATCH>.npz with, per half, frame-aligned arrays:
  label (H/A/D), bx, by (metres, origin at pitch centre), px, py, ps (players, NaN when absent),
  team (0 home / 1 away per player column).

    python analysis/idsse_parse.py            # parses every match not yet cached
"""
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "idsse")
CACHE = os.path.join(SRC, "cache")


def team_ids(match_id):
    info = glob.glob(os.path.join(SRC, "DFL_02_01_matchinformation_*{}.xml".format(match_id)))[0]
    general = ET.parse(info).getroot().find(".//General")
    return general.get("HomeTeamId"), general.get("GuestTeamId"), general.get("MatchTitle")


def parse(path, home_id, away_id):
    halves = {}  # section -> {"ball": {...}, "players": {person: (team, {N: (x,y,s)})}}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "FrameSet":
            continue
        section = el.get("GameSection")
        team = el.get("TeamId")
        h = halves.setdefault(section, {"ball": None, "players": {}})
        frames = el.findall("Frame")
        if team.upper() == "BALL":
            n = np.array([int(f.get("N")) for f in frames])
            h["ball"] = dict(
                n=n,
                x=np.array([float(f.get("X")) for f in frames]),
                y=np.array([float(f.get("Y")) for f in frames]),
                poss=np.array([int(f.get("BallPossession", 0)) for f in frames]),
                status=np.array([int(f.get("BallStatus", 0)) for f in frames]),
            )
        elif team in (home_id, away_id):
            side = 0 if team == home_id else 1
            h["players"][el.get("PersonId")] = (side, {
                int(f.get("N")): (float(f.get("X")), float(f.get("Y")), float(f.get("S", "nan")))
                for f in frames})
        # else: referees etc. -- ignored
        el.clear()

    out = {}
    for k, (section, h) in enumerate(sorted(halves.items(), key=lambda kv: kv[1]["ball"]["n"][0]), 1):
        b = h["ball"]
        n0, n = b["n"][0], len(b["n"])
        idx = b["n"] - n0
        label = np.where(b["status"] == 0, "D", np.where(b["poss"] == 1, "H", "A"))
        persons = sorted(h["players"])
        px = np.full((n, len(persons)), np.nan)
        py, ps = px.copy(), px.copy()
        for j, p in enumerate(persons):
            for fn, (x, y, s) in h["players"][p][1].items():
                i = fn - n0
                if 0 <= i < n:
                    px[i, j], py[i, j], ps[i, j] = x, y, s
        lab = np.full(n, "D", dtype="<U1")
        lab[idx] = label
        pre = "h{}_".format(k)
        out.update({pre + "label": lab, pre + "bx": b["x"], pre + "by": b["y"],
                    pre + "px": px, pre + "py": py, pre + "ps": ps,
                    pre + "team": np.array([h["players"][p][0] for p in persons], dtype=np.int8)})
    return out


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(SRC, "DFL_04_03_positions_raw_observed_*.xml"))):
        match_id = re.search(r"(DFL-MAT-\w+)\.xml$", path).group(1)
        if len(sys.argv) > 1 and match_id not in sys.argv[1:]:
            continue
        dest = os.path.join(CACHE, match_id + ".npz")
        if os.path.exists(dest):
            continue
        home, away, title = team_ids(match_id)
        print("parsing", match_id, title, "...", flush=True)
        np.savez_compressed(dest, title=title, **parse(path, home, away))
    print("done")
