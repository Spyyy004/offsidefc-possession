"""Parse Metrica's open sample data (3 games; organisers confirmed it may be used) into the same
compact arrays as the IDSSE matches, with CONTROL-STYLE labels built from Metrica's own events.

Source: github.com/metrica-sports/sample-data (acknowledged as the source, per its README).
Downloaded into external/metrica/ (git-ignored).

Labels per frame (rulebook R1-R4, D16, D17):
  possession = team of the latest controlled action (PASS, CARRY, RECOVERY, SET PIECE, SHOT);
               a loose ball stays with the last team in control (Opta-style, control-based)
  dead       = from BALL OUT, FAULT RECEIVED (at the foul) or a goal, until the next SET PIECE
Halves run from each period's first event (kick-off) to its last event.

    python analysis/metrica_parse.py
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "metrica", "data")
CACHE = os.path.join(ROOT, "external", "metrica", "cache")
LENGTH, WIDTH, FPS = 105.0, 68.0, 25
CONTROLLED = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}


def speeds(px, py):
    """m/s from positions over a centred 5-frame span (Metrica gives no speeds)."""
    sp = np.full(px.shape, np.nan)
    sp[2:-2] = np.hypot(px[4:] - px[:-4], py[4:] - py[:-4]) / (4 / FPS)
    return sp


def labels_from_events(ev, first, last):
    """ev: DataFrame with team ('H'/'A'), type, subtype, start, end (frames). Returns labels for frames first..last."""
    n = last - first + 1
    team = np.full(n, "", dtype="<U1")
    dead = np.zeros(n, dtype=bool)
    ev = ev.sort_values(["start", "end"]).reset_index(drop=True)
    for _, e in ev.iterrows():
        a = int(e.start) - first
        if e.type in CONTROLLED and 0 <= a < n:
            team[a:] = e.team
    # dead stretches: from a stopping event to the next set piece
    stops = ev[(ev.type == "BALL OUT") | (ev.type == "FAULT RECEIVED") |
               ((ev.type == "SHOT") & ev.subtype.fillna("").str.contains("GOAL"))]
    restarts = ev[ev.type == "SET PIECE"].start.to_numpy()
    for _, e in stops.iterrows():
        s = int(e.end if e.type == "SHOT" else e.start)
        nxt = restarts[restarts > s]
        if len(nxt):
            dead[max(0, s - first):max(0, int(nxt[0]) - first)] = True
    lab = np.where(dead, "D", np.where(team == "", "D", team))
    return lab


def save(name, title, halves):
    out = {"title": title}
    for k, h in enumerate(halves, 1):
        for key, val in h.items():
            out["h{}_{}".format(k, key)] = val
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, name + ".npz"), **out)
    print("saved", name, [len(h["label"]) for h in halves], "frames per half")


def game_csv(g):
    """Games 1 and 2: one tracking CSV per team (3 header rows), events CSV."""
    def read(side):
        path = os.path.join(SRC, "Sample_Game_{0}/Sample_Game_{0}_RawTrackingData_{1}_Team.csv".format(g, side))
        names = pd.read_csv(path, skiprows=2, nrows=0).columns
        df = pd.read_csv(path, skiprows=2)
        cols = [c for c in names if c.startswith("Player")]
        xs = df[cols].to_numpy() * LENGTH
        ys = df[[names[names.get_loc(c) + 1] for c in cols]].to_numpy() * WIDTH
        bx = df["Ball"].to_numpy() * LENGTH
        by = df[names[names.get_loc("Ball") + 1]].to_numpy() * WIDTH
        return df["Period"].to_numpy(), df["Frame"].to_numpy(), xs, ys, bx, by
    per, frame, hx, hy, bx, by = read("Home")
    _, _, ax, ay, _, _ = read("Away")
    px, py = np.hstack([hx, ax]), np.hstack([hy, ay])
    team = np.array([0] * hx.shape[1] + [1] * ax.shape[1], dtype=np.int8)
    ev = pd.read_csv(os.path.join(SRC, "Sample_Game_{0}/Sample_Game_{0}_RawEventsData.csv".format(g)))
    ev = pd.DataFrame({"team": ev.Team.map({"Home": "H", "Away": "A"}), "type": ev.Type, "subtype": ev.Subtype,
                       "period": ev.Period, "start": ev["Start Frame"], "end": ev["End Frame"]})
    return per, frame, px, py, team, bx, by, ev


def game_epts():
    """Game 3: EPTS tracking (player order changes with substitutions across 11 specs), JSON events."""
    meta = ET.parse(os.path.join(SRC, "Sample_Game_3/Sample_Game_3_metadata.xml")).getroot()
    team_ids = [t.get("id") for t in meta.iter("Team")]
    players = [p.get("id") for p in meta.iter("Player")]
    pteam = {p.get("id"): (0 if p.get("teamId") == team_ids[0] else 1) for p in meta.iter("Player")}
    chan = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    specs = []
    for s in meta.iter("DataFormatSpecification"):
        order = [chan[r.get("playerChannelId")] for r in s.iter("PlayerChannelRef")][::2]   # x refs
        specs.append((int(s.get("startFrame")), int(s.get("endFrame")), [players.index(p) for p in order]))
    lines = open(os.path.join(SRC, "Sample_Game_3/Sample_Game_3_tracking.txt")).read().splitlines()
    n = len(lines)
    frame = np.zeros(n, dtype=int)
    px = np.full((n, len(players)), np.nan)
    py = px.copy()
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    for i, line in enumerate(lines):
        f, pl, ball = line.split(":")
        f = int(f)
        frame[i] = f
        cols = next(c for a, b, c in specs if a <= f <= b)
        for col, grp in zip(cols, pl.split(";")):
            x, y = grp.split(",")
            if x != "NaN":
                px[i, col], py[i, col] = float(x) * LENGTH, float(y) * WIDTH
        x, y = ball.split(",")[:2]
        if x != "NaN":
            bx[i], by[i] = float(x) * LENGTH, float(y) * WIDTH
    team = np.array([pteam[p] for p in players], dtype=np.int8)
    d = json.load(open(os.path.join(SRC, "Sample_Game_3/Sample_Game_3_events.json")))["data"]
    ev = pd.DataFrame({"team": ["H" if e["team"]["id"] == team_ids[0] else "A" for e in d],
                       "type": [e["type"]["name"] for e in d],
                       "subtype": [(e.get("subtypes") or {}).get("name") if isinstance(e.get("subtypes"), dict) else None for e in d],
                       "period": [e["period"] for e in d],
                       "start": [e["start"]["frame"] for e in d], "end": [e["end"]["frame"] for e in d]})
    return None, frame, px, py, team, bx, by, ev


def build(name, title, per, frame, px, py, team, bx, by, ev):
    halves = []
    for p in (1, 2):
        e = ev[ev.period == p]
        first, last = int(e.start.min()), int(e.end.max())
        rows = (frame >= first) & (frame <= last)
        f = frame[rows]
        lab = labels_from_events(e, first, last)[f - first]
        sx, sy = px[rows], py[rows]
        halves.append(dict(label=lab, bx=bx[rows], by=by[rows], px=sx, py=sy, ps=speeds(sx, sy), team=team))
    save(name, title, halves)


if __name__ == "__main__":
    for g in (1, 2):
        build("METRICA-G{}".format(g), "Metrica sample game {}".format(g), *game_csv(g))
    build("METRICA-G3", "Metrica sample game 3", *game_epts())
