"""Download and prepare the external training data (not redistributed in this repository).

    python fetch_external.py

1. IDSSE / DFL Bundesliga open data (7 matches, CC-BY 4.0), figshare DOI 10.6084/m9.figshare.28196177
   Bassek, Rein, Weber & Memmert (2025). An integrated dataset of spatiotemporal and event data in
   elite soccer. Scientific Data. -> external/idsse/   (~2.6 GB, checksums verified)
2. Metrica Sports open sample data (3 games), github.com/metrica-sports/sample-data
   -> external/metrica/data/   (~180 MB)
Then parses both into compact arrays (external/idsse/cache, external/metrica/cache).
Already-downloaded files with a matching checksum are skipped.
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
IDSSE_DIR = os.path.join(ROOT, "external", "idsse")
METRICA_DIR = os.path.join(ROOT, "external", "metrica")
FIGSHARE_ARTICLE = "https://api.figshare.com/v2/articles/28196177"
METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/"
METRICA_FILES = [
    "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "data/Sample_Game_3/Sample_Game_3_events.json",
    "data/Sample_Game_3/Sample_Game_3_metadata.xml",
    "data/Sample_Game_3/Sample_Game_3_tracking.txt",
]
HEADERS = {"User-Agent": "offsidefc-possession/1.0"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as r, open(dest + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(dest + ".part", dest)


def fetch_idsse():
    req = urllib.request.Request(FIGSHARE_ARTICLE, headers=HEADERS)
    files = json.load(urllib.request.urlopen(req))["files"]
    for f in files:
        dest = os.path.join(IDSSE_DIR, f["name"])
        expected = f.get("supplied_md5") or f.get("computed_md5")
        if os.path.exists(dest) and md5(dest) == expected:
            print("  ok (cached)", f["name"])
            continue
        print("  downloading", f["name"], "({:.0f} MB)".format(f["size"] / 1e6), flush=True)
        download(f["download_url"], dest)
        if md5(dest) != expected:
            sys.exit("checksum mismatch for " + f["name"])


def fetch_metrica():
    for rel in METRICA_FILES:
        dest = os.path.join(METRICA_DIR, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print("  ok (cached)", rel)
            continue
        print("  downloading", rel, flush=True)
        download(METRICA_RAW + rel, dest)


if __name__ == "__main__":
    print("IDSSE / DFL Bundesliga open data (CC-BY 4.0)")
    fetch_idsse()
    print("Metrica Sports open sample data")
    fetch_metrica()
    print("parsing ...", flush=True)
    for script in ("analysis/idsse_parse.py", "analysis/metrica_parse.py"):
        subprocess.run([sys.executable, os.path.join(ROOT, script)], check=True)
    print("external data ready")
