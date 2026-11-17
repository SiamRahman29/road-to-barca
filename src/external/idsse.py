"""Outside data: the DFL / IDSSE dataset (Bassek, Rein, Weber & Memmert, 2025, "An integrated
dataset of spatiotemporal and event data in elite soccer", Scientific Data 12, 195,
doi:10.1038/s41597-025-04505-y; CC BY 4.0). Seven Bundesliga / 2. Bundesliga 2022/23 matches with
official position data at 25 Hz and, per frame, the ball status (alive / dead) and possession.

    python -m src.external.idsse            download what is missing, parse, print aggregates
    m = match("J03WMX")                     the tracking in src/data's layout, plus labels

`match` returns a dict like src.data.tracking(): "ball" (n, 3) x, y in 0-1 pitch units and height
in metres, "players" (n, slots, 3) x, y, speed, "team" (slots,) 0 home / 1 away, "pitch", plus
"label" (n,) possession_metrics codes H / A / D (X between halves) and "halves" [(first, last)].
Both halves sit in one array with GAP blank frames between them. Not from the competition match.
"""

import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from harness import paths
import possession_metrics as pm

RAW = os.path.join(paths.SHARED, "external", "idsse")
CACHE = os.path.join(paths.CACHE, "external")
BASE_URL = "https://ndownloader.figshare.com/files/"
FILES = {  # figshare file ids: (match information, position data)
    "J03WMX": ("51643475", "51643514"),
    "J03WN1": ("51643472", "51643517"),
    "J03WPY": ("51643487", "51643526"),
    "J03WOH": ("51643478", "51643520"),
    "J03WQQ": ("51643484", "51643529"),
    "J03WOY": ("51643481", "51643523"),
    "J03WR9": ("51643490", "51643532"),
}
MATCHES = tuple(FILES)
FPS = 25
GAP = 250
SPEED_LAG = 2
SECTIONS = ("firstHalf", "secondHalf")


def download(match_id):
    os.makedirs(RAW, exist_ok=True)
    out = []
    for kind, file_id in zip(("info", "pos"), FILES[match_id]):
        path = os.path.join(RAW, "{}_{}.xml".format(match_id, kind))
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            urllib.request.urlretrieve(BASE_URL + file_id, path + ".part")
            os.replace(path + ".part", path)
        out.append(path)
    return out


def _info(path):
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    return general.get("HomeTeamId"), general.get("GuestTeamId"), (float(env.get("PitchX")),
                                                                 float(env.get("PitchY")))


def _parse(info_path, pos_path):
    home_id, away_id, (length, width) = _info(info_path)
    sets = {s: {"ball": None, "people": []} for s in SECTIONS}
    for _, element in ET.iterparse(pos_path, events=("end",)):
        if element.tag != "FrameSet":
            continue
        section, team_id = element.get("GameSection"), element.get("TeamId")
        if section in sets and (team_id in (home_id, away_id) or team_id.lower() == "ball"):
            frames = element.findall("Frame")
            n0 = np.array([int(f.get("N")) for f in frames])
            x = np.array([float(f.get("X")) for f in frames])
            y = np.array([float(f.get("Y")) for f in frames])
            if team_id.lower() == "ball":
                z = np.array([float(f.get("Z") or "nan") for f in frames])
                status = np.array([int(f.get("BallStatus")) for f in frames])
                owner = np.array([int(f.get("BallPossession")) for f in frames])
                sets[section]["ball"] = (n0, x, y, z, status, owner)
            else:
                sets[section]["people"].append((0 if team_id == home_id else 1, n0, x, y))
        element.clear()
    return sets, (length, width)


def _build(match_id):
    info_path, pos_path = download(match_id)
    sets, (length, width) = _parse(info_path, pos_path)
    # Each half gets its own player slots (substitutes differ); the features never look at slots.
    counts = [len(sets[s]["people"]) for s in SECTIONS]
    offsets = [0, counts[0]]
    slots = sum(counts)
    team = np.full(slots, -1, dtype=np.int8)
    blocks, halves, start = [], [], 0
    for h, s in enumerate(SECTIONS):
        n0, x, y, z, status, owner = sets[s]["ball"]
        first_n, n = n0.min(), n0.max() - n0.min() + 1
        ball = np.full((n, 3), np.nan)
        ball[n0 - first_n] = np.column_stack([x / length + 0.5, y / width + 0.5, z])
        label = np.full(n, pm.X, dtype=np.int8)
        label[n0 - first_n] = np.where(status != 1, pm.D, np.where(owner == 1, pm.H, pm.A))
        players = np.full((n, slots, 3), np.nan, dtype=np.float32)
        for k, (team_code, pn, px, py) in enumerate(sets[s]["people"]):
            j = offsets[h] + k
            players[pn - first_n, j, 0] = px / length + 0.5
            players[pn - first_n, j, 1] = py / width + 0.5
            team[j] = team_code
        cols = slice(offsets[h], offsets[h] + counts[h])
        v = np.full((n, counts[h]), np.nan, dtype=np.float32)
        dx = (players[2 * SPEED_LAG:, cols, 0] - players[:-2 * SPEED_LAG, cols, 0]) * length
        dy = (players[2 * SPEED_LAG:, cols, 1] - players[:-2 * SPEED_LAG, cols, 1]) * width
        v[SPEED_LAG:-SPEED_LAG] = np.hypot(dx, dy) * FPS / (2 * SPEED_LAG)
        players[:, cols, 2] = v
        blocks.append((ball, players, label))
        halves.append((start, start + n - 1))
        start += n + GAP
    total = start - GAP
    ball = np.full((total, 3), np.nan)
    players = np.full((total, slots, 3), np.nan, dtype=np.float32)
    label = np.full(total, pm.X, dtype=np.int8)
    for (b, p, l), (a, z) in zip(blocks, halves):
        ball[a:z + 1], players[a:z + 1], label[a:z + 1] = b, p, l
    return {"ball": ball, "players": players, "team": team, "pitch": np.array([length, width]),
            "label": label, "halves": np.array(halves)}


def controlled(label, min_frames):
    """DFL possession turned towards controlled possession: inside play, a team's run shorter than
    min_frames goes to the team before it (or after it, at the start of play), shortest first."""
    out = label.copy()
    while True:
        e = np.flatnonzero(np.diff(out)) + 1
        starts, ends = np.concatenate(([0], e)), np.concatenate((e, [len(out)]))
        codes = out[starts]
        team = (codes == pm.H) | (codes == pm.A)
        short = np.flatnonzero(team & (ends - starts < min_frames))
        # Only runs with a team neighbour on at least one side can be merged.
        fixable = [k for k in short if (k > 0 and team[k - 1]) or (k + 1 < len(codes) and team[k + 1])]
        if not fixable:
            return out
        k = min(fixable, key=lambda j: ends[j] - starts[j])
        if k > 0 and team[k - 1]:
            out[starts[k]:ends[k]] = codes[k - 1]
        else:
            out[starts[k]:ends[k]] = codes[k + 1]


def match(match_id):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "idsse_{}.npz".format(match_id))
    if not os.path.exists(path):
        np.savez(path, **_build(match_id))
    a = np.load(path)
    out = {k: a[k] for k in a.files}
    out["pitch"] = tuple(out["pitch"].tolist())
    out["halves"] = [tuple(h) for h in out["halves"].tolist()]
    return out


def main(ids):
    for match_id in ids or MATCHES:
        m = match(match_id)
        lab = m["label"] != pm.X
        live = (m["label"] == pm.H) | (m["label"] == pm.A)
        print("{}: {:.1f} min, players/frame {:.1f}, ball present {:.3f}, dead {:.3f}, home of live "
              "{:.3f}, slots {}".format(
                  match_id, lab.sum() / FPS / 60,
                  np.mean(np.sum(~np.isnan(m["players"][lab, :, 0]), axis=1)),
                  np.mean(~np.isnan(m["ball"][lab, 0])), np.mean(m["label"][lab] == pm.D),
                  np.mean(m["label"][live] == pm.H), len(m["team"])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
