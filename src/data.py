"""The tracking as numpy arrays, parsed once and cached in shared/cache/.

    from src.data import tracking
    t = tracking()            # tactical; tracking("broadcast") for the other recording
    t["ball"]      (n_frames, 3)      x, y in 0-1 pitch units, speed; NaN where unseen
    t["players"]   (n_frames, slots, 3)  the same per track slot (46 tactical, 51 broadcast)
    t["team"]      (slots,)           0 home, 1 away, per slot
    t["pitch"]     (length_m, width_m)

Row i is video frame i (the ATD counts from 1). Parsing the 187 MB text file takes about a
minute; afterwards a load is a second or two. The cache key includes the raw file's size and
mtime, so a replaced data file is re-parsed rather than silently served stale.
"""

import os

import numpy as np

from harness import paths
from games import GAME, GAMES, require, source_frames
from starter.baseline import team_of_each_track

VALUES = 3  # x, y, speed per group
CHUNK = 20000


def _slots(raw_path):
    """Track slots per record, read off the first line: 46 tactical, 51 broadcast."""
    with open(raw_path) as handle:
        fields = handle.readline().replace(":", ",").replace(";", ",").split(",")
    return (len(fields) - 1) // VALUES - 1


def _parse(raw_path, n_frames):
    n_slots = _slots(raw_path)
    ball = np.full((n_frames, VALUES), np.nan)
    players = np.full((n_frames, n_slots, VALUES), np.nan)
    width = 1 + (n_slots + 1) * VALUES
    with open(raw_path) as handle:
        while True:
            lines = handle.readlines(CHUNK * 1000)
            if not lines:
                break
            text = "".join(lines).replace(":", ",").replace(";", ",")
            values = np.array(text.split(), dtype=object)
            rows = np.array([row.split(",") for row in values], dtype=object)
            if rows.ndim != 2 or rows.shape[1] != width:
                raise ValueError("unexpected ATD line layout in {}".format(raw_path))
            numbers = rows.astype(np.float64)
            index = numbers[:, 0].astype(np.int64) - 1
            players[index] = numbers[:, 1:1 + n_slots * VALUES].reshape(-1, n_slots, VALUES)
            ball[index] = numbers[:, 1 + n_slots * VALUES:]
    return ball, players


def tracking(source="tactical", game=GAME):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    stat = os.stat(raw_path)
    cache = os.path.join(paths.CACHE, "{}_{}_atd_{}_{}.npz".format(
        game, source, stat.st_size, stat.st_mtime_ns))
    if not os.path.isfile(cache):
        ball, players = _parse(raw_path, source_frames(source, game))
        tmp = cache + ".tmp.npz"
        np.savez(tmp, ball=ball, players=players)
        os.replace(tmp, cache)
    arrays = np.load(cache)
    return {
        "ball": arrays["ball"],
        "players": arrays["players"],
        "team": team_of_each_track(meta_path),
        "pitch": GAMES[game]["pitch"],
    }
