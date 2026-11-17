"""Restart and long-ball features for the dead-ball model: the cases where it misses the edges of a
stoppage (a throw-in being walked to the line, a quick free kick) or calls a long ball dead.

    R, names = restart_frames(t)      (n_frames, n_features), NaN outside the graded halves

Tracking (prefix rs_):
    nearest players   distance of the nearest, 2nd and 5th nearest player to the ball, and how many
                      are within 9.15 m: at a restart the opponents stand off, in play they press
    walking carry     ball slow, close to a slow player: a thrower or a free-kick taker placing it
    strike from rest  the ball goes from still to fast within half a second; time since / until
    launch / landing  the ball's speed at its last sighting before a gap and its first after: a long
                      ball leaves fast, a dead ball stops or drifts out of view
Smart Tagging (prefix st_): time since / until the start and end of each tagged set-piece type, and
inside / since / until a LONG BALL phase. Using these makes the harness's `rstrt` diagnostic
(tagged set pieces vs dead -> alive transitions) circular.
"""

import numpy as np

from games import GAME, half_video_blocks
from harness.diagnostics import events
from src.team_features import _interpolate, _quiet, _since, _until, _window_mean

FPS = 25
INTERP_MAX = 12
CAP_S = 20.0
WALL_M = 9.15
CARRY_M = 1.5
CARRY_BALL_MS = 3.0
CARRY_PLAYER_MS = 2.5
REST_MS = 1.0
STRIKE_MS = 8.0
STRIKE_WINDOW = FPS // 2
SPEED_LAG = 2
SET_PIECE_TYPES = ("throw-in", "corner kick", "goal kick", "free kick", "kick off")


def _speed(x, y, length, width):
    v = np.full(len(x), np.nan)
    k = SPEED_LAG
    v[k:-k] = np.hypot((x[2 * k:] - x[:-2 * k]) * length, (y[2 * k:] - y[:-2 * k]) * width) * FPS / (2 * k)
    return v


def _carry_forward(values, valid):
    index = np.arange(len(values))
    last = np.maximum.accumulate(np.where(valid, index, -1))
    return np.where(last >= 0, values[np.maximum(last, 0)], np.nan)


def half_features(t, first, last):
    length, width = t["pitch"]
    ball = t["ball"][first:last + 1]
    players = t["players"][first:last + 1]
    px, py, ps = players[:, :, 0], players[:, :, 1], players[:, :, 2]
    bx, by = ball[:, 0], ball[:, 1]
    seen = ~np.isnan(bx)
    ix, iy = _interpolate(bx, INTERP_MAX), _interpolate(by, INTERP_MAX)
    known = ~np.isnan(ix)
    cols = {}

    d = np.hypot((px - ix[:, None]) * length, (py - iy[:, None]) * width)
    d_sorted = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
    for k, name in ((0, "d1"), (1, "d2"), (4, "d5")):
        v = d_sorted[:, k]
        cols["rs_" + name] = np.where(known & np.isfinite(v), np.log(v + 0.5), np.nan)
    cols["rs_n_wall"] = np.where(known, np.sum(d < WALL_M, axis=1), np.nan)

    speed = _speed(ix, iy, length, width)
    nearest = np.argmin(np.where(np.isnan(d), np.inf, d), axis=1)
    near_speed = ps[np.arange(len(ps)), nearest]
    walk = known & (d_sorted[:, 0] < CARRY_M) & (speed < CARRY_BALL_MS) & (near_speed < CARRY_PLAYER_MS)
    walk_f = np.where(known & ~np.isnan(speed), walk.astype(float), np.nan)
    cols["rs_walk"] = walk_f
    for w in (1.0, 3.0):
        k = int(w * FPS)
        cols["rs_walk_past{:g}".format(w)] = _window_mean(walk_f, -k, 0)
        cols["rs_walk_next{:g}".format(w)] = _window_mean(walk_f, 0, k)

    rest = np.nan_to_num(speed, nan=99.0) < REST_MS
    recent_rest = _window_mean(rest.astype(float), -STRIKE_WINDOW, 0) > 0
    strike = recent_rest & (np.nan_to_num(speed) > STRIKE_MS)
    cols["rs_since_strike"] = _since(strike, CAP_S)
    cols["rs_until_strike"] = _until(strike, CAP_S)

    # Speed at the last sighting before a gap, carried through it; and at the next one, carried back.
    raw_speed = np.where(seen, _speed(bx, by, length, width), np.nan)
    ok = ~np.isnan(raw_speed)
    launch = _carry_forward(raw_speed, ok)
    landing = _carry_forward(raw_speed[::-1], ok[::-1])[::-1]
    cols["rs_launch"] = np.where(seen, np.nan, np.log(launch + 0.1))
    cols["rs_landing"] = np.where(seen, np.nan, np.log(landing + 0.1))
    cols["rs_player_speed"] = _quiet(np.nanmean, ps, axis=1)
    names = sorted(cols)
    return np.column_stack([cols[c] for c in names]).astype(np.float32), names


def tag_columns(n):
    marks = {}
    long_in = np.zeros(n)
    for event in events():
        code = (event["code"] or "").strip().lower()
        start, stop = int(event["start_frame"]), int(event["end_frame"])
        if not 0 <= start < n:
            continue
        if code == "set pieces":
            kind = (event["tags"].get("Type") or "").strip().lower()
            kind = next((k for k in SET_PIECE_TYPES if kind.startswith(k)), None)
            for key in (kind, "any"):
                if key:
                    marks.setdefault((key, "start"), []).append(start)
                    marks.setdefault((key, "end"), []).append(min(stop, n - 1))
        elif code == "long ball":
            long_in[start:stop] = 1.0
            marks.setdefault(("long", "start"), []).append(start)
    cols = {"st_long_in": long_in}
    for (key, edge), frames in marks.items():
        m = np.zeros(n, dtype=bool)
        m[frames] = True
        tag = "st_{}_{}".format(key.replace(" ", "_").replace("-", "_"), edge)
        cols[tag + "_since"] = _since(m, CAP_S)
        cols[tag + "_until"] = _until(m, CAP_S)
    names = sorted(cols)
    return np.column_stack([cols[c] for c in names]).astype(np.float32), names


def restart_frames(t, game=GAME):
    n = len(t["ball"])
    R, names = None, None
    for _, first, last in half_video_blocks(game):
        X, names = half_features(t, first, last)
        if R is None:
            R = np.full((n, X.shape[1]), np.nan, dtype=np.float32)
        R[first:last + 1] = X
    T, tag_names = tag_columns(n)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    T[~graded] = np.nan
    return np.hstack([R, T]), names + tag_names
