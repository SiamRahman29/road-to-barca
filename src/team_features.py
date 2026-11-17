"""Who has the ball: per-frame features for a supervised team (H vs A) model.

    from src.team_features import frame_design
    F, names = frame_design(tracking())     (n_frames, n_features), NaN where unknown; graded
                                             halves only, other rows NaN

Everything is about a team or the ball, never a track slot (slots are tracklets, DATA.md). Within
a half every directional feature is oriented by src/features.home_sign, so positive means "towards
the away goal" and the model sees the same geometry in both halves.

Groups:
    ball      seen, oriented position, speed (from a short-gap interpolated track)
    near      nearest home / away player to the ball, their log ratio, counts within 2 / 5 m,
              from the seen ball and from the interpolated one
    memory    the nearest-team margin at the last and the next sighting, and how far away they are
    touches   src/possession.touch_events: time since / until the last home and away touch and carry
    shape     team centroids, depth, width, mean speed, centroid velocity; who is pushing up
    context   past and future window means of the margin and of "nearest within 2 m is home"
    projection where the ball is heading (Link & Hoernig 2017, "ball prediction"): the nearest home
              / away player to the ball's straight-line position PROJECT_S ahead, and how soon
              that line first passes within PATH_M of a home and of an away player
"""

import warnings

import numpy as np

from games import GAME, half_video_blocks
from src.features import home_sign
from src.possession import touch_events

FPS = 25
INTERP_MAX = 12          # frames: ball gaps up to this are filled linearly
NEAR_M = (2.0, 5.0)      # radii for the player counts around the ball
CLOSE_M = 2.0            # "nearest within this is home / away" for the context shares
MARGIN_CLIP = 3.0
EVENT_CAP_S = 10.0
SIGHT_CAP_S = 10.0
WINDOWS_S = (0.4, 1.0, 2.0, 4.0)
VEL_LAG = 12             # frames either side for the centroid velocity
CONTACT_M = (0.75, 1.25, 2.0)  # radii for the last / next contact of each team
PROJECT_S = (0.5, 1.0, 2.0)    # seconds ahead for the projected ball position
PATH_STEP_S = 0.2              # the ball's line is checked every this many seconds...
PATH_MAX_S = 2.0               # ...up to this far ahead
PATH_M = 2.0                   # a player this close to the line is where the ball is going


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return function(*args, **kwargs)


def _interpolate(x, max_gap):
    """Linear fill of NaN gaps of at most max_gap frames between two seen values."""
    out = x.copy()
    seen = ~np.isnan(x)
    idx = np.flatnonzero(seen)
    if len(idx) < 2:
        return out
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [x[a], x[b]])
    return out


def _window_mean(x, lo, hi):
    """NaN-aware mean of x[i+lo .. i+hi] inclusive, clipped to the array."""
    n = len(x)
    valid = ~np.isnan(x)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(n)
    a, b = np.clip(index + lo, 0, n), np.clip(index + hi + 1, 0, n)
    total = counts[b] - counts[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (sums[b] - sums[a]) / np.maximum(total, 1), np.nan)


def _since(event, cap):
    """Frames since the last True (0 on it), in seconds, capped."""
    index = np.arange(len(event))
    last = np.maximum.accumulate(np.where(event, index, -10 ** 9))
    return np.minimum((index - last) / FPS, cap)


def _until(event, cap):
    return _since(event[::-1], cap)[::-1]


def _carry_last(values, valid):
    """values at the last valid frame (inclusive), NaN before the first."""
    index = np.arange(len(values))
    last = np.maximum.accumulate(np.where(valid, index, -1))
    out = np.where(last >= 0, values[np.maximum(last, 0)], np.nan)
    return out


def _nearest(px, py, bx, by, team, length, width):
    dx = (px - bx[:, None]) * length
    dy = (py - by[:, None]) * width
    d = np.sqrt(dx * dx + dy * dy)
    d_home = _quiet(np.nanmin, d[:, team == 0], axis=1)
    d_away = _quiet(np.nanmin, d[:, team == 1], axis=1)
    counts = {}
    for r in NEAR_M:
        counts[r] = (np.sum(d[:, team == 0] <= r, axis=1).astype(float),
                     np.sum(d[:, team == 1] <= r, axis=1).astype(float))
    return d_home, d_away, counts


def half_features(t, first, last, events):
    length, width = t["pitch"]
    team = t["team"]
    ball = t["ball"][first:last + 1]
    players = t["players"][first:last + 1]
    px, py, ps = players[:, :, 0], players[:, :, 1], players[:, :, 2]
    sign = home_sign(px, team)
    n = len(ball)
    cols = {}

    bx, by = ball[:, 0], ball[:, 1]
    seen = ~np.isnan(bx)
    ix, iy = _interpolate(bx, INTERP_MAX), _interpolate(by, INTERP_MAX)
    iseen = ~np.isnan(ix)
    cols["seen"] = seen.astype(float)
    cols["iseen"] = iseen.astype(float)
    # Oriented ball position: + towards the away goal, metres from the centre.
    cols["ball_x"] = sign * (ix - 0.5) * length
    cols["ball_y"] = np.abs(iy - 0.5) * width
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[2:-2] = (ix[4:] - ix[:-4]) * length * FPS / 4
    vy[2:-2] = (iy[4:] - iy[:-4]) * width * FPS / 4
    cols["ball_vx"] = sign * vx
    cols["ball_speed"] = np.log(np.hypot(vx, vy) + 0.1)

    d_home, d_away, counts = _nearest(px, py, ix, iy, team, length, width)
    margin = np.clip(np.log((d_away + 1.0) / (d_home + 1.0)), -MARGIN_CLIP, MARGIN_CLIP)
    cols["d_home"] = np.log(d_home + 0.5)
    cols["d_away"] = np.log(d_away + 0.5)
    cols["margin"] = margin
    cols["margin_seen"] = np.where(seen, margin, np.nan)
    for r in NEAR_M:
        h, a = counts[r]
        cols["n_home_{:g}".format(r)] = np.where(iseen, h, np.nan)
        cols["n_away_{:g}".format(r)] = np.where(iseen, a, np.nan)
    close = np.minimum(d_home, d_away) <= CLOSE_M
    home_close = np.where(iseen, (close & (d_home < d_away)).astype(float), np.nan)
    away_close = np.where(iseen, (close & (d_away <= d_home)).astype(float), np.nan)
    cols["home_close"] = home_close
    cols["away_close"] = away_close

    # Contact: the last and next frame a player of each team was this close to the ball. The team
    # of the last contact is a proxy for the last touch, which owns the ball while it travels.
    for r in CONTACT_M:
        for name, d_own, d_other in (("H", d_home, d_away), ("A", d_away, d_home)):
            contact = iseen & (d_own <= r) & (d_own < d_other)
            cols["since_close{:g}_{}".format(r, name)] = _since(contact, EVENT_CAP_S)
            cols["until_close{:g}_{}".format(r, name)] = _until(contact, EVENT_CAP_S)
        cols["close{:g}_last_diff".format(r)] = (cols["since_close{:g}_A".format(r)]
                                                 - cols["since_close{:g}_H".format(r)])
        cols["close{:g}_next_diff".format(r)] = (cols["until_close{:g}_A".format(r)]
                                                 - cols["until_close{:g}_H".format(r)])
    # Projection: carry the ball on at its current velocity and ask whose player it is heading to.
    # The velocity is the same 4-frame difference as ball_vx; projections leave the pitch by at
    # most 5 % so a ball going out still points at the nearest player to where it goes out.
    ux = np.full(n, np.nan)
    uy = np.full(n, np.nan)
    ux[2:-2] = (ix[4:] - ix[:-4]) / 4
    uy[2:-2] = (iy[4:] - iy[:-4]) / 4

    def ahead(s):
        qx = np.clip(ix + ux * s * FPS, -0.05, 1.05)
        qy = np.clip(iy + uy * s * FPS, -0.05, 1.05)
        p_home, p_away, _ = _nearest(px, py, qx, qy, team, length, width)
        return p_home, p_away

    for s in PROJECT_S:
        p_home, p_away = ahead(s)
        cols["d_home_proj{:g}".format(s)] = np.log(p_home + 0.5)
        cols["d_away_proj{:g}".format(s)] = np.log(p_away + 0.5)
        cols["margin_proj{:g}".format(s)] = np.clip(np.log((p_away + 1.0) / (p_home + 1.0)),
                                                    -MARGIN_CLIP, MARGIN_CLIP)
    moving = ~np.isnan(ux)
    cap = PATH_MAX_S + PATH_STEP_S
    path = {"home": np.full(n, cap), "away": np.full(n, cap)}
    for step in range(1, int(round(PATH_MAX_S / PATH_STEP_S)) + 1):
        s = step * PATH_STEP_S
        p_home, p_away = ahead(s)
        for name, d in (("home", p_home), ("away", p_away)):
            first_hit = (path[name] == cap) & (d <= PATH_M)
            path[name][first_hit] = s
    for name in ("home", "away"):
        cols["path_{}".format(name)] = np.where(moving, path[name], np.nan)
    cols["margin_path"] = cols["path_away"] - cols["path_home"]

    # Ball acceleration at a few scales (touches), and the margin shifted in time.
    for k in (2, 4, 8):
        acc = np.full(n, np.nan)
        acc[k:-k] = np.hypot(ix[2 * k:] - 2 * ix[k:-k] + ix[:-2 * k],
                             iy[2 * k:] - 2 * iy[k:-k] + iy[:-2 * k]) * length * FPS * FPS / (k * k)
        cols["ball_acc{}".format(k)] = np.log(acc + 0.1)
    for lag in (-50, -25, -12, -6, 6, 12, 25, 50):
        shifted = np.full(n, np.nan)
        if lag > 0:
            shifted[:-lag] = margin[lag:]
        else:
            shifted[-lag:] = margin[:lag]
        cols["margin_lag{:+d}".format(lag)] = shifted

    # Memory: the margin at the last and next sighting, and how long ago / until.
    valid = ~np.isnan(margin)
    cols["margin_last"] = _carry_last(margin, valid)
    cols["margin_next"] = _carry_last(margin[::-1], valid[::-1])[::-1]
    cols["since_seen"] = _since(valid, SIGHT_CAP_S)
    cols["until_seen"] = _until(valid, SIGHT_CAP_S)

    # Touch events (video frames), split by team and kind.
    for who in ("H", "A"):
        for kind in ("touch", "carry"):
            mark = np.zeros(n, dtype=bool)
            frames = [f - first for f, w, k in events if w == who and k == kind and first <= f <= last]
            mark[frames] = True
            cols["since_{}_{}".format(kind, who)] = _since(mark, EVENT_CAP_S)
            cols["until_{}_{}".format(kind, who)] = _until(mark, EVENT_CAP_S)
            for w in (1.0, 3.0):
                k = int(w * FPS)
                cols["n_{}_{}_past{:g}".format(kind, who, w)] = _window_mean(mark.astype(float), -k, 0) * (k + 1)
                cols["n_{}_{}_next{:g}".format(kind, who, w)] = _window_mean(mark.astype(float), 0, k) * (k + 1)
    # Team of the last and the next event of any kind: +1 home, -1 away.
    ev = np.zeros(n)
    for f, w, k in events:
        if first <= f <= last and w:
            ev[f - first] = 1.0 if w == "H" else -1.0
    has = ev != 0
    cols["last_event_team"] = _carry_last(ev, has)
    cols["next_event_team"] = _carry_last(ev[::-1], has[::-1])[::-1]

    # Team shape.
    for code, name in ((0, "home"), (1, "away")):
        x = px[:, team == code]
        y = py[:, team == code]
        cx = _quiet(np.nanmean, x, axis=1)
        cols["{}_cx".format(name)] = sign * (cx - 0.5) * length
        cols["{}_width".format(name)] = _quiet(np.nanstd, y, axis=1) * width
        cols["{}_depth".format(name)] = _quiet(np.nanstd, x, axis=1) * length
        cols["{}_speed".format(name)] = _quiet(np.nanmean, ps[:, team == code], axis=1)
        vel = np.full(n, np.nan)
        vel[VEL_LAG:-VEL_LAG] = sign * (cx[2 * VEL_LAG:] - cx[:-2 * VEL_LAG]) * length * FPS / (2 * VEL_LAG)
        cols["{}_vel".format(name)] = vel
        cols["{}_ball_dx".format(name)] = cols["ball_x"] - cols["{}_cx".format(name)]
    cols["shape_width"] = cols["home_width"] - cols["away_width"]
    cols["shape_depth"] = cols["home_depth"] - cols["away_depth"]
    cols["shape_vel"] = cols["home_vel"] + cols["away_vel"]

    # Context windows.
    for w in WINDOWS_S:
        k = int(w * FPS)
        for name in ("margin_seen", "home_close", "away_close", "shape_vel"):
            cols["{}_past{:g}".format(name, w)] = _window_mean(cols[name], -k, 0)
            cols["{}_next{:g}".format(name, w)] = _window_mean(cols[name], 0, k)
    names = sorted(cols)
    return np.column_stack([cols[c] for c in names]).astype(np.float32), names


def frame_design(t, game=GAME):
    """(F, names): features on every graded frame, NaN rows elsewhere."""
    events = touch_events(t)
    F, names = None, None
    for _, first, last in half_video_blocks(game):
        X, names = half_features(t, first, last, events)
        if F is None:
            F = np.full((len(t["ball"]), X.shape[1]), np.nan, dtype=np.float32)
        F[first:last + 1] = X
    return F, names
