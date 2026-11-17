"""Touches and carries read off the tracking, for the team features.

    from src.possession import touch_events
    events = touch_events(tracking())           # [(frame, team, kind)], kind "touch" or "carry"

A touch is a sharp change of the ball's velocity with a player near it; a carry is the same player
staying close to the ball. Every threshold is a fixed football or tracking value set before scoring,
not fitted to labels (about 15 touches per minute of play; the ball track is smooth, so a kick
spreads over many frames).
"""

import numpy as np


FPS = 25
VELOCITY_K = 12         # frames before/after a touch for the in/out velocity (the ball track is smooth)
TOUCH_DV_MS = 4.0       # change of ball velocity that counts as a touch
TOUCH_RADIUS_M = 2.0    # the toucher is the nearest player, at most this far from the ball
CARRY_RADIUS_M = 1.0    # a carry: the same player this close to the ball...
CARRY_S = 0.4           # ...for at least this long
HOME, AWAY = "H", "A"


def _nearest(t):
    """(slot, team label, distance m, opponent distance m) of the nearest player to the ball per
    frame; -1/''/inf if none. The opponent distance is the nearest player of the other team."""
    length, width = t["pitch"]
    bx, by = t["ball"][:, 0] * length, t["ball"][:, 1] * width
    d = np.hypot(t["players"][:, :, 0] * length - bx[:, None],
                 t["players"][:, :, 1] * width - by[:, None])
    d[np.isnan(d)] = np.inf
    slot = np.argmin(d, axis=1)
    dist = d[np.arange(len(d)), slot]
    team = np.where(t["team"][slot] == 0, HOME, AWAY)
    home = d[:, t["team"] == 0].min(axis=1)
    away = d[:, t["team"] == 1].min(axis=1)
    opponent = np.where(team == HOME, away, home)
    team[~np.isfinite(dist)] = ""
    slot[~np.isfinite(dist)] = -1
    return slot, team, dist, opponent


def touch_events(t, carry_radius=CARRY_RADIUS_M, contest_m=0.0):
    """[(frame, team, kind)] sorted by frame: sharp ball-velocity changes near a player, and carries.

    contest_m > 0 drops events with an opponent that close to the ball: a contested ball stays.
    """
    length, width = t["pitch"]
    p = np.column_stack([t["ball"][:, 0] * length, t["ball"][:, 1] * width])
    n, k = len(p), VELOCITY_K
    v_in = np.full((n, 2), np.nan)
    v_out = np.full((n, 2), np.nan)
    v_in[k:] = (p[k:] - p[:-k]) * FPS / k
    v_out[:-k] = (p[k:] - p[:-k]) * FPS / k
    dv = np.linalg.norm(v_out - v_in, axis=1)
    dv[np.isnan(dv)] = 0.0
    slot, team, dist, opponent = _nearest(t)

    # Touches: local maxima of the velocity change above the threshold, with a player close by.
    peak = dv >= TOUCH_DV_MS
    for shift in range(1, k + 1):
        peak[shift:] &= dv[shift:] >= dv[:-shift]
        peak[:-shift] &= dv[:-shift] > dv[shift:]
    peak &= dist <= TOUCH_RADIUS_M
    events = [(int(f), str(team[f]), "touch") for f in np.flatnonzero(peak)]

    # Carries: the same player within carry_radius for CARRY_S, dated to the run's first frame.
    close = (dist <= carry_radius) & (slot >= 0)
    same = np.concatenate(([False], close[1:] & close[:-1] & (slot[1:] == slot[:-1])))
    start = None
    for f in range(n + 1):
        if f < n and close[f] and (start is None or same[f]):
            if start is None:
                start = f
            continue
        if start is not None and f - start >= int(CARRY_S * FPS):
            events.append((start, str(team[start]), "carry"))
        start = f if f < n and close[f] else None
    if contest_m > 0:
        events = [e for e in events if opponent[e[0]] > contest_m]
    return sorted(events)
