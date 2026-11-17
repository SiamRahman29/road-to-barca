"""Who has the ball, from touches: a touch detector and the competition's possession rule.

    from src.possession import touch_events, assign_teams
    events = touch_events(tracking())           # [(frame, team, kind)], kind "touch" or "carry"
    team = assign_teams(events, live, fallback)  # 'H'/'A' per frame on live frames

CHALLENGE.md's rule: possession changes only when the other team gains *controlled* possession; a
touch without control, a deflection, or an aerial duel won without follow-up control changes
nothing, and a contested or loose ball stays with the last team. Here control is two touches in a
row by the same team within CONTROL_S, or a touch followed by a carry, and the change is dated to
that team's first touch (a header won and controlled belongs to the winner from the header on).

Every threshold is a fixed football or tracking value set before scoring, not fitted to labels. The
touch window and radius were set from the tracking alone (about 15 touches per live minute; the
ball track is smooth, so a kick spreads over many frames).
"""

import numpy as np

from games import GAME, half_video_blocks

FPS = 25
VELOCITY_K = 12         # frames before/after a touch for the in/out velocity (the ball track is smooth)
TOUCH_DV_MS = 4.0       # change of ball velocity that counts as a touch
TOUCH_RADIUS_M = 2.0    # the toucher is the nearest player, at most this far from the ball
CARRY_RADIUS_M = 1.0    # a carry: the same player this close to the ball...
CARRY_S = 0.4           # ...for at least this long
CONTROL_S = 3.0         # two same-team touches this close together are control
CONTEST_M = 1.5         # an opponent this close to the ball makes a touch contested: it does not count
GAP_S = 4.0             # gap_fill: an event-free stretch at least this long...
SUSTAIN_S = 3.0         # ...switches where the shape-based fallback shows the other team this long
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


def assign_teams(events, live, fallback, min_touches=2, gap_fill=False, game=GAME):
    """'H'/'A' on live frames (others ''), from control events inside each live stretch.

    live: boolean per frame. fallback: 'H'/'A' per frame, used for the opening team of a stretch
    whose first events are not yet control, and for a stretch with no control at all.
    min_touches: same-team touches that count as control (2 is CHALLENGE.md's rule read strictly;
    1 follows the public truth, whose ownership flips on single touches -- sub-second spells).
    gap_fill: inside event-free gaps of GAP_S or more, follow the fallback where it shows the other
    team for SUSTAIN_S -- the shape-based team call covers what the touch detector cannot see.
    """
    team = np.full(len(live), "", dtype="<U1")
    frames = np.array([f for f, _, _ in events], dtype=np.int64)
    for _, first, last in half_video_blocks(game):
        span = live[first:last + 1].astype(int)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], span, [0]))))
        for a, b in zip(edges[::2] + first, edges[1::2] + first):
            inside = [events[i] for i in range(np.searchsorted(frames, a), np.searchsorted(frames, b))]
            switches = _control_switches(inside, min_touches)
            current = switches[0][1] if switches else _majority(fallback[a:b])
            team[a:b] = current
            for frame, who in switches:
                team[frame:b] = who
            if gap_fill:
                _fill_gaps(team, fallback, [f for f, _, _ in inside], a, b)
    return team


def _fill_gaps(team, fallback, frames, a, b):
    """In place: inside event-free gaps, take sustained fallback runs that disagree with team."""
    bounds = [a] + list(frames) + [b]
    for gap_start, gap_stop in zip(bounds[:-1], bounds[1:]):
        if gap_stop - gap_start < GAP_S * FPS:
            continue
        span = fallback[gap_start:gap_stop]
        edges = np.flatnonzero(np.concatenate(([True], span[1:] != span[:-1], [True])))
        for x, y in zip(edges[:-1], edges[1:]):
            if y - x >= SUSTAIN_S * FPS and span[x] != team[gap_start + x]:
                # The fallback's run holds until the next event, as a switch would.
                team[gap_start + x:gap_stop] = span[x]


def _control_switches(events, min_touches=2):
    """[(frame, team)] where a team gains control, the first one opening the stretch."""
    switches, current = [], None
    pending = None  # (team, first frame, touches so far)
    for frame, who, kind in events:
        if not who:
            continue
        if who == current:
            pending = None  # the holder touched it again: any opposing touch was a deflection
            continue
        if pending and (pending[0] != who or frame - pending[1] > CONTROL_S * FPS):
            pending = None
        pending = (who, pending[1] if pending else frame, (pending[2] if pending else 0) + 1)
        if kind == "carry" or pending[2] >= min_touches:
            switches.append((pending[1], who))
            current, pending = who, None
    return switches


def _majority(labels):
    home, away = np.sum(labels == HOME), np.sum(labels == AWAY)
    return HOME if home >= away else AWAY
