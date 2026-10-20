"""Signals about a full-match submission that do not come from the 17 public possessions.

Smart Tagging is automatic and noisy, so none of these is a score to optimise. They are a second
opinion: a change that lifts the public spellF1 while these fall is probably fitting the public
window, and a change that moves these a lot while the public score stays inside its interval is
worth looking at on the timeline.

    tag_agree     on frames inside team-tagged phases where the submission says the ball is alive,
                  the share where it names the team the tag implies
    tag_alive     the share of those tagged frames the submission calls alive at all
    restart_hit   the share of tagged set pieces with a dead -> alive transition near them
    restart_team  of those hits, the share restarting with the tagged team
"""

import numpy as np

from harness import paths  # noqa: F401
from harness.score import FULL_RUNS
import possession_metrics as pm
from starter.load_smart_tagging import load as load_smart_tagging

# DATA.md lists the Team values: the home side, the away side (spelt this way in the file) and N/A.
TEAM_CODE = {"barcelona": pm.H, "atletic club": pm.A}
OTHER = {pm.H: pm.A, pm.A: pm.H}

# Phases performed by the team in possession, and phases performed by the team without the ball.
# LONG BALL and GOALS are left out: a long ball changes hands as often as not, and a goal is a
# moment, not a phase.
ATTACKING = {"build up", "progression", "creating chances", "attacking transition",
             "ball in final third", "ball in the box", "players in the box"}
DEFENDING = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}
RESTART = "set pieces"
RESTART_SLACK = 50  # frames either side of the tagged interval, 2 s at 25 fps

_events = None


def events():
    global _events
    if _events is None:
        _events = load_smart_tagging()
        unknown = {(e["tags"].get("Team") or "").strip().lower() for e in _events} \
            - set(TEAM_CODE) - {"n/a", ""}
        if unknown:
            raise ValueError("Smart Tagging has Team values not mapped to H/A: {}".format(
                sorted(unknown)))
    return _events


def _graded(start, stop):
    return any(a <= start and stop <= b for a, b in FULL_RUNS)


def tag_agreement(filled):
    agree = alive = total = 0
    for event in events():
        code = (event["code"] or "").strip().lower()
        team = TEAM_CODE.get((event["tags"].get("Team") or "").strip().lower())
        start, stop = event["start_frame"], event["end_frame"]
        if team is None or stop <= start or not _graded(start, stop):
            continue
        if code in ATTACKING:
            expected = team
        elif code in DEFENDING:
            expected = OTHER[team]
        else:
            continue
        window = filled[start:stop]
        is_alive = (window == pm.H) | (window == pm.A)
        total += len(window)
        alive += int(is_alive.sum())
        agree += int((window == expected).sum())
    return {
        "tag_agree": agree / alive if alive else 0.0,
        "tag_alive": alive / total if total else 0.0,
    }


def restart_alignment(filled):
    restarts = []  # (frame, team code) of every dead -> alive transition
    for frame, before, after in pm.transitions(filled, FULL_RUNS):
        if before == pm.D and after in (pm.H, pm.A):
            restarts.append((frame, after))
    frames = np.array([f for f, _ in restarts], dtype=np.int64)

    n = hits = team_checked = team_agree = 0
    for event in events():
        if (event["code"] or "").strip().lower() != RESTART:
            continue
        start, stop = event["start_frame"], event["end_frame"]
        if not _graded(start, stop):
            continue
        n += 1
        near = np.flatnonzero((frames >= start - RESTART_SLACK) & (frames < stop + RESTART_SLACK))
        if not len(near):
            continue
        hits += 1
        team = TEAM_CODE.get((event["tags"].get("Team") or "").strip().lower())
        if team is not None:
            team_checked += 1
            team_agree += int(restarts[near[0]][1] == team)
    return {
        "restart_hit": hits / n if n else 0.0,
        "restart_team": team_agree / team_checked if team_checked else 0.0,
    }


def diagnose(codes):
    filled = pm.fill_abstentions(codes, FULL_RUNS)
    out = tag_agreement(filled)
    out.update(restart_alignment(filled))
    return out
