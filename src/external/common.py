"""Features computed the same way on the competition tracking and on outside tracking, and the
degradation that makes clean outside tracking look like ours.

    C, names = common_design(t, halves)        per frame: team features (src/team_features),
                                               tracking block columns with 1/3/10 s context
                                               (src/ld_features.frame_features), restart
                                               tracking columns (src/restart_features, rs_)
    d = degrade(m, profile, seed)              outside match with the ball and players lost the
                                               way our tracker loses them
    profile = dfl_only.free_profile()          ball seen / unseen run lengths over our graded
                                               frames, from our tracking alone (no labels)

No broadcast, audio or Smart Tagging columns: the outside data has none.
"""


import numpy as np

from harness import paths  # noqa: F401  puts evaluation/ on sys.path
import possession_metrics as pm
from src import ld_features, restart_features
from src.possession import touch_events
from src.team_features import _window_mean
from src.team_features import half_features as team_half_features

FPS = 25
SCALES_S = (1, 3, 10)
STATE_LAG = 50              # frames: a run redrawn at a live / dead change starts within this
PLAYER_SEEN_MEAN = 1000     # frames: a tracked player's run, and a lost one's, on average
PLAYER_LOST_MEAN = 60


def common_design(t, halves):
    events = touch_events(t)
    n = len(t["ball"])
    blocks, names = [], None
    for first, last in halves:
        F, team_names = team_half_features(t, first, last, events)
        B = ld_features.frame_features(t, first, last)
        ctx, ctx_names = [B], ["ld_" + c for c in ld_features.NAMES]
        for s in SCALES_S:
            k = s * FPS
            for side, (lo, hi) in (("past", (-k, -1)), ("next", (1, k))):
                ctx.append(np.column_stack([_window_mean(B[:, j], lo, hi) for j in range(B.shape[1])]))
                ctx_names += ["ld_{}_{}{}".format(c, side, s) for c in ld_features.NAMES]
        R, r_names = restart_features.half_features(t, first, last)
        blocks.append((first, last, np.hstack([F, np.hstack(ctx), R]).astype(np.float32)))
        names = team_names + ctx_names + r_names
    C = np.full((n, len(names)), np.nan, dtype=np.float32)
    for first, last, X in blocks:
        C[first:last + 1] = X
    return C, names


def _runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return e[1::2] - e[::2]


def degrade(m, profile, seed, air_m):
    """air_m: a ball higher than this is lost (src/dfl_only.CEILING_M, the provider's own ceiling)."""
    rng = np.random.default_rng(seed)
    n = len(m["ball"])
    dead = m["label"] == pm.D
    seen = np.ones(n, dtype=bool)
    state = None
    status, left = True, 0
    i = 0
    while i < n:
        s = "dead" if dead[i] else "live"
        if s != state:
            # A live / dead change: after a short random lag the run restarts from this state.
            lag = int(rng.integers(0, STATE_LAG))
            seen[i:i + lag] = status
            i += lag
            state = s
            if i >= n:
                break
            status = rng.random() < profile[s]["share"]
            left = int(rng.choice(profile[s]["seen" if status else "unseen"]))
            continue
        take = min(left, n - i)
        # Stop at the next state change, so the new state's runs take over there.
        change = np.flatnonzero(dead[i:i + take] != dead[i])
        if len(change):
            take = int(change[0])
        seen[i:i + take] = status
        i += take
        left -= take
        if left <= 0:
            status = not status
            left = int(rng.choice(profile[s]["seen" if status else "unseen"]))
    z = m["ball"][:, 2]
    seen &= ~(np.nan_to_num(z) > air_m)
    ball = m["ball"].copy()
    ball[~seen, :2] = np.nan
    ball[:, 2] = np.nan
    players = m["players"].copy()
    present = ~np.isnan(players[:, :, 0])
    for k in range(players.shape[1]):
        on = np.flatnonzero(present[:, k])
        if not len(on):
            continue
        lost = np.zeros(n, dtype=bool)
        f = on[0]
        while f <= on[-1]:
            f += int(rng.geometric(1.0 / PLAYER_SEEN_MEAN))
            g = int(rng.geometric(1.0 / PLAYER_LOST_MEAN))
            lost[f:f + g] = True
            f += g
        players[lost, k] = np.nan
    out = dict(m)
    out["ball"], out["players"] = ball, players
    return out
