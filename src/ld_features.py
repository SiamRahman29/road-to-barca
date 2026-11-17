"""Live/dead features from the tracking, per frame and pooled into short blocks.

    from src.ld_features import frame_features
    F = frame_features(tracking(), first, last)   (n_frames, len(NAMES)) per frame, NaN = unknown

The submission path (src/external/common.py) uses only the tracking columns, NAMES.

What src/features.py lacks for this question is memory: while the ball is unseen, where and how was
it last seen? A ball lost mid-pitch in a crowd is usually still in play; one last seen at a line,
heading out, usually is not. Every feature is about the ball or the whole field, never a track
slot, and nothing depends on which team attacks which goal.
"""

import warnings

import numpy as np

FPS = 25

# "seen", "out", "stationary" hold 0/1 (NaN if unknown); the rest are real. NAMES is the column order.
REAL = ("ball_speed", "player_speed", "moving_share", "unseen_time", "line_dist",
        "outward_speed", "crowd", "n_players", "seen_window", "speed_window", "spot_dist")
NAMES = ("seen", "out", "stationary") + REAL
OUT_MARGIN_M = 1.0       # ball this far outside a line counts as out of play
SPEED_FLOOR = 0.1        # m/s, keeps the logs finite
BALL_LAG = 2             # frames either side for the ball speed
VELOCITY_LAG = 4         # frames back for the ball's velocity at a sighting
STATIONARY_MS = 1.0      # a seen ball slower than this is being placed or held
MOVING_MS = 2.0          # a player faster than this is jogging or more
CROWD_M = 5.0            # radius around the last-seen ball for the crowd count
SEEN_WINDOW = 2 * FPS    # +/- frames for the share of frames with the ball seen
SPEED_WINDOW = 3 * FPS   # +/- frames for the mean player speed
LINE_CLIP = (-5.0, 35.0)
OUTWARD_CLIP = 30.0
SPOT_CLIP = 20.0
GOAL_AREA = (5.5, 18.32 / 2)  # depth from the goal line, half-width, metres
PENALTY_SPOT_M = 11.0
def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return function(*args, **kwargs)


def _last_valid_index(valid):
    """Index of the last frame <= i where valid, -1 before the first."""
    return np.maximum.accumulate(np.where(valid, np.arange(len(valid)), -1))


def _carry(values, index):
    return np.where(index >= 0, values[np.maximum(index, 0)], np.nan)


def _centred_mean(values, half_width):
    """NaN-aware mean over a centred window of +/- half_width frames."""
    valid = ~np.isnan(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = counts[high] - counts[low]
    return np.where(n > 0, (sums[high] - sums[low]) / np.maximum(n, 1), np.nan)


def _spot_distance(x, y, length, width):
    """Metres to the nearest restart location: a corner, the centre, a penalty spot, a goal area."""
    centre_y = width / 2
    points = [(0, 0), (0, width), (length, 0), (length, width), (length / 2, centre_y),
              (PENALTY_SPOT_M, centre_y), (length - PENALTY_SPOT_M, centre_y)]
    best = np.min([np.hypot(x - px, y - py) for px, py in points], axis=0)
    depth, half_width = GOAL_AREA
    dy = np.maximum(np.abs(y - centre_y) - half_width, 0.0)
    for near_x in (np.maximum(x - depth, 0.0), np.maximum(length - depth - x, 0.0)):
        best = np.minimum(best, np.hypot(near_x, dy))
    return best


def frame_features(t, first, last):
    """(n_frames, len(NAMES)) in physical units for video frames first..last inclusive."""
    length, width = t["pitch"]
    ball = t["ball"][first:last + 1]
    players = t["players"][first:last + 1]
    bx, by = ball[:, 0] * length, ball[:, 1] * width
    px, py, ps = players[:, :, 0] * length, players[:, :, 1] * width, players[:, :, 2]
    n = len(bx)
    seen = ~np.isnan(bx)

    ball_speed = np.full(n, np.nan)
    ball_speed[BALL_LAG:-BALL_LAG] = np.hypot(bx[2 * BALL_LAG:] - bx[:-2 * BALL_LAG],
                                              by[2 * BALL_LAG:] - by[:-2 * BALL_LAG]) \
        / (2 * BALL_LAG / FPS)
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    vx[VELOCITY_LAG:] = (bx[VELOCITY_LAG:] - bx[:-VELOCITY_LAG]) * FPS / VELOCITY_LAG
    vy[VELOCITY_LAG:] = (by[VELOCITY_LAG:] - by[:-VELOCITY_LAG]) * FPS / VELOCITY_LAG

    # Signed distance to the nearest line (negative outside) and the outward normal of that line.
    edges = np.column_stack([bx, length - bx, by, width - by])
    nearest = np.argmin(np.where(np.isnan(edges), np.inf, edges), axis=1)
    line = edges[np.arange(n), nearest]
    normal_x = np.array([-1.0, 1.0, 0.0, 0.0])[nearest]
    normal_y = np.array([0.0, 0.0, -1.0, 1.0])[nearest]
    outward = vx * normal_x + vy * normal_y

    # While the ball is unseen, carry what was known at its last sighting.
    last_seen = _last_valid_index(seen)
    last_moving = _last_valid_index(seen & ~np.isnan(outward))
    carried_x, carried_y = _carry(bx, last_seen), _carry(by, last_seen)
    unseen_time = np.where(last_seen >= 0, (np.arange(n) - last_seen) / FPS, np.nan)

    crowd = np.sum(np.hypot(px - carried_x[:, None], py - carried_y[:, None]) < CROWD_M, axis=1)
    mean_speed = _quiet(np.nanmean, ps, axis=1)
    tracked = np.sum(~np.isnan(px), axis=1)

    columns = {
        "seen": seen.astype(float),
        "out": np.where(seen, line < -OUT_MARGIN_M, np.nan),
        "stationary": np.where(np.isnan(ball_speed), np.nan, ball_speed < STATIONARY_MS),
        "ball_speed": np.log(ball_speed + SPEED_FLOOR),
        "player_speed": np.log(mean_speed + SPEED_FLOOR),
        "moving_share": np.where(tracked > 0, np.sum(ps > MOVING_MS, axis=1) / np.maximum(tracked, 1),
                                 np.nan),
        "unseen_time": np.log1p(unseen_time),
        "line_dist": np.clip(_carry(line, last_seen), *LINE_CLIP),
        "outward_speed": np.clip(_carry(outward, last_moving), -OUTWARD_CLIP, OUTWARD_CLIP),
        "crowd": np.where(last_seen >= 0, crowd, np.nan).astype(float),
        "n_players": tracked.astype(float),
        "seen_window": _centred_mean(seen.astype(float), SEEN_WINDOW),
        "speed_window": np.log(_centred_mean(mean_speed, SPEED_WINDOW) + SPEED_FLOOR),
        "spot_dist": np.minimum(_spot_distance(carried_x, carried_y, length, width), SPOT_CLIP),
    }
    return np.column_stack([columns[name] for name in NAMES])
