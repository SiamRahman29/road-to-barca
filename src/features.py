"""Collective match-state features from the tracking, pooled into short blocks for sequence models.

    from src.features import block_features
    halves = block_features(tracking(), step=5)
    halves[k]["X"]       (n_blocks, n_features)  NaN where a feature is unavailable
    halves[k]["start"]   first video frame of the half; block b covers start + b*step ...

Every feature is about a team or the whole field, never a track slot, because slots are tracklets
and not people (DATA.md). Only graded frames are used: the two halves from games.half_video_blocks.

Orientation: within a half each team attacks one goal. The team whose deepest player usually sits
nearer x = 0 defends the left goal; HOME_SIGN is +1 when home attacks towards x = 1, and every
directional feature is multiplied by it so that positive always means "towards the away goal".
"""

import warnings

import numpy as np

from games import GAME, half_video_blocks

# Feature columns, in order. Bernoulli ones hold 0/1 (NaN if unavailable), the rest are real.
NAMES = ("seen", "out", "margin", "ball_speed", "player_speed", "territory", "spread", "velocity")
BERNOULLI = ("seen", "out")

OUT_MARGIN_M = 1.0     # ball this far outside a line counts as out of play
MARGIN_CLIP = 3.0      # log distance ratio, so a far-away second team cannot dominate
SPEED_FLOOR = 0.1      # m/s, keeps the logs finite on a stationary ball or a standing team
VELOCITY_LAG = 25      # frames either side for the centroid velocity, +/- 1 s
BALL_LAG = 2           # frames either side for the ball speed; the ATD speed column is mostly NaN


def _nan_quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return function(*args, **kwargs)


def home_sign(players_x, team):
    """+1 if home attacks towards x = 1 over these frames, else -1."""
    deepest_home = _nan_quiet(np.nanmedian, _nan_quiet(np.nanmin, players_x[:, team == 0], axis=1))
    deepest_away = _nan_quiet(np.nanmedian, _nan_quiet(np.nanmin, players_x[:, team == 1], axis=1))
    return 1.0 if deepest_home < deepest_away else -1.0


def _area_without_keeper(x, y, own_goal_x):
    """std(x) * std(y) of a team per frame, dropping the player nearest its own goal line."""
    depth = np.abs(x - own_goal_x)
    depth = np.where(np.isnan(depth), np.inf, depth)
    keeper = np.argmin(depth, axis=1)
    x = x.copy()
    y = y.copy()
    rows = np.arange(len(x))
    x[rows, keeper] = np.nan
    y[rows, keeper] = np.nan
    return _nan_quiet(np.nanstd, x, axis=1) * _nan_quiet(np.nanstd, y, axis=1)


def frame_features(t, first, last):
    """(n_frames, len(NAMES)) for video frames first..last inclusive, NaN where unavailable."""
    length, width = t["pitch"]
    team = t["team"]
    ball = t["ball"][first:last + 1]
    players = t["players"][first:last + 1]
    px, py, ps = players[:, :, 0], players[:, :, 1], players[:, :, 2]
    sign = home_sign(px, team)

    bx, by = ball[:, 0], ball[:, 1]
    ball_speed = np.full(len(bx), np.nan)
    ball_speed[BALL_LAG:-BALL_LAG] = np.hypot(
        (bx[2 * BALL_LAG:] - bx[:-2 * BALL_LAG]) * length,
        (by[2 * BALL_LAG:] - by[:-2 * BALL_LAG]) * width) / (2 * BALL_LAG / 25.0)
    seen = ~np.isnan(bx)
    out_x = OUT_MARGIN_M / length
    out_y = OUT_MARGIN_M / width
    out = (bx < -out_x) | (bx > 1 + out_x) | (by < -out_y) | (by > 1 + out_y)

    dx = (px - bx[:, None]) * length
    dy = (py - by[:, None]) * width
    distance = np.sqrt(dx * dx + dy * dy)
    d_home = _nan_quiet(np.nanmin, distance[:, team == 0], axis=1)
    d_away = _nan_quiet(np.nanmin, distance[:, team == 1], axis=1)
    margin = np.clip(np.log((d_away + 1.0) / (d_home + 1.0)), -MARGIN_CLIP, MARGIN_CLIP)

    centroid = _nan_quiet(np.nanmean, px, axis=1)
    territory = sign * (centroid - 0.5) * length
    velocity = np.full(len(centroid), np.nan)
    velocity[VELOCITY_LAG:-VELOCITY_LAG] = sign * length * (
        centroid[2 * VELOCITY_LAG:] - centroid[:-2 * VELOCITY_LAG]) / (2 * VELOCITY_LAG / 25.0)

    home_goal = 0.0 if sign > 0 else 1.0
    home_area = _area_without_keeper(px[:, team == 0], py[:, team == 0], home_goal)
    away_area = _area_without_keeper(px[:, team == 1], py[:, team == 1], 1.0 - home_goal)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.log(home_area / away_area)
    spread[~np.isfinite(spread)] = np.nan

    columns = {
        "seen": seen.astype(np.float64),
        "out": np.where(seen, out, np.nan).astype(np.float64),
        "margin": np.where(seen, margin, np.nan),
        "ball_speed": np.log(ball_speed + SPEED_FLOOR),
        "player_speed": np.log(_nan_quiet(np.nanmean, ps, axis=1) + SPEED_FLOOR),
        "territory": territory / 10.0,
        "spread": spread,
        "velocity": velocity,
    }
    return np.column_stack([columns[name] for name in NAMES]), sign


def pool(features, step):
    """Block means over `step` frames; Bernoulli columns become majority (seen) or any (out)."""
    n_blocks = -(-len(features) // step)
    padded = np.full((n_blocks * step, features.shape[1]), np.nan)
    padded[:len(features)] = features
    blocks = padded.reshape(n_blocks, step, features.shape[1])
    pooled = _nan_quiet(np.nanmean, blocks, axis=1)
    seen, out = NAMES.index("seen"), NAMES.index("out")
    pooled[:, seen] = (pooled[:, seen] >= 0.5).astype(np.float64)
    pooled[:, out] = np.where(np.isnan(pooled[:, out]), np.nan, (pooled[:, out] > 0).astype(float))
    return pooled


def block_features(t, step=5, game=GAME):
    """Pooled features per graded half, continuous columns z-scored over both halves together."""
    halves = []
    for _, first, last in half_video_blocks(game):
        features, sign = frame_features(t, first, last)
        halves.append({"X": pool(features, step), "start": first, "stop": last + 1,
                       "home_sign": sign})

    stacked = np.vstack([half["X"] for half in halves])
    for k, name in enumerate(NAMES):
        if name in BERNOULLI:
            continue
        mean = np.nanmean(stacked[:, k])
        std = np.nanstd(stacked[:, k])
        for half in halves:
            half["X"][:, k] = (half["X"][:, k] - mean) / std
    return halves
