"""A DFL match as our tracking would present it: the same fields, and no provider artifacts that
ours never shows. Structure only; losing ball and players the way our tracker does is
src/external/common.degrade, a separate step.

    v = view(m, ball="taker", restart="kick")    m from src.external.idsse.match

What differs in kind (python -m src.external.parity, python -m src.external.dfl_status):
- ball column 2: DFL height on every frame; ours height on 8% of seen frames. Dropped here (the
  live/dead features never read it).
- dead ball: DFL freezes the ball's coordinates for the whole stoppage (99.5% of dead frames,
  60-74% of them parked outside the lines) and teleports it to the restart spot at the kick
  (median 43 m/s at the flip). Ours follows the real ball, or loses it. `ball` chooses what the
  dead frames get:
      "raw"     the frozen coordinates (the control: what every earlier DFL row used)
      "hide"    unseen
      "taker"   the restart taker's position, the player nearest the ball at the first live
                frame: the ball is fetched, carried and placed by the player who plays it
- restart convention: DFL's flag turns alive at the kick (the ball moving > 2 m/s at the flip in
  99.7% of restarts). `restart` chooses the label:
      "kick"    DFL as it is
      "touch"   dead until the next touch after the kick (a player within TOUCH_M, at least
                MIN_FLIGHT frames after the ball moved, capped at CAP_S), the reading of the
                public ground truth's corner and goal kick in notes/ld-match.md
"""

import numpy as np

import possession_metrics as pm

FPS = 25
LAG = 2
MOVE_MS = 2.0
TOUCH_M = 1.0
MIN_FLIGHT = 3
CAP_S = 3.0
BALLS = ("raw", "hide", "taker")
RESTARTS = ("kick", "touch")


def _speed(xy):
    v = np.full(len(xy), np.nan)
    d = xy[2 * LAG:] - xy[:-2 * LAG]
    v[LAG:-LAG] = np.hypot(d[:, 0], d[:, 1]) * FPS / (2 * LAG)
    return v


def dead_spells(label):
    """[(start, stop)] of D runs, stop exclusive, that end in play (followed by H or A)."""
    dead = (label == pm.D).astype(np.int8)
    e = np.flatnonzero(np.diff(np.concatenate(([0], dead, [0]))))
    return [(a, b) for a, b in zip(e[::2], e[1::2]) if b < len(label) and label[b] in (pm.H, pm.A)]


def view(m, ball="taker", restart="kick"):
    assert ball in BALLS and restart in RESTARTS
    length, width = m["pitch"]
    label = m["label"].copy()
    b = m["ball"].copy()
    b[:, 2] = np.nan
    players = m["players"]
    dead = label == pm.D
    spells = dead_spells(label)
    if ball == "hide":
        b[dead, :2] = np.nan
    elif ball == "taker":
        b[dead, :2] = np.nan
        for start, stop in spells:
            d = np.hypot(players[stop, :, 0] - b[stop, 0], (players[stop, :, 1] - b[stop, 1]) * width / length)
            if np.all(np.isnan(d)):
                continue
            k = int(np.nanargmin(d))
            b[start:stop, :2] = players[start:stop, k, :2]
    if restart == "touch":
        xy = m["ball"][:, :2] * (length, width)
        v = _speed(xy)
        p = players[:, :, :2].astype(np.float64) * (length, width)
        cap = int(CAP_S * FPS)
        for _, stop in spells:
            hi = min(len(label), stop + cap)
            moving = np.flatnonzero(v[stop:hi] > MOVE_MS)
            if not len(moving):
                continue
            j0 = stop + moving[0] + MIN_FLIGHT
            if j0 >= hi:
                continue
            with np.errstate(all="ignore"):
                near = np.nanmin(np.hypot(p[j0:hi, :, 0] - xy[j0:hi, None, 0],
                                          p[j0:hi, :, 1] - xy[j0:hi, None, 1]), axis=1)
            touch = np.flatnonzero(near < TOUCH_M)
            end = j0 + touch[0] if len(touch) else hi
            live = (label[stop:end] == pm.H) | (label[stop:end] == pm.A)
            label[stop:end][live] = pm.D
    out = dict(m)
    out["ball"], out["label"] = b, label
    return out
