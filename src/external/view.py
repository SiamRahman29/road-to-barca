"""A DFL match as our tracking would present it: the same fields, and no provider artifacts that
ours never shows. Structure only; losing ball and players the way our tracker does is
src/external/common.degrade, a separate step.

    v = view(m, ball="taker")    m from src.external.idsse.match

What differs in kind :
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
  99.7% of restarts). The labels are kept as they are.
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


def view(m, ball="taker"):
    assert ball in BALLS
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
    out = dict(m)
    out["ball"], out["label"] = b, label
    return out
