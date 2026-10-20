"""The organisers' baseline, vectorised: nearest tracked player to the ball, then a majority vote.

With window=50 it must reproduce evaluation/results/baseline_smooth50.csv byte for byte, which is
what harness/selftest.py checks. It is the parent of everything else.
"""

import numpy as np

from src.data import tracking
from starter.baseline import AWAY, HOME, NOBODY, smooth

PARAMS = {"window": 50}
TUNED_ON_PUBLIC = False


def nearest_team(t):
    """Team code of the player nearest the ball per frame, NOBODY where there is no answer.

    Same arithmetic, in the same order, as starter/baseline.py, so ties and rounding match.
    """
    length, width = t["pitch"]
    ball_x = t["ball"][:, 0] * length
    ball_y = t["ball"][:, 1] * width
    dx = t["players"][:, :, 0] * length - ball_x[:, None]
    dy = t["players"][:, :, 1] * width - ball_y[:, None]
    distance = dx * dx + dy * dy
    distance[np.isnan(distance)] = np.inf

    # argmin returns the first minimum, as the starter's strict '<' scan keeps the first one.
    seen = ~np.isnan(ball_x) & np.isfinite(distance).any(axis=1)
    nearest = np.argmin(distance, axis=1)
    codes = np.full(len(ball_x), NOBODY, dtype=np.int8)
    codes[seen] = np.where(t["team"][nearest[seen]] == 0, HOME, AWAY)
    return codes


def predict(params):
    return smooth(nearest_team(tracking()), int(params["window"]))
