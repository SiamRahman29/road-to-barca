"""Team-swap mirror of the team features: the same frame with home and away exchanged."""

import numpy as np


# Columns whose sign flips when the teams are swapped: directional quantities oriented to home's
# attack, and home-minus-away differences.
_NEGATE_PREFIX = ("ball_x", "ball_vx", "margin", "close", "last_event_team", "next_event_team",
                  "shape_")
_NEGATE_SUFFIX = ("_cx", "_vel", "_ball_dx")
_SWAP_TOKENS = {"home": "away", "away": "home", "H": "A", "A": "H"}


def swap_plan(names):
    """(source index, sign) per column such that F[:, source] * sign is the same frame seen with
    home and away exchanged: the team-swap mirror used for augmentation."""
    index = {n: i for i, n in enumerate(names)}
    plan = []
    for name in names:
        source = "_".join(_SWAP_TOKENS.get(tok, tok) for tok in name.split("_"))
        negate = name.startswith(_NEGATE_PREFIX) or name.endswith(_NEGATE_SUFFIX)
        plan.append((index[source], -1.0 if negate else 1.0))
    return plan


def swapped(F, names):
    plan = swap_plan(names)
    out = np.empty_like(F)
    for k, (source, sign) in enumerate(plan):
        out[:, k] = F[:, source] * sign
    return out
