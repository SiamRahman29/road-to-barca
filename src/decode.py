"""The joint H / A / D decoder, kept apart from the hand-label pipeline so the label-free path
(src/dfl_only.py) imports nothing that reads hand labels.

Viterbi over H, A, D per row with emissions

    D: log P(dead)        H: log(1 - P(dead)) + log P(home)        A: likewise with P(away)

each scaled by 1/2, a cost `cd` for every switch between live and dead and `ct` for every switch
between the teams.
"""

import numpy as np


def viterbi3(ld, lt, cd, ct):
    """Path over states 0 H, 1 A, 2 D. ld: log-odds dead, lt: log-odds home (given live)."""
    ld, lt = np.asarray(ld, float), np.asarray(lt, float)
    log_dead = -np.logaddexp(0, -ld)
    log_live = -np.logaddexp(0, ld)
    eh = 0.5 * (log_live - np.logaddexp(0, -lt))
    ea = 0.5 * (log_live - np.logaddexp(0, lt))
    ed = 0.5 * log_dead
    n = len(ld)
    eh, ea, ed = eh.tolist(), ea.tolist(), ed.tolist()
    back = np.zeros((n, 3), dtype=np.int8)
    sh, sa, sd = eh[0], ea[0], ed[0]
    for i in range(1, n):
        # into H
        ch, ca, cdd = sh, sa - ct, sd - cd
        if ch >= ca and ch >= cdd:
            nh, bh = ch, 0
        elif ca >= cdd:
            nh, bh = ca, 1
        else:
            nh, bh = cdd, 2
        ch, ca, cdd = sh - ct, sa, sd - cd
        if ca >= ch and ca >= cdd:
            na, ba = ca, 1
        elif ch >= cdd:
            na, ba = ch, 0
        else:
            na, ba = cdd, 2
        ch, ca, cdd = sh - cd, sa - cd, sd
        if cdd >= ch and cdd >= ca:
            nd, bd = cdd, 2
        elif ch >= ca:
            nd, bd = ch, 0
        else:
            nd, bd = ca, 1
        back[i, 0], back[i, 1], back[i, 2] = bh, ba, bd
        sh, sa, sd = nh + eh[i], na + ea[i], nd + ed[i]
    state = int(np.argmax([sh, sa, sd]))
    out = np.empty(n, dtype=np.int8)
    for i in range(n - 1, -1, -1):
        out[i] = state
        state = back[i, state]
    return out
