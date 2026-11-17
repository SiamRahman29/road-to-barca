"""Baseline plus the kick-off guard: `D` from the first graded frame of each half until its kick-off.

A half can be graded from before its kick-off (FAQ.md: in BAR-ATH the second half is), and no model
sees those frames in any labelled stretch, so models invent possessions there. The kick-off is read
from the Smart Tagging file, never typed in: each half's `SET PIECES` / `Kick Off (Start)` tag marks
a window of a few seconds around the restart, and the guard stops at the start of that window. The
seconds inside the window are left to the model. A half without such a tag is left unguarded.

Pure post-process: `guard(labels)` works on any model's labels. One tagged set piece of the match
feeds it, so the harness's `rstrt` diagnostic is no longer fully independent for that one restart.
"""

import sys

import numpy as np

from harness.diagnostics import RESTART, events
from games import GAME, half_video_blocks
from src.experiments import baseline_nearest

PARAMS = {"window": 50}
TUNED_ON_PUBLIC = False

KICKOFF_TYPE = "kick off (start)"
HALF_TAG = {1: "1st Half", 2: "2nd Half"}


def kickoff_tags():
    """{half tag value: first frame of the earliest Kick Off (Start) window in that half}."""
    out = {}
    for e in events():
        if ((e["code"] or "").strip().lower() == RESTART
                and (e["tags"].get("Type") or "").strip().lower() == KICKOFF_TYPE):
            half = e["tags"].get("Half")
            out[half] = min(out.get(half, e["start_frame"]), e["start_frame"])
    return out


def guard_spans(game=GAME):
    """[(first, stop)] frames to force to `D`: graded start of a half up to its tagged kick-off."""
    tags = kickoff_tags()
    spans = []
    for half, first, last in half_video_blocks(game):
        start = tags.get(HALF_TAG.get(half))
        if start is None:
            print("kickoff_guard: no Kick Off (Start) tag for half {}, unguarded".format(half),
                  file=sys.stderr)
            continue
        stop = min(start, last + 1)
        if stop > first:
            spans.append((first, stop))
    return spans


def guard(labels):
    labels = np.array(labels, copy=True)
    for first, stop in guard_spans():
        labels[first:stop] = "D"
    return labels


def predict(params):
    return guard(baseline_nearest.predict(params))
