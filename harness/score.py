"""Scoring, validation and the submission file format, all delegated to the organisers' code.

Nothing here re-implements a metric. Scores come from evaluation/possession_metrics.py exactly as
the official scorer computes them; this module only chooses what to report and how many bootstrap
draws to spend.
"""

import hashlib
import os

import numpy as np

from harness import paths
import possession_metrics as pm
from games import GAME, GAMES, half_video_blocks

N_FRAMES = GAMES[GAME]["n_video_frames"]
BOOTSTRAP_N = 1000

# Half-open graded spans for the full match, the same ones valid_runs() derives from the private
# ground truth: [(11185, 90652), (110808, 190198)]. Used for statistics that need no labels.
FULL_RUNS = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]

_public_gt = None


def public_gt():
    global _public_gt
    if _public_gt is None:
        _public_gt = pm.label_codes(pm.read_label_csv(paths.PUBLIC_GT))
    return _public_gt


def write_submission(path, labels):
    """The submission CSV, byte-for-byte the format starter/baseline.py writes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path):
    """None if the written file passes the official checks, else the scorer's one-line reason."""
    try:
        table = pm.read_label_csv(path)
    except Exception as error:
        return "unreadable: {}".format(error)
    return pm.validate_submission(table, N_FRAMES)


def read_codes(path):
    return pm.label_codes(pm.read_label_csv(path))


def score_public(codes):
    """Official metrics on the public 5 minutes, with possession-bootstrap intervals."""
    result = pm.evaluate(public_gt(), codes, bootstrap_n=BOOTSTRAP_N)
    return {
        "pub_spell": result["spell_f1_mean"],
        "pub_spell_lo": result["spell_bootstrap_low"],
        "pub_spell_hi": result["spell_bootstrap_high"],
        "pub_macro": result["macro_f1"],
        "pub_macro_lo": result["bootstrap_low"],
        "pub_macro_hi": result["bootstrap_high"],
        "pub_n_spells": result["n_pred_spells"],
        "pub_gt_spells": result["n_gt_spells"],
        "pub_mean_dur_s": result["mean_spell_duration_s"],
    }


def full_match_stats(codes):
    """What the submission claims about the whole graded match. No ground truth involved.

    Abstentions are forward-filled first, as the official scorer does before counting spells.
    """
    filled = pm.fill_abstentions(codes, FULL_RUNS)
    stats = pm.possession_stats(filled, FULL_RUNS)
    graded = np.concatenate([codes[a:b] for a, b in FULL_RUNS])
    return {
        "full_n_spells": stats["n_spells"],
        "full_mean_dur_s": stats["mean_duration_s"],
        "full_home_share": stats["home_share"],
        "full_dead_share": float(np.mean(graded == pm.D)),
        "full_abstain_share": float(np.mean(graded == pm.X)),
    }


def spell_agreement(codes_a, codes_b):
    """spellF1 of b scored as if a were the truth, over the full graded match.

    Not a quality score -- neither side is the truth -- but a measure of how differently two runs
    segment the match. 1.0 means they report the same possessions; it is symmetric under F1.
    """
    a = pm.fill_abstentions(codes_a, FULL_RUNS)
    b = pm.fill_abstentions(codes_b, FULL_RUNS)
    return pm.spell_scores(a, b, FULL_RUNS)["mean_f1"]
