"""Check the harness against numbers it did not produce. Run after any change to the environment.

    python -m harness.selftest

1. The vectorised baseline in src/ regenerates the four committed organiser baselines byte for
   byte -- so data loading, frame alignment and the file writer are right.
2. Scoring those files here gives the public spellF1 and macroF1 that README.md publishes -- so the
   harness reports what the official scorer reports.
"""

import os
import sys
import tempfile

from harness import paths
from harness.score import read_codes, score_public, sha256, write_submission
from src.data import tracking
from src.experiments.baseline_nearest import nearest_team
from starter.baseline import smooth

# README.md, "Try it right now": public-window scores of the committed baselines.
PUBLISHED = {
    "baseline_raw": (0, 0.0378, 0.5972),
    "baseline_smooth25": (25, 0.2868, 0.6200),
    "baseline_smooth50": (50, 0.3256, 0.6207),
    "baseline_smooth125": (125, 0.4242, 0.5812),
}


def main():
    codes = nearest_team(tracking())
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, (window, spell, macro) in PUBLISHED.items():
            path = os.path.join(tmp, name + ".csv")
            write_submission(path, smooth(codes, window))
            committed = os.path.join(paths.EVALUATION, "results", name + ".csv")
            same_file = sha256(path) == sha256(committed)
            scores = score_public(read_codes(path))
            same_score = (round(scores["pub_spell"], 4) == spell
                          and round(scores["pub_macro"], 4) == macro)
            ok = same_file and same_score
            failures += not ok
            print("{}  {:<20} file {}  spellF1 {:.4f} (published {:.4f})  "
                  "macroF1 {:.4f} (published {:.4f})".format(
                      "ok  " if ok else "FAIL", name, "identical" if same_file else "DIFFERS",
                      scores["pub_spell"], spell, scores["pub_macro"], macro))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
