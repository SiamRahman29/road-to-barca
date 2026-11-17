"""Regenerate the submitted file from the data bundle and the outside data.

    uv run python -m src.submit                 # writes Scio.csv in the repository root
    uv run python -m src.submit --out x.csv

Downloads the seven IDSSE / DFL matches (CC BY 4.0, figshare, no account) into shared/external/ on the
first run, trains the dead-ball and team models on them, labels our match and validates the file
with the organisers' validator. Prints its sha256 and nothing per frame.
"""

import argparse
import os

from harness import paths
from harness.score import sha256, validate, write_submission
from src.experiments import dfl_clean

EXPECTED_SHA256 = "84dedecb3279d776362840c2dfe2fe37f0edd594797baf49089c81bbebff3137"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(paths.CODE_ROOT, "Scio.csv"))
    args = parser.parse_args()
    labels = dfl_clean.predict(dict(dfl_clean.PARAMS))
    write_submission(args.out, labels)
    reason = validate(args.out)
    if reason:
        raise SystemExit("invalid submission: " + reason)
    digest = sha256(args.out)
    print("wrote {} ({} rows), valid".format(args.out, len(labels)))
    print("sha256 {}  {}".format(digest, "matches the submitted file" if digest == EXPECTED_SHA256
                                 else "DIFFERS from the submitted file " + EXPECTED_SHA256))


if __name__ == "__main__":
    main()
