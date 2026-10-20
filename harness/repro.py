"""Prove a logged run regenerates byte-for-byte from its commit, in a fresh process.

    python -m harness.repro e007

Checks the commit out into a temporary worktree, reruns the experiment there with the logged
params, and compares the CSV's sha256 to the logged one. This is the competition's reproduction
rule tested before the organisers test it: a different hash means something in the pipeline
depends on more than the commit and the data (set ordering, an unseeded RNG, a cache, the clock).
"""

import argparse
import subprocess
import sys
import tempfile

from harness import log, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp_id")
    args = parser.parse_args()

    row = log.find(args.exp_id)
    if row["status"] != "ok":
        sys.exit("{} did not produce a valid file ({})".format(row["exp_id"], row["status"]))

    with tempfile.TemporaryDirectory(prefix="repro-") as tmp:
        worktree = tmp + "/wt"
        subprocess.run(["git", "worktree", "add", "--detach", "--quiet", worktree,
                        row["git_commit"]], cwd=paths.CODE_ROOT, check=True)
        try:
            out = subprocess.run(
                [sys.executable, "-m", "harness.run", row["experiment"], "--scratch",
                 "--sha-only", "--params-json", row["params"]],
                cwd=worktree, capture_output=True, text=True)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", worktree],
                           cwd=paths.CODE_ROOT, check=True)

    got = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr.strip()
    if got == row["sha256"]:
        print("REPRODUCED  {}  sha256 {}".format(row["exp_id"], got[:16]))
        return
    print("MISMATCH    {}\n  logged {}\n  rerun  {}".format(row["exp_id"], row["sha256"], got))
    sys.exit(1)


if __name__ == "__main__":
    main()
