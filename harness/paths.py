"""Where everything lives, resolved the same way from the main checkout and from any worktree.

Every worktree has its own copy of the code, but there is exactly one data/ and one shared/: both
live in the main checkout, which git calls the common dir's parent. That is what lets parallel
experiments in separate worktrees share one dataset, one cache and one log.

    shared/
      log.csv          one row per logged run, appended under a lock
      log.lock
      runs/<exp_id>/   submission.csv + result.json for each logged run
      cache/           parsed tracking arrays (src/data.py)
      derived/         artifacts produced elsewhere, e.g. on a remote GPU
"""

import os
import subprocess
import sys

# Code root of *this* checkout (main or worktree) -- the code that is actually running.
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _main_checkout():
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=CODE_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return os.path.dirname(common)


MAIN_ROOT = _main_checkout()
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(MAIN_ROOT, "data")
SHARED = os.environ.get("MTAH_SHARED_DIR") or os.path.join(MAIN_ROOT, "shared")
RUNS = os.path.join(SHARED, "runs")
CACHE = os.path.join(SHARED, "cache")
DERIVED = os.path.join(SHARED, "derived")
LOG = os.path.join(SHARED, "log.csv")
LOCK = os.path.join(SHARED, "log.lock")

# The scorer the organisers run. Imported from this checkout, never copied or modified.
EVALUATION = os.path.join(CODE_ROOT, "evaluation")
PUBLIC_GT = os.path.join(EVALUATION, "gt", "BAR-ATH_possession_gt_public.csv")

# games.py reads MTAH_DATA_DIR at import time, so this must run before anything imports it.
os.environ["MTAH_DATA_DIR"] = DATA_DIR
for path in (CODE_ROOT, EVALUATION):
    if path not in sys.path:
        sys.path.insert(0, path)

for path in (RUNS, CACHE, DERIVED):
    os.makedirs(path, exist_ok=True)
