"""The experiment log: shared/log.csv, one row per logged run, written only under a file lock.

Parallel runs in different worktrees append to the same file, so allocating an id and appending
the row happen inside one lock. Rows are never rewritten except through set_note().
"""

import csv
import fcntl
import os
from contextlib import contextmanager

from harness import paths

COLUMNS = [
    "exp_id", "timestamp", "status", "experiment", "params", "parent", "description",
    "git_commit", "branch", "tuned_on_public",
    "pub_spell", "pub_spell_lo", "pub_spell_hi", "pub_macro", "pub_macro_lo", "pub_macro_hi",
    "pub_n_spells", "pub_gt_spells", "pub_mean_dur_s",
    "full_n_spells", "full_mean_dur_s", "full_home_share", "full_dead_share",
    "full_abstain_share",
    "tag_agree", "tag_alive", "restart_hit", "restart_team",
    "sha256", "runtime_s", "notes",
]


@contextmanager
def locked():
    with open(paths.LOCK, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read():
    if not os.path.isfile(paths.LOG):
        return []
    with open(paths.LOG, newline="") as handle:
        return list(csv.DictReader(handle))


def _format(value):
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return "" if value is None else str(value)


def append(row, name):
    """Allocate e<NNN>-<name>, append the row under it, return the id."""
    with locked():
        rows = read()
        exp_id = "e{:03d}-{}".format(len(rows) + 1, name)
        new = not rows
        with open(paths.LOG, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow({key: _format(row.get(key)) for key in COLUMNS} | {"exp_id": exp_id})
    return exp_id


def find(exp_id):
    """The row for an id, accepting the bare 'e012' prefix."""
    for row in read():
        if row["exp_id"] == exp_id or row["exp_id"].split("-")[0] == exp_id:
            return row
    raise KeyError("no run {} in {}".format(exp_id, paths.LOG))


def set_note(exp_id, note):
    with locked():
        rows = read()
        target = find(exp_id)["exp_id"]
        for row in rows:
            if row["exp_id"] == target:
                row["notes"] = note
        with open(paths.LOG, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
