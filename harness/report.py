"""Read the log in the terminal.

    python -m harness.report                 # every logged run, best public spellF1 first
    python -m harness.report --last 10       # the ten most recent, in run order
    python -m harness.report --compare e004 e007
    python -m harness.report --note e007 "split possessions on long ball losses"

The public interval column is the thing to read first: two runs whose intervals overlap are not
distinguishable on the public window, whatever their point scores say.
"""

import argparse
import json
import os
import sys

import numpy as np

from harness import log, paths
from harness.score import FULL_RUNS, public_gt, read_codes, spell_agreement
import possession_metrics as pm


def f(value, width=6, digits=3):
    if value in ("", None):
        return "-".rjust(width)
    return "{:>{w}.{d}f}".format(float(value), w=width, d=digits)


def table(rows):
    header = ("{:<38} {:>6} {:>15} {:>6} {:>7} {:>9} {:>5} {:>6} {:>6}  {}".format(
        "exp_id", "spell", "public 90% CI", "macro", "pub n", "full n", "dur", "tag",
        "rstrt", "description"))
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["status"] != "ok":
            print("{:<38} {}  {}".format(row["exp_id"][:38], row["status"].upper(),
                                         row["notes"]))
            continue
        flag = "*" if row["tuned_on_public"] else " "
        print("{:<37}{} {} [{}, {}] {} {:>3}/{:<3} {:>9} {} {} {}  {}".format(
            row["exp_id"][:37], flag, f(row["pub_spell"], 6, 4), f(row["pub_spell_lo"], 5),
            f(row["pub_spell_hi"], 5), f(row["pub_macro"]), row["pub_n_spells"],
            row["pub_gt_spells"], row["full_n_spells"], f(row["full_mean_dur_s"], 5, 1),
            f(row["tag_agree"]), f(row["restart_hit"]), row["description"][:60]))
    if any(row["tuned_on_public"] for row in rows):
        print("\n* tuned on the public window: its public score is contaminated")


def compare(id_a, id_b):
    """How two runs differ, in aggregates only: where they disagree, not what the data says."""
    row_a, row_b = log.find(id_a), log.find(id_b)
    codes = [read_codes(os.path.join(paths.RUNS, r["exp_id"], "submission.csv"))
             for r in (row_a, row_b)]
    graded = np.zeros(len(codes[0]), dtype=bool)
    for a, b in FULL_RUNS:
        graded[a:b] = True
    differ = (codes[0] != codes[1]) & graded

    print("{}  vs  {}".format(row_a["exp_id"], row_b["exp_id"]))
    print("  params A {}".format(row_a["params"]))
    print("  params B {}".format(row_b["params"]))
    print("  frames that differ: {} of {} graded ({:.1f}%)".format(
        int(differ.sum()), int(graded.sum()), 100.0 * differ.mean() / graded.mean()))
    names = pm.LABELS
    pairs = {}
    for x, y in zip(codes[0][differ], codes[1][differ]):
        pairs[(names[x], names[y])] = pairs.get((names[x], names[y]), 0) + 1
    for (x, y), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print("    A {} -> B {}  {:>7} frames".format(x, y, n))

    edges = np.flatnonzero(np.diff(differ.astype(np.int8)) == 1)
    print("  disagreement stretches: {}".format(len(edges)))
    print("  full-match spell agreement (spellF1 of B against A): {:.4f}".format(
        spell_agreement(codes[0], codes[1])))
    for key in ("pub_spell", "full_n_spells", "full_mean_dur_s", "tag_agree", "restart_hit"):
        print("  {:<16} {:>10}  {:>10}".format(key, row_a[key], row_b[key]))

    gt = public_gt()
    window = gt != pm.X
    for label, c in (("A", codes[0]), ("B", codes[1])):
        print("  public window: {} wrong on {} of {} frames".format(
            label, int(((c != gt) & window).sum()), int(window.sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--last", type=int)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--note", nargs=2, metavar=("EXP", "TEXT"))
    parser.add_argument("--json", metavar="EXP", help="print a run's full result.json")
    args = parser.parse_args()

    if args.note:
        log.set_note(*args.note)
        return
    if args.compare:
        compare(*args.compare)
        return
    if args.json:
        row = log.find(args.json)
        with open(os.path.join(paths.RUNS, row["exp_id"], "result.json")) as handle:
            print(json.dumps(json.load(handle), indent=1))
        return

    rows = log.read()
    if not rows:
        sys.exit("no runs logged yet: python -m harness.run <experiment>")
    if args.last:
        rows = rows[-args.last:]
    else:
        rows = sorted(rows, key=lambda r: -float(r["pub_spell"] or -1))
    table(rows)


if __name__ == "__main__":
    main()
