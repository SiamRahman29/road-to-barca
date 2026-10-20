"""Run an experiment end to end and log it: predict, write, validate, score, diagnose, append.

    python -m harness.run <experiment> [-d "what changed"] [--parent e004] [--set k=v ...]
    python -m harness.run <experiment> --grid window=25,50,125 [--grid k=a,b] [--jobs 3]
    python -m harness.run <experiment> --scratch          # try it without logging

<experiment> is a module in src/experiments/ (or any dotted path) that defines

    PARAMS = {...}             defaults; --set and --grid override them
    TUNED_ON_PUBLIC = False    True if anything was fitted or chosen on the public 5 minutes
    def predict(params):       -> 199319 labels from H/A/D/X, row i = video frame i

A logged run needs a clean working tree, so every row in the log maps to a commit that
regenerates it. --scratch skips that requirement and the log, and writes to shared/runs/_scratch/.
"""

import argparse
import importlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np

from harness import log, paths
from harness.diagnostics import diagnose
from harness.score import (N_FRAMES, full_match_stats, read_codes, score_public, sha256,
                           validate, write_submission)


def git(*args):
    return subprocess.run(["git", *args], cwd=paths.CODE_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def load_experiment(name):
    module_name = name if "." in name else "src.experiments." + name
    return importlib.import_module(module_name)


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def execute(experiment, params, out_dir):
    """Everything for one parameter set. Runs in a worker process for grids."""
    started = time.time()
    try:
        labels = np.asarray(load_experiment(experiment).predict(params))
        if labels.shape != (N_FRAMES,):
            raise ValueError("predict() returned shape {}, expected ({},)".format(
                labels.shape, N_FRAMES))
        path = os.path.join(out_dir, "submission.csv")
        write_submission(path, labels)
        reason = validate(path)
        if reason:
            return {"status": "invalid", "notes": reason,
                    "runtime_s": time.time() - started}
        codes = read_codes(path)
        result = {"status": "ok", "sha256": sha256(path)}
        result.update(score_public(codes))
        result.update(full_match_stats(codes))
        result.update(diagnose(codes))
        result["runtime_s"] = time.time() - started
        return result
    except Exception:
        return {"status": "error", "notes": traceback.format_exc(limit=3).strip().splitlines()[-1],
                "traceback": traceback.format_exc(), "runtime_s": time.time() - started}


def summary(exp_id, result):
    if result["status"] != "ok":
        return "{:<28} {}: {}".format(exp_id, result["status"].upper(), result.get("notes"))
    return ("{:<28} spellF1 {:.4f} [{:.3f}, {:.3f}]  macroF1 {:.4f}  "
            "pub poss {}/{}  full poss {} ({:.1f} s)  tag {:.3f}  restart {:.3f}  {}").format(
        exp_id, result["pub_spell"], result["pub_spell_lo"], result["pub_spell_hi"],
        result["pub_macro"], result["pub_n_spells"], result["pub_gt_spells"],
        result["full_n_spells"], result["full_mean_dur_s"], result["tag_agree"],
        result["restart_hit"], result["sha256"][:12])


def variants(defaults, sets, grids):
    base = dict(defaults)
    base.update(sets)
    if not grids:
        return [base]
    keys = list(grids)
    out = []
    for combo in itertools.product(*(grids[k] for k in keys)):
        params = dict(base)
        params.update(zip(keys, combo))
        out.append(params)
    return out


def variant_name(name, params, grids):
    if not grids:
        return name
    return name + "-" + "-".join("{}{}".format(k, params[k]) for k in grids)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment")
    parser.add_argument("--name", help="short run name (default: the experiment name)")
    parser.add_argument("-d", "--description", default="")
    parser.add_argument("--parent", default="", help="exp_id this run builds on")
    parser.add_argument("--set", action="append", default=[], metavar="K=V")
    parser.add_argument("--grid", action="append", default=[], metavar="K=A,B,C")
    parser.add_argument("--params-json", help="the complete params dict, replacing defaults")
    parser.add_argument("--jobs", type=int, default=3, help="parallel grid workers (RAM: ~7 GB)")
    parser.add_argument("--tuned-on-public", action="store_true")
    parser.add_argument("--scratch", action="store_true", help="do not log; dirty tree allowed")
    parser.add_argument("--sha-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    module = load_experiment(args.experiment)
    name = args.name or args.experiment.split(".")[-1]
    tuned = args.tuned_on_public or bool(getattr(module, "TUNED_ON_PUBLIC", False))

    sets = dict(item.split("=", 1) for item in args.set)
    sets = {k: parse_value(v) for k, v in sets.items()}
    grids = {k: [parse_value(v) for v in values.split(",")]
             for k, values in (item.split("=", 1) for item in args.grid)}
    defaults = json.loads(args.params_json) if args.params_json else module.PARAMS
    unknown = (set(sets) | set(grids)) - set(defaults)
    if unknown:
        sys.exit("unknown params {} for {}; it has {}".format(
            sorted(unknown), args.experiment, sorted(defaults)))

    if not args.scratch and git("status", "--porcelain"):
        sys.exit("working tree is dirty: commit first, or use --scratch for a try-out.\n"
                 + git("status", "--short"))
    commit, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")

    todo = []
    for params in variants(defaults, sets, grids):
        out_dir = os.path.join(paths.RUNS, "_tmp-" + uuid.uuid4().hex[:8])
        todo.append((params, out_dir))

    if len(todo) > 1 and args.jobs > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(todo))) as pool:
            results = list(pool.map(execute, [args.experiment] * len(todo),
                                    [p for p, _ in todo], [d for _, d in todo]))
    else:
        results = [execute(args.experiment, p, d) for p, d in todo]

    failed = False
    for (params, tmp_dir), result in zip(todo, results):
        failed |= result["status"] != "ok"
        run_name = variant_name(name, params, grids)
        if args.sha_only:
            print(result.get("sha256") or result["status"] + ": " + str(result.get("notes")))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue
        if args.scratch:
            exp_id = run_name
            final_dir = os.path.join(paths.RUNS, "_scratch", run_name)
        else:
            row = dict(result)
            row.update({
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "experiment": args.experiment,
                "params": json.dumps(params, sort_keys=True),
                "parent": args.parent, "description": args.description,
                "git_commit": commit[:12], "branch": branch,
                "tuned_on_public": "yes" if tuned else "",
            })
            exp_id = log.append(row, run_name)
            final_dir = os.path.join(paths.RUNS, exp_id)
        shutil.rmtree(final_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.move(tmp_dir, final_dir)
        with open(os.path.join(final_dir, "result.json"), "w") as handle:
            json.dump({"exp_id": exp_id, "params": params, **result}, handle, indent=1,
                      default=float)
        print(summary(exp_id, result))
        if "traceback" in result:
            print(result["traceback"], file=sys.stderr)
    if tuned and not args.sha_only:
        print("note: tuned on the public window -- its public score is not evidence of anything")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
