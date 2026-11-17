"""Rule-safe choice of the dead-model drop group and decoder costs, on outside data only
(notes/final-queue-2026-11-13.md, protocol P1).

    uv run python -m src.clean_cv --seedsets 01,23,45

Leave-one-DFL-match-out. Models train on plain degraded rows (tag h7.3raw: raw DFL labels, ball lost
above the provider's own 7.3 m ceiling). Each held-out match is scored, against its raw DFL labels,
in three renderings of its stoppages:
    plain   DFL as it is: the ball frozen for the whole stoppage, teleported at the kick
    hide    the dead ball unseen                        (src/external/view.py)
    taker   the dead ball carried by the restart taker  (src/external/view.py)
Our tracker never shows a frozen ball (notes/ld-match.md, label-free), so the transfer score is the
mean of hide and taker; plain is reported. Held-out log-odds are saved for the decoder studies.
Prints aggregates only (outside data).
"""
import argparse
import itertools
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from harness import paths  # noqa: F401
import possession_metrics as pm
from src import dfl_only
from src.external import idsse
from src.team_swap import swapped

BASE = "h{:g}".format(dfl_only.CEILING_M)
RENDERS = ("plain", "hide", "taker")
DROPS = ("", "A", "B", "C")
CDS = (1.0, 2.0, 4.0)
CTS = (0.5, 0.75, 1.0, 1.5, 2.0)
OUT = os.path.join(dfl_only.CACHE, "clean_cv")


def tag(render):
    return BASE + ("" if render == "plain" else render) + "raw"


def fit(train, names):
    X = np.vstack([r["C"] for r in train])
    y = np.concatenate([r["y"] for r in train])
    play = y != pm.X
    dead = {}
    for drop in DROPS:
        dc = dfl_only.dead_cols(names, drop)
        dead[drop] = (HistGradientBoostingClassifier(**dfl_only.DEAD_HP)
                      .fit(X[play][:, dc], (y[play] == pm.D).astype(int)), dc)
    live = (y == pm.H) | (y == pm.A)
    tc = dfl_only.team_cols(names)
    Xt = X[live][:, tc]
    ht = (y[live] == pm.H).astype(int)
    Xt = np.vstack([Xt, swapped(Xt, [names[i] for i in tc])])
    team = HistGradientBoostingClassifier(**dfl_only.TEAM_HP).fit(Xt, np.concatenate([ht, 1 - ht]))
    return dead, (team, tc)


def logodds(model, X):
    p = np.clip(model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def run_seedset(seeds, res):
    """One fold at a time, each saved on its own (results and held-out log-odds), so an interrupted
    run resumes at the next fold."""
    os.makedirs(OUT, exist_ok=True)
    key = "".join(map(str, seeds))
    todo = [mid for mid in idsse.MATCHES if not os.path.exists(os.path.join(OUT, "fold_s{}_{}.npz".format(key, mid)))]
    plain = {(i, s): dfl_only.match_rows(i, s, tag("plain")) for s in seeds for i in idsse.MATCHES} if todo else {}
    names = list(plain[(idsse.MATCHES[0], seeds[0])]["names"]) if todo else None
    for mid in idsse.MATCHES:
        path = os.path.join(OUT, "fold_s{}_{}.npz".format(key, mid))
        if os.path.exists(path):
            with np.load(path, allow_pickle=True) as a:
                res.extend(tuple(r) for r in a["res"])
            continue
        t0 = time.time()
        fold, saved = [], {}
        dead, (team, tc) = fit([plain[(k, s)] for s in seeds for k in idsse.MATCHES if k != mid], names)
        for s in seeds:
            for render in RENDERS:
                r = plain[(mid, s)] if render == "plain" else dfl_only.match_rows(mid, s, tag(render))
                lt = logodds(team, r["C"][:, tc])
                halves = [tuple(h) for h in r["halves"]]
                for drop in DROPS:
                    model, dc = dead[drop]
                    ld = logodds(model, r["C"][:, dc])
                    saved["{}_{}_{}_{}_ld".format(mid, s, render, drop or "none")] = ld
                    for cd, ct in itertools.product(CDS, CTS):
                        pred = dfl_only.decode(ld, lt, r["idx"], halves, len(r["label"]), cd, ct)
                        f1, mf = dfl_only.score(r["label"], pred)[:2]
                        fold.append((key, mid, s, render, drop or "none", cd, ct, f1, mf))
                saved["{}_{}_{}_lt".format(mid, s, render)] = lt
        del dead, team
        np.savez(path, res=np.array(fold, dtype=object), **saved)
        res.extend(fold)
        print("  seeds {} held out {} {:.0f}s".format(seeds, mid, time.time() - t0), flush=True)
    del plain
    # one file per seed set for src/decoder_pack.py
    merged = {}
    for mid in idsse.MATCHES:
        with np.load(os.path.join(OUT, "fold_s{}_{}.npz".format(key, mid)), allow_pickle=True) as a:
            merged.update({k: a[k] for k in a.files if k != "res"})
    np.savez(os.path.join(OUT, "logits_s{}.npz".format(key)), **merged)


def summarise(res):
    a = np.array([(int("".join(map(str, r[0]))), RENDERS.index(r[3]), DROPS.index("" if r[4] == "none" else r[4]),
                   r[5], r[6], r[7], r[8]) for r in res])
    sets = sorted(set(a[:, 0]))
    rows = []
    for d, cd, ct in itertools.product(range(len(DROPS)), CDS, CTS):
        k = (a[:, 2] == d) & (a[:, 3] == cd) & (a[:, 4] == ct)
        per = {}
        for ss in sets:
            q = k & (a[:, 0] == ss)
            per[ss] = [a[q & (a[:, 1] == ri), 5].mean() for ri in range(3)]
        plain = np.mean([per[ss][0] for ss in sets])
        transfer_by_set = [np.mean(per[ss][1:]) for ss in sets]
        rows.append((DROPS[d] or "none", cd, ct, float(np.mean(transfer_by_set)), plain, transfer_by_set))
    rows.sort(key=lambda r: -r[3])
    print("\ndrop   cd   ct   transfer(hide,taker)  plain   transfer per seed set")
    for r in rows[:20]:
        print("{:5s} {:4.1f} {:5.2f}   {:.4f}             {:.4f}  {}".format(
            r[0], r[1], r[2], r[3], r[4], " ".join("{:.4f}".format(x) for x in r[5])))
    print("\nbest per drop group:")
    for d in DROPS:
        b = max((r for r in rows if r[0] == (d or "none")), key=lambda r: r[3])
        print("{:5s} cd {:.1f} ct {:.2f}  transfer {:.4f}  plain {:.4f}".format(*b[:5]))
    best = rows[0][3]
    tied = [r for r in rows if r[3] >= best - 0.005]
    pick = max(tied, key=lambda r: r[4])
    print("\nP1 rule: {} configs within 0.005 of the best transfer {:.4f}; highest plain among them: "
          "drop {} cd {} ct {} (transfer {:.4f}, plain {:.4f})".format(len(tied), best, *pick[:5]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seedsets", default="01,23,45")
    a = p.parse_args()
    res = []
    for ss in a.seedsets.split(","):
        run_seedset(tuple(int(c) for c in ss), res)
        np.save(os.path.join(OUT, "results.npy"), np.array(res, dtype=object), allow_pickle=True)
    summarise(res)


if __name__ == "__main__":
    main()
