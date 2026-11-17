"""Label-free H/A/D pipeline: models trained on degraded outside matches only (DFL / IDSSE), checked
leave-one-match-out, then applied to our match. No hand label, no private ground truth.

    uv run python -m src.dfl_only cv [--name NAME] [--cd 4,8,16] [--ct 8,16,32] [--seeds 0,1]
    uv run python -m src.dfl_only apply [--cd 8 --ct 16 --seeds 0,1]    # writes shared/cache/dfl_only logits

Validation is 7 independent held-out matches scored with the organisers' scorer, so the number
carries a real spread. The public 5 minutes of our match is a transfer check, never a selection
criterion (17 possessions: one moves spellF1 by ~0.06). Every run is traced in MLflow
(workspace/mlflow.db). Output is aggregates only.
"""

import argparse
import os
import re
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from harness import paths
import possession_metrics as pm
from games import GAME, half_video_blocks
from src.external import common, idsse
from src.decode import viterbi3
from src.team_swap import swapped

STEP = 5                      # frames between rows (0.2 s); labels are repeated to 25 fps
MIN_FRAMES = 37               # controlled possession: runs under 1.5 s join the team before them
# The highest ball height our provider reports on graded frames (7.27 m, label-free), rounded up:
# the rule-safe replacement for common.AIR_M in tags "h7.3...raw".
CEILING_M = 7.3
WORK = os.path.join(paths.MAIN_ROOT, "workspace")   # MLflow tracing only
# Under shared/, so `harness.repro --cold` really starts empty. Keyed by tag and seed, not by source:
# after editing the features or the degradation, delete shared/cache/dfl_only/.
CACHE = os.path.join(paths.CACHE, "dfl_only")
DEAD_HP = dict(max_leaf_nodes=31, min_samples_leaf=200, learning_rate=0.05, max_iter=400,
               l2_regularization=1.0, random_state=0)
TEAM_HP = dict(max_leaf_nodes=15, min_samples_leaf=100, learning_rate=0.05, max_iter=300,
               l2_regularization=1.0, random_state=0)


def _runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return e[1::2] - e[::2]


def our_tracking():
    """Our match: the provider's tracking, parsed by src/data.py."""
    from src.data import tracking
    return tracking()


def free_profile():
    """Ball seen / unseen run lengths over our graded frames, the same for live and dead play.
    Uses only our tracking (no label of any kind), so it regenerates from a fresh clone."""
    seen = ~np.isnan(our_tracking()["ball"][:, 0])
    g = np.zeros(len(seen), bool)
    for _, a, z in half_video_blocks(GAME):
        g[a:z + 1] = True
    one = {"seen": _runs(seen & g), "unseen": _runs(~seen & g), "share": float(seen[g].mean())}
    return {"live": one, "dead": one}


def match_rows(match_id, seed, tag="free"):
    """Degraded features (every STEP-th frame inside play) and controlled labels for one match."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "{}_s{}_{}.npz".format(match_id, seed, tag))
    if not os.path.exists(path):
        m = idsse.match(match_id)
        hv = re.match(r"h(\d+(?:\.\d+)?)(hide|taker)?raw$", tag)
        if not hv:
            raise ValueError("tag must be h<metres>raw, h<metres>hideraw or h<metres>takerraw: " + tag)
        # "h<metres>raw" loses balls above the given height (the provider's own ceiling, CEILING_M);
        # "h<metres>{hide,taker}raw" first renders DFL's stoppages without its frozen ball
        # (src/external/view.py), for held-out evaluation only (src/clean_cv.py).
        src = m
        if hv.group(2):
            from src.external import view
            src = view.view(m, ball=hv.group(2), restart="kick")
            src["ball"] = src["ball"].copy()
            src["ball"][:, 2] = m["ball"][:, 2]     # view drops the height; the ceiling needs it
        d = common.degrade(src, free_profile(), seed=seed * 7 + idsse.MATCHES.index(match_id),
                           air_m=float(hv.group(1)))
        m = src
        C, names = common.common_design(d, m["halves"])
        # The DFL labels as they are. CHALLENGE.md: the ground truth is mechanical and "nothing is
        # smoothed ... afterwards", and the public truth keeps sub-second spells.
        label = m["label"].copy()
        idx = np.concatenate([np.arange(f, l + 1, STEP) for f, l in m["halves"]])
        np.savez(path, C=C[idx], idx=idx, y=label[idx], names=np.array(names), label=label,
                 halves=np.array(m["halves"]))
    a = np.load(path)
    return {k: a[k] for k in a.files}


def our_design(tag="ours"):
    """Features on our match at the same STEP grid, inside the graded halves."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "{}_design.npz".format(tag))
    if not os.path.exists(path):
        halves = [(f, l) for _, f, l in half_video_blocks(GAME)]
        C, names = common.common_design(our_tracking(), halves)
        idx = np.concatenate([np.arange(f, l + 1, STEP) for f, l in halves])
        np.savez(path, C=C[idx], idx=idx, names=np.array(names), halves=np.array(halves))
    a = np.load(path)
    return {k: a[k] for k in a.files}


def our_logits(seeds=(0, 1), tag="h7.3raw", name=None, drop="", tdrop="", hp=""):
    """Models trained on all 7 matches (every seed), log-odds on our match. Cached under `name`."""
    name = name or "{}_s{}{}{}".format(tag, "".join(map(str, seeds)), "_d" + drop if drop else "",
                                           "_t" + tdrop if tdrop else "") + ("_hp" + hp if hp else "")
    path = os.path.join(CACHE, "ours_logits_{}.npz".format(name))
    if not os.path.exists(path):
        ours = our_design()
        rows = [match_rows(i, s, tag) for s in seeds for i in idsse.MATCHES]
        names = list(ours["names"])
        models = fit_models(rows, names, *HPS[hp], drop, tdrop)
        ld, lt = logits(models, ours["C"])
        np.savez(path, ld=ld, lt=lt)
    a = np.load(path)
    return a["ld"], a["lt"]


BIG_DEAD_HP = dict(DEAD_HP, max_leaf_nodes=63, max_iter=600, min_samples_leaf=100)
BIG_TEAM_HP = dict(TEAM_HP, max_leaf_nodes=31, max_iter=500, min_samples_leaf=50)
HPS = {"": (DEAD_HP, TEAM_HP), "big": (BIG_DEAD_HP, BIG_TEAM_HP)}

# Dead-model columns that the clean outside ball makes trivially perfect (a dead ball is frozen) but our
# tracker's ball is not (public window: ball_speed AUC dead-vs-live 0.31 ours, 0.01 DFL).
DROP_A = ("ball_speed", "ld_ball_speed", "ball_acc", "ball_vx", "ld_stationary", "ld_outward_speed")
DROP_B = DROP_A + ("rs_", "since_touch", "until_touch", "last_event", "next_event", "n_touch", "n_carry",
                   "since_carry", "until_carry")
# C: like B but keeps the player-based restart columns (rs_d*, rs_walk, rs_n_wall, rs_player_speed)
DROP_C = DROP_A + ("rs_landing", "rs_launch", "rs_since_strike", "rs_until_strike", "since_touch",
                   "until_touch", "last_event", "next_event", "n_touch", "n_carry", "since_carry", "until_carry")
DROPS = {"": (), "A": DROP_A, "B": DROP_B, "C": DROP_C}


def dead_cols(names, drop=""):
    import re
    pats = DROPS[drop]
    return [i for i, n in enumerate(names)
            if not any(re.sub(r"_(past|next)[\d.]+$", "", n).startswith(p) for p in pats)]


TOUCH_GROUP = ("since_touch", "until_touch", "n_touch", "n_carry", "since_carry", "until_carry",
               "last_event", "next_event")
TDROPS = {"": (), "T": TOUCH_GROUP}


def team_cols(names, tdrop=""):
    import re
    pats = TDROPS[tdrop]
    return [i for i, n in enumerate(names)
            if not (n.startswith("ld_") or n.startswith("rs_"))
            and not any(re.sub(r"_(past|next)[\d.]+$", "", n).startswith(p) for p in pats)]


def fit_models(rows, names, dead_hp=DEAD_HP, team_hp=TEAM_HP, drop="", tdrop=""):
    """rows: list of match_rows dicts. Dead model on every in-play row; team model on live rows,
    each also mirrored (home and away exchanged), so 'home' is no club."""
    X = np.vstack([r["C"] for r in rows])
    y = np.concatenate([r["y"] for r in rows])
    play = y != pm.X
    dc = dead_cols(names, drop)
    dead = HistGradientBoostingClassifier(**dead_hp).fit(X[play][:, dc], (y[play] == pm.D).astype(int))
    live = (y == pm.H) | (y == pm.A)
    tc = team_cols(names, tdrop)
    Xt = X[live][:, tc]
    ht = (y[live] == pm.H).astype(int)
    Xt = np.vstack([Xt, swapped(Xt, [names[i] for i in tc])])
    ht = np.concatenate([ht, 1 - ht])
    team = HistGradientBoostingClassifier(**team_hp).fit(Xt, ht)
    return dead, team, tc, dc


def logits(models, C):
    dead, team, tc, dc = models
    p = np.clip(dead.predict_proba(C[:, dc])[:, 1], 1e-4, 1 - 1e-4)
    q = np.clip(team.predict_proba(C[:, tc])[:, 1], 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)), np.log(q / (1 - q))


CODES = np.array([pm.H, pm.A, pm.D])


def decode(ld, lt, idx, halves, n, cd, ct):
    """Viterbi inside each half on the STEP grid, expanded to every frame; X elsewhere."""
    out = np.full(n, pm.X, dtype=np.int8)
    for f, l in halves:
        sel = np.flatnonzero((idx >= f) & (idx <= l))
        path = CODES[viterbi3(ld[sel], lt[sel], cd, ct)]
        for j, code in zip(idx[sel], path):
            out[j:min(j + STEP, l + 1)] = code
    return out


def score(gt, pred):
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_gt_spells"], r["n_pred_spells"]


def lomo(seeds, cds, cts, tag="h7.3raw", dead_hp=DEAD_HP, team_hp=TEAM_HP, drop="", tdrop=""):
    """Leave-one-match-out. Returns {(cd, ct): [(spellF1, macroF1) per match and seed]} plus the
    per-match breakdown."""
    data = {s: [match_rows(i, s, tag) for i in idsse.MATCHES] for s in seeds}
    names = list(np.load(os.path.join(CACHE, "{}_s{}_{}.npz".format(idsse.MATCHES[0], seeds[0], tag)))["names"])
    res = {(cd, ct): [] for cd in cds for ct in cts}
    per_match = {}
    saved = {}
    for j, match_id in enumerate(idsse.MATCHES):
        t0 = time.time()
        train = [data[s][k] for s in seeds for k in range(len(idsse.MATCHES)) if k != j]
        models = fit_models(train, names, dead_hp, team_hp, drop, tdrop)
        for s in seeds:
            r = data[s][j]
            ld, lt = logits(models, r["C"])
            saved[(match_id, s)] = (ld, lt)
            n = len(r["label"])
            gt = r["label"]
            for cd in cds:
                for ct in cts:
                    f1, mf, ng, npd = score(gt, decode(ld, lt, r["idx"], [tuple(h) for h in r["halves"]], n, cd, ct))
                    res[(cd, ct)].append((f1, mf, npd / max(ng, 1)))
                    per_match.setdefault((cd, ct), {}).setdefault(match_id, []).append(f1)
        print("  held out {}  {:.0f}s".format(match_id, time.time() - t0), flush=True)
    lomo.saved = saved
    return res, per_match, names


def save_lomo_logits(name):
    """After lomo(): keep the held-out log-odds for offline decoder studies (shared/cache/dfl_only)."""
    np.savez(os.path.join(CACHE, "lomo_logits_{}.npz".format(name)),
             **{"{}_{}_{}".format(m, s, k): v for (m, s), (ld, lt) in lomo.saved.items()
                for k, v in (("ld", ld), ("lt", lt))})


class _NoTrace:
    """Stands in for mlflow when it is not installed: tracing is optional, the printout is the record."""
    def __getattr__(self, name):
        return lambda *a, **k: self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main():
    try:
        import mlflow
    except ImportError:
        mlflow = _NoTrace()
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["cv"])
    p.add_argument("--name", default="dfl_only_baseline")
    p.add_argument("--cd", default="4,8,16")
    p.add_argument("--ct", default="8,16,32")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--tag", default="h7.3raw")
    p.add_argument("--drop", default="")
    p.add_argument("--tdrop", default="")
    p.add_argument("--save", default="")
    p.add_argument("--dead_hp", default="", help="overrides, e.g. max_leaf_nodes=63,max_iter=600")
    p.add_argument("--team_hp", default="")
    a = p.parse_args()
    def hp(base, spec):
        out = dict(base)
        for kv in filter(None, spec.split(",")):
            k, v = kv.split("=")
            out[k] = type(base[k])(float(v)) if not isinstance(base[k], int) else int(float(v))
        return out
    dead_hp, team_hp = hp(DEAD_HP, a.dead_hp), hp(TEAM_HP, a.team_hp)
    seeds = [int(x) for x in a.seeds.split(",")]
    cds, cts = [float(x) for x in a.cd.split(",")], [float(x) for x in a.ct.split(",")]
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(WORK, "mlflow.db"))
    mlflow.set_experiment("dfl_only")
    with mlflow.start_run(run_name=a.name):
        mlflow.log_params({"seeds": a.seeds, "tag": a.tag, "dead_drop": a.drop, "step": STEP, "min_frames": MIN_FRAMES,
                           **{"dead_" + k: v for k, v in dead_hp.items()},
                           **{"team_" + k: v for k, v in team_hp.items()}, "tdrop": a.tdrop})
        res, per_match, names = lomo(seeds, cds, cts, a.tag, dead_hp, team_hp, drop=a.drop, tdrop=a.tdrop)
        mlflow.log_param("n_features", len(names))
        if a.save:
            save_lomo_logits(a.save)
        print("{:>5s} {:>5s}  {:>8s} {:>8s} {:>10s}".format("cd", "ct", "spellF1", "macroF1", "pred/true"))
        for (cd, ct), v in res.items():
            f1, mf, ratio = (np.mean([x[i] for x in v]) for i in range(3))
            print("{:5.1f} {:5.1f}  {:8.4f} {:8.4f} {:10.2f}".format(cd, ct, f1, mf, ratio))
        best = max(res, key=lambda k: np.mean([x[0] for x in res[k]]))
        pm_best = {m: float(np.mean(v)) for m, v in per_match[best].items()}
        f1s = np.array(list(pm_best.values()))
        print("best (cd, ct) = {}  mean {:.4f}  sd across matches {:.4f}  per match {}".format(
            best, f1s.mean(), f1s.std(), {k: round(v, 3) for k, v in pm_best.items()}))
        mlflow.log_metrics({"lomo_spellF1": float(f1s.mean()), "lomo_spellF1_sd": float(f1s.std()),
                            "lomo_macroF1": float(np.mean([x[1] for x in res[best]])),
                            "best_cd": best[0], "best_ct": best[1],
                            **{"match_" + k: v for k, v in pm_best.items()}})


if __name__ == "__main__":
    main()
