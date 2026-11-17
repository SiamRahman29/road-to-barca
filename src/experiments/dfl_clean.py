"""The submitted pipeline (team Scio): models trained on degraded DFL matches only, decoded with a
three-state Viterbi, plus the second-half kick-off guard (src/dfl_only.py, src/experiments/kickoff_guard.py).

Every tunable choice was made on outside data:
- ball lost above the provider's own 7.3 m ceiling (label-free);
- dead-model drop group and decoder costs chosen by src/clean_cv.py on DFL held-out matches with the
  stoppages rendered without DFL's frozen ball;
- seeds 0-3 fixed a priori; kick-off guard from the Smart Tagging tag.
Nothing here was chosen on the public truth, on hand labels or on the video.
"""

import numpy as np

from harness import paths  # noqa: F401  puts evaluation/ on sys.path
import possession_metrics as pm
from src import dfl_only
from src.experiments import kickoff_guard

# cd, ct, drop: the pre-registered rule's pick (src/clean_cv.py; 13 configs within 0.005 of the best
# transfer 0.5678, highest plain among them). C and B at cd 2 ct 0.75 differ by 0.00005 in plain: the rule
# decides, not a judgement.
PARAMS = {"cd": 2.0, "ct": 0.75, "seeds": "0,1,2,3", "guard": 1, "tag": "h7.3raw", "drop": "C", "tdrop": "",
          "hp": ""}
TUNED_ON_PUBLIC = False


def predict(params):
    seeds = tuple(int(x) for x in str(params["seeds"]).split(","))
    views = [dfl_only.our_logits(seeds, tag=t, drop=str(params["drop"]), tdrop=str(params["tdrop"]), hp=str(params["hp"]))
             for t in str(params["tag"]).split("+")]   # "a+b": average the models' log-odds
    ld, lt = np.mean([v[0] for v in views], 0), np.mean([v[1] for v in views], 0)
    d = dfl_only.our_design()
    n = 199319
    codes = dfl_only.decode(ld, lt, d["idx"], [tuple(h) for h in d["halves"]], n,
                            float(params["cd"]), float(params["ct"]))
    labels = np.array(pm.LABELS)[codes]
    return kickoff_guard.guard(labels) if int(params["guard"]) else labels
