# Team Scio — Phase 1: Ball Possession

Technical document for the possession file `Scio.csv`. More than a Hack 2027, Barcelona v Athletic Club.

## In one paragraph

We never fit anything on the provided match. Two gradient-boosted classifiers are trained on seven
**outside** matches (IDSSE / DFL, Bundesliga and 2. Bundesliga, CC BY 4.0). Those matches come with
official frame-level ball status and possession. Their clean tracking is first **degraded** so that it
looks like the automatic tracking we are given. One model says whether the ball is dead and the other
says which team has it. A three-state Viterbi decoder (H / A / D) turns their frame scores into
possessions. Every tunable choice was made on held-out DFL matches by a rule written down before the
results were seen. The provided data are inputs only, and the public ground truth was used for evaluation
only.

| | spellF1 | macroF1 |
|---|---|---|
| Organisers' baseline (`baseline_smooth50`, full match) | 0.2433 | |
| Nearest-player baseline, public 5 minutes | 0.424 | 0.581 |
| **Scio.csv, public 5 minutes** (17 possessions, 90% bootstrap CI) | **0.638** [0.486, 0.768] | 0.826 [0.630, 0.890] |
| Same model family, leave-one-match-out on DFL | 0.630 | |

The public score was recorded once and was not used to choose anything. With 17 possessions, one
possession moves spellF1 by about 0.06.

## 1. Approach

### Why outside data
Phase 1 forbids training, fitting or calibrating on the ground truth or the video, and our team went
further: no model is fitted on the provided match at all, unsupervised models included. That leaves
two options: hand-written rules, or models learned elsewhere. Rules (the nearest-player baseline and
its variants) stall at about 0.42 public spellF1, because the ball is missing on 41% of graded frames. Learned models need labelled frames, and
the IDSSE dataset (Bassek et al., *Scientific Data* 12, 195, 2025) has seven full matches of official
DFL tracking at 25 Hz with `BallStatus` and `BallPossession` on every frame.

### Pipeline (`src/dfl_only.py`, `src/experiments/dfl_clean.py`)
1. **Outside labels as they are.** DFL ball status gives `D`; DFL possession gives `H` / `A`. They
   are not smoothed. CHALLENGE.md says the truth is mechanical and nothing is smoothed afterwards.
2. **Degradation** (`src/external/common.degrade`). The clean DFL tracking is made to look like ours:
   - the ball is lost in seen / unseen runs whose lengths are drawn from our tracking's own ball
     visibility over the graded frames (a label-free statistic);
   - a ball above 7.3 m is lost (our provider never reports one higher);
   - about 6% of player frames are dropped.

   Four independent degradation draws (seeds 0–3, fixed in advance) are pooled.
3. **Features** (`src/external/common.common_design`). The same code runs on DFL and on our tracking,
   on a 0.2 s grid, 246 columns:
   - team shape and pressure, and which team is nearest the ball and by what margin (`team_features`);
   - ball and player kinematics, where and how the ball was last seen, and line and goal geometry,
     with 1 / 3 / 10 s context (`ld_features`);
   - restart geometry: players walking, a wall, a taker standing over the spot (`restart_features`);
   - touch and carry events read off the tracking (`possession`).

   No broadcast, audio or Smart Tagging columns, because the outside data have none.
4. **Two models** (scikit-learn `HistGradientBoostingClassifier`).
   - *Dead model*: P(dead) on every in-play row, 190 columns.
   - *Team model*: P(home | live) on live rows, 134 columns. Every row also enters mirrored, with home
     and away exchanged and the label flipped, so "home" is no particular club.
5. **Decoder** (`src/decode.viterbi3`). A three-state Viterbi inside each half on the log-odds:
   switching between live and dead costs `cd = 2`, switching teams costs `ct = 0.75`. The path is
   expanded back to 25 fps. Frames outside the graded halves are `X`.
6. **Kick-off guard** (`src/experiments/kickoff_guard.py`). From the second half's first graded frame
   to the start of its Smart Tagging `Kick Off (Start)` tag, the ball is dead. No tracking model can
   see that stretch, and without the guard the models invent possessions in it.

### The one domain trap that mattered
In DFL a dead ball is **frozen** in place for 99.5% of dead frames and teleports at the restart. So on
DFL, ball speed separates dead from live almost perfectly (AUC 0.01), and a model trained there leans
on it. An automatic tracker never shows a still ball, so the same model would miss most
stoppages. The fix is to keep ball-motion and touch-timing columns out of the dead
model (drop group C) while keeping the player-based restart cues. That choice, and the decoder costs,
were made on DFL held-out matches **re-rendered without the frozen ball** (next section).

### How the settings were chosen (`src/clean_cv.py`)
- Leave-one-match-out over the 7 DFL matches, three seed sets, truth = raw DFL labels.
- Each held-out match is scored in three renderings:
  - `plain`, as DFL recorded it;
  - `hide`, with the dead ball unseen;
  - `taker`, with the dead ball carried by the restart taker (`src/external/view.py`).
- *Transfer score* = mean of `hide` and `taker`.
- Candidates: drop group {none, A, B, C} × cd {1, 2, 4} × ct {0.5, 0.75, 1, 1.5, 2}.
- **Pre-registered rule:** keep every configuration within 0.005 of the best transfer score, then take
  the one with the highest plain score. It picked **drop C, cd 2, ct 0.75** (transfer 0.565,
  plain 0.630).
- Dropping the frozen-ball columns costs about 0.01 on plain DFL and gains about 0.04 on transfer. Low
  team-switch costs win everywhere.

The Metrica sample games, scored for reference only, point the same way: dropping helps, C ≈ B, and a
low `ct` wins.

## 2. Assets: used, and tried and dropped

**Used in the submission:** the automatic tracking of the tactical feed (players, ball, ball height,
and the team of each track via the starter's `team_of_each_track`); the half boundaries in `games.py`;
one Smart Tagging tag (the second-half kick-off); the organisers' scorer and validator. Outside: the
seven IDSSE / DFL matches.

**Tried and dropped** (each was logged and scored; none beat the noise or survived the rules):

| asset / idea | outcome |
|---|---|
| Broadcast tracking, homographies, triangulated ball height, Smart Tagging as features, weak labels or decoder rules | no spellF1 gain; kinematics lift macroF1 only |
| Audio (whistle tonality) for dead balls | no spellF1 gain; and DFL has no audio |
| Ball from the tactical video: a classical detector (background subtraction + homography + physics-gated tracklets), then a detector trained on Kaggle GPUs on public SoccerTrack v2 | the classical one fills 75% of gaps but a third are over 5 m off, best 0.619 gated; the trained one had 46% gross fills. Most provider ball gaps are stoppages, so neither could help. |
| Event detectors on DFL: goals and kick-offs, restarts, turnovers | detect well on DFL (restart F1 0.94), no spellF1 gain |
| Domain-randomising the DFL ball (jitter, drift) | hurt the team model |
| Sequence models (TCN over a player-set encoder), TabFM, larger trees | tie or lose against the trees |
| Minimum-Bayes-risk decoding for spellF1, other decoder variants | fail the pre-registered acceptance bar on DFL |
| A small vision-language model on screenshots | no edge |
| SkillCorner, PFF, Metrica as training data | worse, no data, or no licence (Metrica: reference scoring only) |

## 3. The possession rules we implement

Our rules are DFL's official conventions, learned from their data and transferred:

- **Alive v dead.** Dead from DFL's whistle / ball-out moment to the restart kick. A set piece's
  possession begins at the kick, with the team taking it.
- **Who has it.** The team DFL credits with possession. It stays with that team through a loose ball
  or a lost duel until the other team takes control. The labels are not smoothed, so short spells are
  kept.
- **Challenges and loose balls** are settled by the decoder: a team switch needs evidence worth
  `ct = 0.75` log-odds, so a single contested frame does not flip possession, while a clean
  interception about a second long does.
- **Abstain (`X`)** only outside the graded halves (before kick-off, half-time, after the final whistle).
  Inside play every frame gets `H`, `A` or `D`.
- **Kick-off.** Dead from the start of a graded half until its tagged kick-off.

Where this differs from the organisers' annotation, as far as the public definition and the 5 public
minutes show: the public truth ends some dead spells near the first reception, not at the kick, and
places a hand-over about half a second before the new team's first touch, where DFL places it. We did
not adjust for either. Fitting that offset would be tuning on the truth.

## 4. Limitations

- **Team calls are the loss.** On the public window, for the predecessor of this model (same pipeline,
  earlier settings), perfect team calls with its dead/live would score 0.95; perfect dead/live with
  its team calls, 0.68. Most misses are short possessions (under
  5 s) and hand-overs.
- **Too many possessions.** 343 over the full match, 10.4 s mean, against 14.2 s on the public window.
  Extra short possessions in contested play cost precision.
- **The ball is missing** on 41% of graded frames. Even on DFL's clean tracking this model family
  stops at about 0.70 leave-one-match-out, and at about 0.80 with a perfect ball.
- **Seed noise is about the size of most gains.** Redrawing the degradation seeds moves the scores
  noticeably, so every comparison was made over three seed sets.
- **One provider.** DFL tracking is cleaner than ours in ways the degradation only approximates, most
  visibly the frozen dead ball. A second licensed provider with frame-level possession would be the
  first thing we add.

**With more time:** a ball detector with ~1 m accuracy on the frames where the provider loses it
during live play; a hand-over model that reads first-touch events from the video; and more licensed
outside matches.

## 5. Rule compliance

- No hand labels and no manually created data about this match. The kick-off comes from a Smart
  Tagging tag, not a typed frame number.
- No model is trained, fitted or calibrated on the provided data, the ground truth or the video. Every
  tunable setting was chosen on the outside matches (section 1). The public ground truth was used for
  evaluation only.
- The provided tracking is used as input, including two label-free statistics of it that shape the
  degradation of the outside data: the ball's seen / unseen run lengths over the graded frames and its
  maximum reported height (7.27 m, rounded up to 7.3 m).
- The drop groups follow from a property of the outside data: DFL freezes a dead ball in place and
  teleports it at the restart. The choice among them was made on DFL by the pre-registered rule.
- Outside data: IDSSE (Bassek, Rein, Weber & Memmert 2025), CC BY 4.0, downloaded from figshare by the
  code. None of it involves either club or La Liga.

## 6. Reproduction

```bash
git clone https://github.com/SiamRahman29/road-to-barca && cd road-to-barca
# put the organisers' data bundle in data/ (as in the original repository), then:
uv sync
uv run python -m src.submit          # writes Scio.csv, validates it, prints its sha256
```

- `uv sync` installs Python 3.11 and the pinned dependencies (`uv.lock`; scikit-learn 1.5.2,
  numpy 2.2.6, pandas 2.2.3, scipy 1.17.1).
- The first run downloads the seven IDSSE matches (about 2.5 GB, no account) into
  `shared/external/`; how long that takes depends on the connection. Parsing them, building the
  degraded training rows, training the models and labelling our match then takes about 10 minutes on
  12 CPU cores, with about 4 GB of free RAM. Later runs reuse `shared/cache/`.
- The output is deterministic. `src/submit.py` compares the sha256 against the submitted file's
  (`84dedecb3279d776…`), and a run from a fresh clone with an empty `shared/` reproduces it exactly.
- `uv run python -m src.clean_cv --seedsets 01,23,45` reruns the settings selection on DFL (about 2 h).
