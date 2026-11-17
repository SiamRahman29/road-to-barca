# Outside data

## Source

**IDSSE**: M. Bassek, R. Rein, H. Weber and D. Memmert (2025), *An integrated dataset of
spatiotemporal and event data in elite soccer*, Scientific Data 12, 195,
doi:10.1038/s41597-025-04505-y. Licence **CC BY 4.0**, hosted on figshare.

It has seven matches from the 2022/23 season: two from the Bundesliga and five from the
2. Bundesliga. Each has official DFL position data at 25 Hz for every player and the ball, including
ball height. Every ball frame also carries the official `BallStatus` (alive / dead) and
`BallPossession` (home / away).

**Not the competition fixture.** None of these matches involves Barcelona or Athletic Club, and none
is from La Liga. The rules forbid "any other source of corrected possession data for this fixture";
this is a different competition, different teams and a different season.

`python -m src.external.idsse` downloads the files (figshare file ids are in `idsse.FILES`) into
`shared/external/idsse/` and caches a parsed copy. It needs no account and no credentials.

## How it is used

1. **Degradation** (`common.degrade`). The DFL tracking is corrected and complete; ours is automatic.
   The ball is lost in seen / unseen runs whose lengths are drawn from ours, separately in live and
   dead play. Balls above 8 m are dropped, and about 6% of player frames are lost.
   - Calibration over the seven matches: in live play the ball is visible 80% of the time against
     our 87%, and 20% against our 23% in dead play. Players tracked per frame: 20.6 against our 20.7.
   - The only thing taken from the competition data is these aggregate run-length distributions.
     They do depend on the hand labels, which split live from dead play.
   - They are frozen once in `shared/derived/dfl_profile_<labels>.npz` (`python -m src.external.common`),
     so correcting a hand label does not change the degraded rows.
2. **Possession convention** (`idsse.controlled`). DFL possession changes on almost every touch:
   4.6 possessions per minute, 22% of them under 1.5 s. The organisers' convention is controlled
   possession: about 3.0 per minute, 11% under 1.5 s.
   - Possession runs shorter than 1.5 s are merged into the team before them. This gives 3.3 per
     minute and 43% under 5 s, against our 42%.
3. **Pooled team model** (`had_cv.pooled_team_logit`, experiment `had_ext`). Our labelled frames
   and the outside frames train one model.
   - Every outside frame also enters mirrored, with home and away exchanged and the label flipped.
     "Home" in the outside data is no club we know, so the model can't learn any one team's habits
     from it.
   - Outside frames carry a source column and weight 0.3. Predictions are made with the source set
     to ours.

4. **Pooled dead-ball model** (`had_cv.pooled_dead_logit`, `had_ext` `dead_pool`). A second
   dead-ball model is trained on the shared features (`common.common_design`). Its training rows are
   our labelled frames plus the degraded DFL frames with their official ball status, at weight 0.3
   with a source column. It is averaged with our main dead-ball model, which keeps the broadcast and
   audio inputs that DFL lacks.

## Evidence (4-fold leave-one-stretch-out CV, labels d5ebe3c7, 130 possessions)

Each variant is scored at its own best decoder costs. The "mean" column averages over 12 cost
settings, which is less flattering.

| variant | best spellF1 | mean | macroF1 |
|---|---|---|---|
| no outside data | 0.614 | 0.601 | 0.867 |
| second dead model **without** DFL rows (control) | 0.617 | 0.608 | 0.866 |
| dead model averaged with a DFL-pooled one, degradation seed 0 / 1 | 0.633 / 0.631 | 0.617 / 0.616 | 0.869 / 0.868 |
| + DFL-pooled team model averaged over 3 degradation seeds (`had_ext`, e224) | 0.636 | 0.619 | 0.874 |

- **The team model's spellF1 depends on the degradation seed.** With one seed, CV ranged 0.597–0.637
  (seed 0 gave the high end). That's why e224 onwards set `team_seeds=0,1,2`; the `had_ext` default
  is one seed.
- **Robust across seeds:** team log-loss 0.310 → 0.203 (−35%) and macroF1 +0.005.
- **The degradation matters:** clean (undegraded) DFL rows give 0.608, against 0.637 with the same
  seed-0 setting.
- **Failed:** DFL-only priors used as stacked features scored 0.597. Models trained on DFL alone,
  with none of our labels, score 0.49.

The gain is about +0.02 spellF1, consistent across seeds and controls, but inside the CV interval
(±0.07).
