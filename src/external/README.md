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

`idsse.match(match_id)` downloads the files on first use (figshare file ids are in `idsse.FILES`) into
`shared/external/idsse/` and caches a parsed copy. It needs no account and no credentials.

## How the submission uses it

1. **Labels**: `BallStatus` dead gives `D`, `BallPossession` gives `H` / `A`, taken as they are
   (tags ending `raw` in `src/dfl_only.py`).
2. **Degradation** (`common.degrade`): the DFL tracking is corrected and complete, ours is automatic.
   The ball is lost in seen / unseen runs whose lengths are drawn from our tracking's ball visibility
   over the graded frames (`dfl_only.free_profile`, no labels); a ball above the provider's own
   ceiling (`dfl_only.CEILING_M`, 7.3 m) is lost; about 6% of player frames are lost.
3. **Rendering for selection only** (`view.py`): DFL freezes the dead ball, ours never does. For
   choosing settings (`src/clean_cv.py`), held-out matches are also scored with the dead ball hidden
   (`hide`) or carried by the restart taker (`taker`). Training rows are never re-rendered.
4. **Models**: `src/dfl_only.fit_models` trains the dead-ball and team models on the degraded rows
   of all seven matches and four degradation seeds. Team rows also enter mirrored (home and away
   exchanged, label flipped), so "home" is no club.
