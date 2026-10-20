# More than a Hack 2027 — Phase 1: Ball Possession

## Competition facts
- Task: label all 199,319 tactical-video frames `H`/`A`/`D`/`X` (home/away possession, dead ball, abstain). Barcelona (H) vs Athletic Club (A).
- Ranked on `spellF1` (possessions matched by temporal IoU, mean over 0.3–0.7); `macroF1` is the tie-break. Defined in `evaluation/possession_metrics.py`, which is authoritative over every document.
- Floor to beat: `baseline_smooth50`, full-match spellF1 0.2433.
- Ground truth: the public 5 minutes only (frames 11185–18684, 17 possessions). The full-match truth is private. **No leaderboard and no feedback before the deadline.**
- Deadline 2026-11-17. Deliverables: `<team>.csv`, a **public** repo that regenerates that exact file, a technical document.
- Rules: any method and any external data, but labels must come from an algorithm (no hand annotation). No use of the private ground truth. Teams of 3–5.
- Hardware: 12 CPU cores, 7 GB RAM, no GPU on this machine. GPU work runs remotely (see below).

## Match data: never into an LLM's context
The data terms forbid making the videos, datasets, "or any extracts thereof" available to a hosted LLM, and Claude is one.
- Never read, print, head, grep or plot anything under `data/`, `shared/cache/`, `shared/derived/` or a run's `submission.csv`, by any tool, including Bash. `.claude/settings.json` denies the Read tool there; the rule covers everything else.
- Code may process the data; what comes back to the conversation is aggregates only: scores, counts, shares, durations, shapes, error messages. Never per-frame values, coordinates or labels.
- The public ground truth CSV and everything else in the organisers' repo are public and fine to read.
- Never commit the data or derived per-frame arrays. The submission repo is public.

## The harness — frozen after review
`harness/` wraps the organisers' scorer; it does not re-implement it. Frozen at tag `harness-v1`; `.claude/settings.json` denies edits to it. Changing it is the user's call, and a change re-runs `harness.selftest`. `evaluation/`, `games.py` and `starter/` are the organisers' code and are never edited.

| command | what it does |
|---|---|
| `uv run python -m harness.run <exp> -d "..." --parent eNNN` | predict → write → validate → score public window with bootstrap CI → full-match stats → Smart Tagging diagnostics → append to `shared/log.csv` |
| `... --set k=v` / `--grid k=a,b,c --jobs 3` | override params / run a parameter grid in parallel, one log row per variant |
| `... --scratch` | try without logging (dirty tree allowed), output in `shared/runs/_scratch/` |
| `uv run python -m harness.report` | log table, best public spellF1 first (`--last N` for recent) |
| `uv run python -m harness.report --compare eA eB` | where two runs disagree, as aggregates |
| `uv run python -m harness.report --note eNNN "..."` | attach what you learned to a run |
| `uv run python -m harness.repro eNNN` | regenerate a run from its commit in a fresh worktree and compare sha256 |
| `... harness.repro eNNN --cold` | the same with an empty `shared/` (no cache, no derived): the organisers' fresh-clone situation |
| `uv run python -m harness.selftest` | harness regenerates the organisers' baselines byte for byte and matches their published scores |

An experiment is a module in `src/experiments/` with `PARAMS`, `TUNED_ON_PUBLIC` and `predict(params) -> 199319 labels`. Copy `baseline_nearest.py`. Parsed tracking comes from `src/data.py:tracking()` (cached).

## Judging results
- The public window has 17 possessions; one possession moves spellF1 by ~0.06. **Never choose between two runs whose public CIs overlap on the public score alone.** EVALUATION.md shows tuning on it picks the wrong model even for one smoothing parameter.
- Read the diagnostics with the score: full-match possession count and mean duration (the public window averages 14.2 s per possession), `tag` (agreement with Smart Tagging team phases), `rstrt` (tagged set pieces with a dead → alive transition nearby). They are noisy second opinions, not targets.
- Anything fitted or chosen on the public 5 minutes sets `TUNED_ON_PUBLIC = True`; its public score is then contaminated and marked `*` in the report.
- `X` costs nothing mid-possession on spellF1 (the scorer forward-fills it) but carries the old label across a hand-over.

## Experiment workflow
- One idea per branch; parallel ideas in separate git worktrees. Data, cache and log are shared through `harness/paths.py`.
- Commit, then run. Log every run, including failures. Set `--parent` so the log reads as a tree, and add a note with what was learned.
- Agents explore and log; **the user decides what merges to main**, and picks and submits the final file.
- Runs must be deterministic: `harness.repro --cold` passes before anything is a submission candidate.

## Remote GPU (Kaggle, Colab, others)
- Not set up yet. Uploading match data to any GPU service needs the user's go-ahead after checking the data terms; never upload it on your own.
- Disable AI assistants in those notebooks (Colab's Gemini, Kaggle's) while match data is attached — they are hosted LLMs too.
- GPU outputs come back into `shared/derived/<name>/` with a manifest (producing commit, notebook version, sha256 of files) and are consumed by CPU experiments in `src/`.
- The public repo must still regenerate the submission, so any GPU stage needs its own script in the repo and a determinism story.
