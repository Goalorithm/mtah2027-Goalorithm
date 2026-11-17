# Project context for Claude Code

Team Goalorithm's entry to More than a Hack 2027, Phase 1: label every frame of the tactical video
(199,319 frames, 25 fps) of FC Barcelona (H) vs Athletic Club (A) as H, A, D (dead ball) or X
(abstain). Ranked on spellF1 (possession-level), not per-frame accuracy. Read CHALLENGE.md and
EVALUATION.md before proposing methods. The team speaks Bahasa Indonesia; answer in Indonesian
unless asked otherwise. Code, comments and commit messages are in English.

## Hard rules

- The match data is licensed for this competition only: never copy it out of `data/` and never
  commit `data/`, `cache/`, `outputs/`, videos or `.npz` files. The repo becomes public at submission.
- Submission labels must come from an algorithm. Updated upstream rules at `f8f2629` prohibit
  human labels on the provided game for any purpose, including training and validation.
  Do not create or import them. See `docs/RULE_UPDATE.md`; the old manual-label workflow is retired.
- Do not modify `evaluation/`, `starter/` or `games.py`: they are the organisers' code, kept
  as shipped so `git pull upstream main` never conflicts. Import from them via `mtah.paths`.
- Do not tune or choose between versions on the public 5-minute window (17 possessions).
  Local AI pseudo-labels are algorithm outputs, not independent ground truth. Measure accuracy
  on permitted labelled data from other games or through the organisers' evaluation.
- Output must be deterministic: no unseeded randomness. Same inputs, byte-identical CSV.

- Fit and calibrate models only on external matches. Do not restore excluded research listed
  in `.github/excluded-paths.txt`; run `python tools/check_repository_content.py` before publishing.
- Historical outputs and checkpoints must be rebuilt against the retained source before use.

## Structure and conventions

- Pipeline: `src/mtah/` — `load.py` (cached tracking) → `features.py` → `dead_ball.py` →
  `possession.py` → `predict.py`. Arrays are indexed by video frame (index 0 = submission row 0),
  positions in metres, NaN for missing. Label codes: H=0, A=1, D=2, X=3 (`from mtah import H, A, D, X`).
- New ideas are new methods registered in `METHODS` in `dead_ball.py` / `possession.py`,
  selected via `config/*.toml`. Keep existing methods; the baseline must stay runnable.
- Every tunable number goes in `config/`, read from the `cfg` dict, never hard-coded.
- New libraries: pin with `==` in `requirements-team.txt`.

## Commands

```
python -m mtah.predict --name exp_x --set section.key=value   # run, validate, score
python tools/cek_setup.py                                     # environment + baseline parity
python tools/timeline.py --window publik --pred baseline_smooth50 exp_x
python tools/match_report.py baseline_smooth50 exp_x
pytest -q && ruff check src tools tests scripts && ruff format src tools tests scripts
```

## Workflow

Branch from `dev` (the default, staging branch) as `<name>/<topic>`, one idea per PR into `dev`,
fill the before/after table in the PR template. `dev` and `main` are protected: PR + 1 approval
from another member + green CI. Only `dev` is merged into `main` (release, merge commit).
