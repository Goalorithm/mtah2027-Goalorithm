# Tech 1: HMM sequence candidate

The bounded-memory v2 candidate and its regression results are documented in
[HMM_V2.md](HMM_V2.md). This page records the original v1 candidate.

**Rules update checked 11 November 2026:** human annotations on this game are prohibited for
training and validation too. The original manual dev/holdout plan below is archived. See
[RULE_UPDATE.md](RULE_UPDATE.md). `tools/ai_labels.py` generates local algorithmic pseudo-labels;
these are not independent ground truth and cannot establish this candidate's accuracy.

This candidate jointly decodes home possession (`H`), away possession (`A`), and dead ball (`D`).
It is selected by `config/exp_hmm_v1.toml`. The default config and the organisers' baseline remain
unchanged. No dependency was added to the inference environment.

## Model

`src/mtah/hmm.py` implements log-space Viterbi with NumPy. Transition rows represent the previous
state and columns the next state. Dynamic programming stores the best predecessor of each state,
then backtracks the best complete path. Scores are rescaled at every step to keep long sequences
numerically stable. Ties use the lowest state index. Zero-probability transitions are forbidden.

This is an HMM-style sequence decoder with hand-specified evidence potentials, not a trained
generative model. The observation evidence uses distance to the nearest player of each team,
ball visibility, the last known ball position relative to the pitch, player motion, ball motion,
and weak automatic phase tags. A far-away nearest player gives weaker team evidence. Defensive
phase tags favour the opposite possessor. Unknown team tags and set-piece intervals do not become
team labels. Overlapping phase tags are averaged so their count does not multiply their weight.

All evidence weights, geometric and speed thresholds, interpolation limits, starting probabilities,
and transition probabilities live in `[hmm]` in the experiment config. `emission_scale` controls
the strength of correlated per-frame evidence relative to switch costs. Strong state persistence
reduces changes caused by brief touches and tracking noise, but can also merge real possessions.

The two published graded half spans are decoded independently, with a fresh initial distribution
for each half. Pre-match, halftime, and post-match frames are `X`. Within graded halves the decoder
outputs `H`, `A`, or `D`. No public/private label, manual annotation, or ground-truth statistic is an
inference input.

The `hmm` adapters in `dead_ball.py` and `possession.py` share one decode per feature dictionary.
Both read the same top-level `[hmm]` section. Existing method inputs, outputs, and baseline methods
are preserved. Cached results are local to a single feature dictionary and keyed by config and
input object identities. Inputs must remain unchanged during that pipeline run.

Reference for the dynamic programming algorithm:
[Jurafsky and Martin, Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/A.pdf).

## Ball history and interpolation

The independent helper `hmm.ball_features()` returns:

| Key | Meaning |
| --- | --- |
| `ball_last_xy` | Last complete observation, forward-filled inside each half |
| `ball_gap` | Frames since that observation; `0` when observed, `-1` before any observation |
| `ball_xy_filled` | Raw observations plus linear interpolation of eligible short gaps |

Interpolation requires two observed endpoints in the same half, at most `max_gap` missing frames,
and endpoint speed no greater than `max_interp_speed_mps`. Leading, trailing, long, or implausible
gaps remain `NaN`. Missing or nonfinite coordinates are not complete observations. Raw tracking and
existing feature keys are never modified. This follows the independent Tech 1 lane in the supplied
assignment screenshot; Tech 2 and Tech 3 do not need to change their features or methods to use the
baseline.

## Reproduction

From the repository root, activate the existing Python 3.11 environment:

```powershell
.venv\Scripts\Activate.ps1
python tools/cek_setup.py
python -m mtah.predict --config config/exp_hmm_v1.toml --name exp_hmm_v1
python -m mtah.predict --config config/exp_hmm_v1.toml --name exp_hmm_v1_repeat --no-score
python evaluation/validate_submission.py outputs/exp_hmm_v1.csv
python tools/match_report.py baseline_smooth50 exp_hmm_v1
```

`data/tactical` must contain the tactical ATD XML/TXT and Smart Tagging XML. On the current laptop,
it is a directory junction to the existing downloaded bundle. The junction is local and ignored
by Git; another laptop must supply its own data or configure `MTAH_DATA_DIR`.

## Validation and current measurements

Measured on 11 November 2026 using the provided tracking and the fixed candidate config:

| Measurement | Baseline smooth50 | HMM candidate |
| --- | ---: | ---: |
| Public spellF1 | 0.3256 | 0.4286 |
| Public macroF1 | 0.6207 | 0.6846 |
| Full-match predicted possessions, graded halves | 594 | 237 |
| Mean predicted possession duration, seconds | 7.2 | 17.0 |
| Predicted possessions shorter than 2 seconds | 186 | 6 |
| Dead-ball share on graded frames | 32.3% | 36.5% |
| Automatic set pieces near a predicted stoppage | 51 / 59 | 54 / 59 |
| Abstentions on graded frames | 0% | 0% |

The set-piece statistic uses the existing `match_report.py` diagnostic; it is not a measured
restart-timing accuracy. Its tag intervals extend around the restart. Full-match possession
statistics describe predictions, not agreement with private ground truth.

The first unscaled-evidence prototype produced 753 possessions and 294 spells shorter than
2 seconds. It was rejected using these no-ground-truth fragmentation diagnostics. Evidence was
then softened with `emission_scale = 0.1`. The public score was inspected after this change;
no public-score sweep or holdout tuning was performed.

The public window has only 17 true possessions; this candidate predicts 11 there. Its apparent
public improvement is not evidence of full-match superiority. Reviewed dev annotations are absent,
so this config has not replaced `config/default.toml` and no final model has been released.
With 1,000 seeded possession-bootstrap draws, the public spellF1 5th-95th percentile interval is
0.2286-0.6143 for HMM and 0.2140-0.4465 for baseline. The intervals overlap substantially.

Two real-data runs produced the same CSV SHA-256:

```text
48f403095696ac83a60a06787c1320eff2db5c8ca95a7f81ffc8c44752773e86
```

Tests cover Viterbi against exhaustive path enumeration, impossible paths, long sequences and
ties, interpolation boundaries and missingness, single-touch suppression, sustained control,
dead-ball evidence, defensive tags, half isolation, shared decoding, and byte reproducibility.

## Archived manual-label plan — superseded by RULE_UPDATE.md

Do not run the manual annotation/selection commands below on BAR-ATH. The six windows now have
AI pseudo-labels: 45,000 frames and 108 start/change points. Run `python tools/ai_labels.py`.
Output hashes and provenance are in `outputs/ai_labels/manifest.json`; disagreement intervals
are in `intervals_ai.csv`. The pseudo-label CSV schema is distinct from human-label imports.
No reference labels or previous prediction files are read. These outputs do not select a winner.

The designated six windows are in `labels/windows.csv`: four dev and two holdout windows.
The current workbook and connected Sheet contain no change-point annotations.

```powershell
# Check reviewed annotations, then build official-length validation files.
python tools/labels_to_csv.py --check-only
python tools/labels_to_csv.py

# Choose on dev only; inspect holdout only for the already-selected candidate at the final decision.
python tools/compare_models.py baseline_smooth50 exp_hmm_v1
python tools/compare_models.py --holdout
```

Importer exit codes: `0` means complete, `1` means malformed/unreadable, and `2` means structurally
valid but annotations are missing. Helper formula columns outside the annotation inputs are
ignored. Formulas inside annotation inputs are rejected because cached values may be stale.
An empty workbook cannot erase existing annotations. The converter validates every annotation
and prepares all split arrays before writing, so missing holdout annotations cannot leave a newly
generated dev split behind. Missing labels are an explicit incomplete selection, never a zero score.

Selection requires exactly the designated dev frames; partial coverage or holdout frames in a dev
CSV are rejected. The saved decision pins SHA-256 hashes of the dev CSV, window definitions, and
each candidate. The separate `--holdout` command requires that decision and unchanged inputs; it
does not select again. An identical repeated command reprints the saved final result. A failed
check preserves the existing decision. Dev selection cannot overwrite a completed holdout check
at the same output path. These checks protect the local workflow, not against deliberate editing
of receipt files or use of a different output path.

Model outputs must not be used as model-selection ground truth: doing so would measure agreement
with the candidate itself. The former suggestion export and local video review tools were removed:
the organisers' rules prohibit human labels on this match and checking outputs against the video
(see [RULE_UPDATE.md](RULE_UPDATE.md)). The tactical video has only passed a full decode and
frame-timestamp integrity check; see [DATA_AUDIT.md](DATA_AUDIT.md).

## Limitations and release conditions

- Tracking loss is not equivalent to a stoppage, and automatic phase tags can be wrong.
- Persistent state priors can remove real short possessions and delay transitions.
- Nearest-player geometry does not prove control, especially on aerial balls and deflections.
- Short-gap linear interpolation does not model acceleration, collisions, or the ball's height.
- Human dev/holdout labels on BAR-ATH are prohibited under the updated rules. Tech 2/3 rule
  candidates are implemented locally; see [README tim](../README.md).

Before release, integrate available rule candidates, assess accuracy on permitted other-game
labelled data or through the organisers, and reproduce the selected CSV on a second laptop.
Follow the PR/review/CI flow from dev to main. Do not create human BAR-ATH validation labels.
