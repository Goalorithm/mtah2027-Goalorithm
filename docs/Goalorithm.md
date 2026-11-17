# Goalorithm: technical document (More than a Hack 2027, Phase 1)

Team Goalorithm · Repository: <https://github.com/Goalorithm/mtah2027-Goalorithm> (branch `main`)

Submitted file: `Goalorithm.csv`, 199,319 rows, SHA-256
`10fbac793413867cafaa5bb5dd6739e52c59e8d1429828463adb60370a9bc199`.

## 1. Summary

The competition tracking is automatic. The organisers report the ball missing on 41% of graded
frames and about 20.5 tracked objects per frame, and corrected public tracking looks nothing like
that: IDSSE freezes the ball wherever play stopped, Metrica blanks it, and the ball is otherwise
always there. A model trained on such data learns the provider's habits instead of the game.
So we take labelled matches from other fixtures, re-measure them until they behave like automatic
tracking, train a temporal convolutional network (TCN) on those copies and decode its output into
possessions with Viterbi. Three seeds are refitted and averaged. Every weight, decoder setting and
selection comes from external matches; at inference the model reads only the provided automatic
tracking, with no video, no broadcast recording and no Smart Tagging.

We found out how much this matters the hard way. Our previous final model, a TCN blended with
gradient-boosted trees and trained on corrected IDSSE tracking, scored 0.80 on clean IDSSE test
input but only 0.434 once the same test matches were re-measured like automatic tracking. The
model submitted here scored 0.714 on exactly the same input (Section 6).

## 2. Why a sequence model, and why not rules

spellF1 treats a possession as one object and matches it by temporal IoU, so a single wrong flip
in the middle of a long spell costs two objects while a per-frame metric would barely register
it. That's the segmental-F1 view of action segmentation [7, 8]. We decode a whole half at once,
with an explicit model of which state can follow which.

Rule-based possession detection from tracking works when the ball is tracked and the data carries
an in-play flag; Vidal-Codina et al. [1] name both as the main limitation of their method [1, §4].
Our data has neither, and our rule variants (`out_of_bounds`, `set_piece_restart`,
`hold_to_switch`) and a hand-specified HMM never came close to the learned models (Section 6.2).

## 3. Approach

### 3.1 Re-measuring external matches like automatic tracking (`tools/atd_sim.py`)

Each labelled external match becomes several randomly measured copies before training:

- **Ball losses.** During live play the ball disappears in stretches drawn from a renewal process
  that ignores the labels. At stoppages it is re-synthesised: it rolls on, is fetched and handled,
  is carried or kicked back to the restart spot, waits, and then is missing, visible or partly
  visible. This removes the frozen-ball shortcut that our earlier models had learned.
- **Smoothing.** The ball track goes through a zero-phase low-pass filter.
- **Players.** Players drop out by tracklet and in shared outages.
- **Ranges, not single values.** Every parameter is drawn per copy from a range in
  `config/atd.toml`; validation and test copies use the middle of each range.

Where did the ranges come from? From the organisers' published figures (41% ball loss, about 20.5
tracked objects) and, for some of them, from label-free descriptive statistics of the provided
tracking: typical ball-gap lengths, how smooth the ball track is and how a tracked ball moves at
stoppages. No label, no video and no ground truth of the provided match went into them, and no
model weight or decoder setting was fitted on it. We checked whether the result depends on those
statistics (Section 6.1): replacing the stoppage-ball range that came from them with a wide
uninformed one, or switching that behaviour off, moved the test score by at most 0.01.

### 3.2 Label sources and the possession definition

CHALLENGE.md says a touch without control changes nothing and that "in phases where the ball is
contested or loose, possession stays with the last team that had it". SoccerTrack v2 marks team
control and contested phases separately, so its contested frames go to the last team that had the
ball. IDSSE and Metrica references are used as their providers define them.

### 3.3 Model, decoder and ensemble (`tools/atd_features.py`, `tools/train_atd.py`, `tools/ensemble_atd.py`)

- **Inputs:** 118 features per frame from the automatic ball and the players, among them each
  team's distance and approach to the ball over several windows, ball speed, how long the ball has
  been missing or resting, and team shape and movement. Attack direction is read per half, so the
  features are symmetric in H and A.
- **Network:** a non-causal TCN with 96 channels and dilations 1 to 128 (about ±20 s of context).
  One head predicts the state (H, A or D); the other predicts change events.
- **Training data:** 3 IDSSE matches and 2 Metrica sample matches (65% of samples) and 6
  SoccerTrack v2 matches (35%), each as several randomly measured copies.
- **Decoder:** Viterbi over the three states, with transitions counted on the IDSSE and Metrica
  training references. The emission scale and event weight were chosen on validation matches from
  all three sources, each source weighted equally.
- **Ensemble and refit:** three seeds fix the epoch count and the decoder on validation; three
  seeds are then refitted on training plus validation matches with both settings frozen, and
  their probabilities are averaged before decoding.

## 4. Assets used and dropped

| Asset | Final system | Notes |
|---|---|---|
| Tactical automatic tracking (`_atd.xml`, `_atd.txt`) | Used | The only input; the run log reports `video_ball_share: 0.0` |
| Tactical video | Not used | Integrity check only. An earlier ball detector built from this match's video was removed after the organisers' 3 November clarification |
| Homography | Not used | The tracking is already in pitch metres |
| Broadcast video, tracking and homography | Not used | A broadcast-cutaway feature exists in the code but is switched off |
| Smart Tagging | Not used | Only the HMM and rule baselines read it |
| Public 5-minute ground truth | Correctness check only | Scored after the file was built; never used to fit, select or tune |

## 5. Training data

None of it comes from the provided match, and we didn't create or correct a single label.

| Dataset | Labels, and who produced them | Amount | Role | Licence |
|---|---|---|---|---|
| IDSSE, DFL/Sportec official tracking [9] | Provider channels `BallStatus` and `BallPossession` (DFL/Sportec) | 7 matches: 3 training, 2 validation, 2 test | Train, select, test | CC BY 4.0 |
| Metrica Sports sample games 1–3, pinned commit `e706dd5` | Derived mechanically from the provider's data: dead where the corrected ball is blank, otherwise the team of the latest owning event | 3 matches: 2 training, 1 validation | Train, select | Public repository without an explicit licence; its README asks for responsible use |
| SoccerTrack v2 [10] | Provider's per-frame possession state | 8 matches: 6 training, 1 decoder validation, 1 test | Train, select, test | CC BY 4.0 (Hugging Face, terms accepted) |

## 6. Results (external matches only)

All figures come from the organisers' scorer (mean spellF1 over IoU 0.3–0.7) on matches re-measured
like automatic tracking, using the middle of every range. We had already scored the IDSSE test
matches with earlier models, so treat those rows as a re-evaluation rather than an untouched holdout.

| Check | spellF1 | macroF1 |
|---|---|---|
| IDSSE test J03WMX and J03WN1, 4 copies | 0.714 | 0.933 |
| SoccerTrack v2 match 117092, never trained or selected on (contested frames to the last team) | 0.756 | 0.926 |
| SoccerTrack v2 match 117092, contested frames not graded | 0.818 | 0.950 |
| Validation used to choose the decoder: IDSSE / Metrica / SoccerTrack | 0.743 / 0.737 / 0.820 | |

The three seeds scored 0.737, 0.738 and 0.736 on validation before averaging.

### 6.1 Against our previous final model, on the same input

We scored both models with the same evaluator on the same four re-measured IDSSE test copies
(`tools/evaluate_atd.py --legacy-dir`), on a second laptop:

| Test input (IDSSE J03WMX + J03WN1, 4 copies) | This model | Previous final: TCN + trees |
|---|---|---|
| As configured | **0.714** | 0.434 |
| Stoppage ball never handled (the range set from tracking statistics unused) | **0.705** | 0.447 |
| Stoppage-ball pace drawn from a wide 0.5–10 m/s range | **0.714** | 0.434 |

The previous model scored 0.802 on clean IDSSE input, so nearly all of its apparent strength came
from the frozen-ball convention. Its predecessor, a TCN trained on clean IDSSE alone, scored 0.385
here.

### 6.2 Everything else we tried

Before the re-measurement existed we compared 20 models on IDSSE test input that kept the
provider's frozen ball, only adding synthetic ball dropout and jitter. Those numbers rank the
models against each other but overstate how well any of them handles automatic tracking.

| Candidate | Clean | Ball dropout | Dropout + jitter |
|---|---|---|---|
| Organisers' baseline (nearest player, majority vote) | 0.296 | 0.248 | 0.244 |
| Rules: sustained close control (± guarded abstention) | 0.273 | 0.207 | 0.192 |
| Hand-specified HMM (v1, v2) | 0.278 | 0.233 | 0.213 |
| Rules: dead ball from outside-pitch evidence, control rules | 0.365–0.366 | 0.275–0.291 | 0.286–0.301 |
| Gradient-boosted trees, 26 features (v1) / with augmentation (v2) | 0.488 | 0.433–0.436 | 0.242 / 0.427 |
| Trees with player–ball interaction features (v3, v4, balanced) | 0.632–0.657 | 0.564–0.572 | 0.552–0.569 |
| Trees + handover classifier ("joint"; refit, calibrated, dense variants) | 0.713–0.720 | 0.600–0.605 | 0.587–0.594 |
| Joint mixed with a random forest | 0.719 | 0.618 | 0.603 |
| TCN + trees (previous final) | 0.802 | 0.688 | 0.675 |

## 7. Possession rules implemented

- **Dead ball (D)** runs from the moment play stops until the restart is taken, as the providers
  mark it.
- **Team (H or A)** during live play is the team that last controlled the ball; contested and loose
  balls stay with that team.
- **Touches without control** don't change the label unless the network sees sustained evidence
  for the other team.
- **Restarts:** the restarting team's possession begins when the ball is back in play.
- **A missing ball isn't a dead ball.** The model decides from player movement, how long the ball
  has been missing and the surrounding context.
- **X** appears only outside the two graded halves; inside them spellF1 would forward-fill it anyway.

## 8. Limitations and what we'd do next

We don't know the score on BAR-ATH. Every figure above comes from other matches re-measured to
resemble automatic tracking, and anything the re-measurement doesn't capture, such as identity
switches between nearby players, isn't covered.

Short possessions and duels remain the main loss: about one possession in four is missed or
wrongly bounded on IDSSE labels, mostly those under two seconds, and when players of both teams
are within reach of the ball, positions alone can't say who controlled it.

Label conventions differ by provider. IDSSE records possession more finely than the competition's
definition, which is why the decoder is chosen on three sources weighted equally.

Some simulation ranges came from label-free statistics of the provided tracking (Section 3.1).
The sensitivity check suggests the result doesn't hinge on them, but a cleaner version would set
every range from external data alone, and that's the first thing we'd change.

With more time we'd train a ball and player-contact model on footage of other matches to decide
duels, and look for more labelled matches with automatic tracking.

## 9. Reproduction

Python 3.11, with every dependency pinned using `==`. Unpack the tactical tracking of the bundle
into `data/` (or set `MTAH_DATA_DIR`). The trained weights are released in `release/atd/`, so no
training is needed:

```
python -m pip install -r requirements-team.txt
python -m pip install -r requirements-temporal.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python tools/restore_release.py
python tools/evaluate_atd.py --model-dir outputs/atds_refit_ensemble --mode competition --name Goalorithm
```

This writes `outputs/Goalorithm.csv` and prints its SHA-256. The file is always computed on CPU,
the evaluator runs the prediction twice and refuses to write a file if the runs differ, and two
different laptops produced the same SHA-256 (`10fbac79…c199`). Retraining from the external data:
`run_all_tech.cmd` for the IDSSE and Metrica downloads, then `run_atd.cmd`.

The previous final model stays reproducible with `python tools/predict_temporal.py`
(`dd610799b8ad715d7371c1dfbc783ba0a484eeb9784696c79f93e14721822af3`); see `models/README.md`.

## 10. Rules compliance

- No human labels or human-produced data about the provided match, in any form.
- No weight, decoder setting or model choice fitted, tuned or selected on its ground truth, its
  video or its tracking. Some simulation ranges were set from label-free descriptive statistics of
  its automatic tracking (Section 3.1); we disclose that here, and the sensitivity check in
  Section 6.1 shows the result barely depends on them.
- Predictions were never checked against the video, and the repository no longer contains the
  video-review and label-suggestion tools.
- No competition data went to any AI service; everything ran locally.

## References

1. F. Vidal-Codina, N. Evans, B. El Fakir, J. Billingham. Automatic event detection in football
   using tracking data. *Sports Engineering* 25:18, 2022. doi:10.1007/s12283-022-00381-6
2. D. Link, M. Hoernig. Individual ball possession in soccer. *PLOS ONE* 12(7):e0179953, 2017.
   doi:10.1371/journal.pone.0179953
3. H. Kim, H.-J. Choi, C. J. Kim, J. Yoon, S.-K. Ko. Ball trajectory inference from multi-agent
   sports contexts using Set Transformer and hierarchical Bi-LSTM. *KDD* 2023.
   doi:10.1145/3580305.3599779
4. L. R. Rabiner. A tutorial on hidden Markov models and selected applications in speech
   recognition. *Proc. IEEE* 77(2), 1989. doi:10.1109/5.18626
5. G. Ke et al. LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS* 2017.
6. Y. Abu Farha, J. Gall. MS-TCN: multi-stage temporal convolutional network for action
   segmentation. *CVPR* 2019. doi:10.1109/CVPR.2019.00369
7. C. Lea, M. D. Flynn, R. Vidal, A. Reiter, G. D. Hager. Temporal convolutional networks for
   action segmentation and detection. *CVPR* 2017. doi:10.1109/CVPR.2017.113
8. S. Li, Y. Abu Farha, Y. Liu, M.-M. Cheng, J. Gall. MS-TCN++. *IEEE TPAMI* 45(6), 2023.
   doi:10.1109/TPAMI.2020.3021756
9. M. Bassek, R. Rein, H. Weber, D. Memmert. An integrated dataset of spatiotemporal and event
   data in elite soccer. *Scientific Data* 12:195, 2025. doi:10.1038/s41597-025-04505-y
10. SoccerTrack v2, Scott et al., arXiv:2508.01802 (as cited by the dataset card).
