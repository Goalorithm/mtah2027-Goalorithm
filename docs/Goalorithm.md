# Goalorithm: technical document (More than a Hack 2027, Phase 1)

Team Goalorithm · Repository: <https://github.com/Goalorithm/mtah2027-Goalorithm> (branch `main`)

## 1. Summary

Every frame of the FC Barcelona (H) vs Athletic Club (A) tactical video gets one label, H, A or
D, and we write X only outside the two halves. The final system reads nothing but the automatic
tracking. It turns each frame into geometry and motion features and scores them twice: once with
gradient-boosted trees and once with a temporal convolutional network (TCN) that sees about five
seconds either side. We average the two, then decode each half with Viterbi so that a possession
has to earn both its start and its end. Both models were trained on public Bundesliga matches
(IDSSE, CC BY 4.0) and nothing else.

We didn't fit, tune, calibrate or select anything on the provided match, its video, its public
5-minute ground truth or any human-produced data about it.

## 2. Why a sequence model

spellF1 treats a possession as one object and matches it by temporal IoU, so a single wrong flip
in the middle of a long spell costs two objects while a per-frame metric would barely register
it. That's the segmental-F1 view of action segmentation [7, 8], and it explains why the
organisers' baseline (nearest player, then a majority vote) fragments so badly. We decode a whole
half at once, with an explicit model of which state can follow which.

Could hand-written rules do the job? Vidal-Codina et al. [1] detect most event types above 90%
with a deterministic decision tree on tracking, but their method assumes a tracked ball and an
in-play flag, and they name both as its main limitation [1, §4]. Our data has neither: the ball is
missing on 41% of graded frames and nothing in the tracking says when play stops. We still wrote
rule variants (`out_of_bounds`, `set_piece_restart`, `hold_to_switch`) and a hand-specified HMM.
They stay runnable as baselines. On IDSSE the best rules beat the organisers' baseline (0.366
against 0.296 spellF1 on clean input) but stay far below the learned models, and the HMM doesn't
beat the baseline at all (Section 6.1).

For most of the project we expected to submit the trees alone. Our TCN runs had been on a GPU, and
CUDA probabilities differed from CPU ones by up to about 0.001 with TF32 on, which is a problem
when the repository has to regenerate the submitted file byte for byte. On the deadline evening we
retrained the TCN on a laptop CPU, which took 36 minutes, and then tested whether the CPU thread
count changes anything: across 1, 4 and 14 threads the probabilities moved by at most 1.3e-6 and
not one decoded frame changed, out of roughly 870,000 validation frames. That made the TCN safe to
ship on CPU, so the final system blends both models and its reproduction script hides any GPU.

### 2.1 Features

Both models read the same 93 features, all from tracking and computed per half, so no window,
interpolation or history crosses half-time. Ball gaps of up to 25 frames get linear interpolation
when the implied speed stays under 45 m/s.

- Ball: observed or interpolated, frames since the last and until the next observation, speed,
  5-frame net motion, turning angle and acceleration.
- Team proximity: each team's nearest-player distance, which team was nearest before and after a
  missing stretch, and vote shares over ±5, ±15, ±25 and ±75 frames.
- Player–ball contact: for radii of 0.75, 1.5 and 2.5 m, how long ago each team was last that close
  and how soon it will be next; also second-nearest distance, approach speed, velocity alignment
  and team distances at offsets of ±3, ±10 and ±25 frames. The radii sit around the 0.5–1 m
  possession zones of [1], and the models decide how much each one matters.
- Context: player speeds, how many players are nearly still, tracked fraction, ball distance to
  the pitch edge.

Neither model learns which club is which. The trees train on every row twice, once with the teams
swapped, and at inference both models average their probabilities over the two orientations.

### 2.2 The trees

The state classifier is scikit-learn's `HistGradientBoostingClassifier`, a histogram method in
the style of [5], with 220 iterations, 31 leaves, learning rate 0.07, L2 of 8, seed 20261111 and a
single thread; short possessions and frames near transitions carry extra weight. A second
classifier with 160 iterations predicts handovers, trained on frames within ±2 of a possession
change in the training matches. We export both as JSON trees and evaluate them in NumPy, which
reproduces scikit-learn's probabilities exactly. On validation this "joint" model beat a
possession-balanced variant by 0.0009, which is far too close to call either way, and its
handover classifier lifted it from 0.6448 to 0.6657.

### 2.3 The TCN

The network stacks non-causal dilated 1-D convolutions over the 93 features: 64 channels,
dilations 1, 2, 4, 8, 16 and 32, about 160,000 parameters and a reach of 126 frames on each side.
It has two heads, one for the state (H, A or D) and one for handovers. Training ran 35 epochs of
220 steps on batches of sixteen 256-frame windows from the same three training matches and the
same input profiles as the trees, with learning rate 0.001, weight decay 0.01, a handover-loss
weight of 0.2 and seed 20261114. Validation spellF1 climbed from 0.720 at epoch 5 to 0.741 at
epoch 30, then slipped to 0.740 at epoch 35, so we kept epoch 30.

### 2.4 Blending and decoding

Decoding runs log-space Viterbi over H, A and D [4], one half at a time, with initial and
transition probabilities counted on the three training matches and a handover potential on H↔A
moves. On the validation matches we chose how much of the state and handover probabilities
should come from the TCN (state share from {0, 0.25, 0.5, 0.75, 1}, handover share from {0, 0.5})
and the emission scale (0.3 or 1). An even 50/50 blend on both heads at scale 0.3 won with a mean
validation spellF1 of 0.7439. The trees alone had 0.6657, and handing the state entirely to the
TCN gave 0.7382, so each model covers some of the other's mistakes.

## 3. Assets used and dropped

| Asset | Final system | Notes |
|---|---|---|
| Tactical tracking (`_atd.xml`, `_atd.txt`) | Used | The only input at inference |
| Smart Tagging | Not used | Feeds the HMM and rule variants. IDSSE has no tagging, so a model trained there can't learn to use it |
| Homography | Not used | The tracking is already in pitch metres |
| Broadcast tracking | Not used | A synchronisation script exists; it isn't part of the final pipeline |
| Videos | Not used | Integrity check only (frame count, timestamps). Earlier we built a ball detector from this match's video and fitted an HMM with EM on this match's tracking; we removed both after the organisers' 3 November rules clarification |

## 4. Training data

| Dataset | Use | Labels | Amount |
|---|---|---|---|
| IDSSE [9]: DFL/Sportec official tracking of 7 Bundesliga matches, CC BY 4.0, files pinned by SHA-256 (PySport mirror, revision `a715a38`) | Training (3 matches), validation (2), test (2) | Provider channels `BallStatus` (dead gives D) and `BallPossession` (H or A), produced by DFL/Sportec | 425,961 / 289,155 / 287,528 labelled frames |
| Metrica Sports sample data, games 1 and 2 (pinned commit) | Partial evaluation only | Team of each PASS event at its start frame | 798 and 964 anchors |

We didn't create or correct a single label. Every training input goes through three profiles:
clean; synthetic ball dropout, a Markov process hiding 40% of frames in gaps averaging 25 frames;
and dropout plus 0.15 m coordinate jitter. For the trees we take every fifth frame and add the
team-swapped copy, which comes to 511,164 rows. None of it is from the provided match.

The Metrica sample repository has no explicit licence; its README only asks users to be
responsible with the data. We used it for a partial sanity check, never for training or for
choosing the final model, and the submission doesn't depend on it.

## 5. Possession rules implemented

The models learn the provider's mechanical definitions, D whenever the ball is out of play and
otherwise the team that owns it, so most of our line-drawing lives in the decoder.

Loose balls, deflections and aerial duels don't get a rule of their own. Switching teams costs
something in the transition matrix, and only sustained evidence or a confident handover score pays
that cost, so a touch or a ball passing close to a player doesn't flip the state. That's the
challenge's "possession stays with the last team that had it", and it agrees with [1], where a
touch counts only when the ball changes direction or speed.

We wrote no set-piece rules either: the D→H and D→A transitions, together with what happens right
after the restart, decide who restarts. For dead balls the models look at ball position and
motion, at how the ball data goes missing and at whether players have slowed down. A missing ball
on its own isn't treated as dead.

X appears only outside the published half boundaries. Inside a half we always commit, because
spellF1 forward-fills X with the previous label anyway.

## 6. Results (external data only)

Every number here comes from the organisers' scorer, as mean spellF1 over IoU 0.3–0.7. We had
already used IDSSE test matches J03WMX and J03WN1 in earlier iterations, so read this as a
re-evaluation rather than an untouched holdout.

| Model (mean of J03WMX and J03WN1) | Clean | Ball dropout | Dropout + jitter |
|---|---|---|---|
| Baseline (nearest player + majority vote) | 0.296 | 0.248 | 0.244 |
| Hand-specified HMM v2 | 0.278 | 0.233 | 0.213 |
| Joint trees | 0.720 | 0.604 | 0.594 |
| **TCN + trees (final)** | **0.802** | **0.688** | **0.675** |

The final blend beat the trees on all six match-and-profile rows, by 0.06 to 0.10. Per match it
scored 0.787 and 0.817 on clean input and never dropped below 0.654; the organisers put the spellF1
noise band at about 0.04 on a full match. It also predicts 435 and 440 possessions against the
provider's 511 and 500, closer than the trees' 417 and 437, though short possessions are still
the main loss (Section 7).

The baseline's numbers come with a caveat that cuts against it. It calls D only when the ball
disappears, and in IDSSE the ball never really disappears, so its dead-ball F1 is 0.00 on clean
input. The HMM goes wrong the other way, predicting about 200 possessions per match where the
provider has about 500.

Selection used only the validation matches J03WQQ and J03WR9 (0.7439 for the final blend, 0.6657
for the trees). Earlier iterations had seen the test matches, which is why we call the table a
re-evaluation. We have no accuracy figure for the provided match. The public 5-minute window
served one purpose: checking that the file was well formed and the frames aligned.

### 6.1 Everything else we tried

Dropped candidates matter as much as the winner, so on the deadline evening we rebuilt every
retained non-TCN variant from the current code and scored all 19 on the same test matches
(`config/idsse_benchmark_nontcn.toml`, 114 runs). Nothing here was used to choose the final model;
the earlier choices came from validation, and this table only checks them.

| Candidate | What it adds | Clean | Ball dropout | Dropout + jitter |
|---|---|---|---|---|
| `default` | Organisers' baseline: nearest player, majority vote | 0.296 | 0.248 | 0.244 |
| `exp_pos_control`, `exp_pos_x` | Rules: sustained close control; the second adds guarded abstention | 0.273 | 0.207 | 0.192 |
| `exp_hmm_v1`, `exp_hmm_v2` | Hand-specified HMM; v2 forgets outside-pitch evidence after 2 s | 0.278 | 0.233 | 0.213 |
| `exp_rules_v1` | Rules: control and dead-ball rules combined | 0.366 | 0.275 | 0.286 |
| `exp_bm_out`, `exp_bm_restart` | Rules: dead ball from confirmed outside-pitch evidence; restarts use tagging, absent in IDSSE | 0.365 | 0.291 | 0.301 |
| `exp_supervised_v1` | Trees, 26 features, clean training only | 0.488 | 0.436 | 0.242 |
| `exp_supervised_v2` | + dropout/jitter augmentation and net-motion features | 0.488 | 0.433 | 0.427 |
| `exp_supervised_v4` | + interaction features, larger trees | 0.632 | 0.564 | 0.555 |
| `exp_supervised_v3` | + interaction features (93 in total) | 0.635 | 0.567 | 0.552 |
| `exp_supervised_balanced` | v3 with possession-balanced sampling | 0.657 | 0.572 | 0.569 |
| `exp_supervised_refit` | Joint refitted on training and validation matches together | 0.713 | 0.600 | 0.593 |
| `exp_supervised_boundary_dense` | Handover classifier keeping every positive transition frame | 0.720 | 0.605 | 0.587 |
| `exp_supervised_joint` | v3 + handover classifier (the trees in Section 2.2) | 0.720 | 0.604 | 0.594 |
| `exp_supervised_boundary`, `exp_supervised_calibrated` | Same joint; calibration chose zero bias, so identical | 0.720 | 0.604 | 0.594 |
| `exp_supervised_ensemble` | Mixture with a random forest | 0.719 | 0.618 | 0.603 |
| **Final: TCN + trees** | Section 2.3 and 2.4 | **0.802** | **0.688** | **0.675** |

A few results stood out. The interaction features gave the biggest single jump (0.49 to 0.64),
bigger than any decoder change, and the forest ensemble's small edge on corrupted input (about
0.01) sat inside the noise band and had already failed the team's +0.01 adoption threshold on
validation. The rule variants did better on IDSSE than on Metrica pass-start anchors, where they
fell below the baseline.

## 7. Limitations and what we'd do next

The domain gap is the biggest risk. IDSSE is corrected in-stadium optical tracking, whereas the
provided match comes from automatic tactical-camera tracking that loses the ball often; our
augmentation imitates dropout and jitter but not identity switches or systematic ball errors.
[1] says data like ours needs larger possession zones and per-provider re-tuning, and the rules
allow that only on external data, which is where we did it.

IDSSE also hides a shortcut: during nearly every dead frame the ball's coordinates freeze in place.
External validation therefore says little about dead-ball detection when the ball has simply
vanished, which is the common case in the provided match.

Short possessions under two seconds still get missed or merged.

On Metrica pass-start anchors (our 13 November run) the learned models came out slightly behind
the plain baseline, at about 92–93% against 94–96%, even though they're far ahead on IDSSE.
Anchors cover passes only and [1] reports 10–15% disagreement between event annotators, so we
don't treat them as decisive; the gap still warns about cross-provider transfer.

Reproducing the TCN exactly depends on running it on a CPU. We tested thread counts on one laptop,
not different CPU vendors, so we'd like to confirm the output on an AMD and an Intel machine.

With more time we'd add formation-based set-piece triggers as dead-ball evidence and
Savitzky–Golay ball smoothing [1], try ball-trajectory inference for missing-ball stretches [3]
and look for labelled automatic-tracking data from other providers.

## 8. Reproduction

The repository ships both trained models (`models/`), so regenerating the submission takes one
command after installation:

```
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-team.txt
python -m pip install -r requirements-temporal.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
# data bundle: unpack tactical/ into data/ (or set MTAH_DATA_DIR)
python tools/predict_temporal.py                    # -> outputs/Goalorithm.csv, CPU only
```

Rebuilding the models from the pinned IDSSE files first:

```
python tools/prepare_idsse.py                       # downloads/verifies the pinned IDSSE files
python tools/train_supervised.py --config config/exp_supervised_v3.toml --out-dir outputs/supervised_v3
python tools/train_boundaries.py
python tools/train_supervised.py --config config/exp_supervised_balanced.toml --out-dir outputs/supervised_balanced
python tools/select_joint.py                        # trees -> outputs/supervised_joint/
python tools/train_temporal.py                      # TCN  -> outputs/temporal/
```

Then copy `outputs/supervised_joint/model.json` and the five files of `outputs/temporal/` into
`models/` as listed in `models/README.md`. Python 3.11, with every dependency pinned using `==`.
Seeds are fixed, the trees train on one thread, and the TCN loader checks the hashes of its weights,
its teacher and the source files it was trained with. Two consecutive runs produced the same
`Goalorithm.csv`, SHA-256 `dd610799b8ad715d7371c1dfbc783ba0a484eeb9784696c79f93e14721822af3`.

## 9. Rules compliance

- No human labels or human-produced data about the provided match, in any form.
- Nothing fitted, tuned, calibrated or selected on its ground truth, its video or its tracking.
- We never checked predictions against the video, and the repository no longer contains the
  review and label-suggestion tools.
- No competition data went to any AI service; everything ran locally.
- External data: IDSSE (CC BY 4.0, cited below) for training and selection; Metrica's public
  sample data (no explicit licence) for a partial check only.

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
