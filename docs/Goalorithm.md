# Goalorithm: technical document (More than a Hack 2027, Phase 1)

Team Goalorithm · Repository: <https://github.com/Goalorithm/mtah2027-Goalorithm> (branch `main`)

## 1. Summary

Every frame of the FC Barcelona (H) vs Athletic Club (A) tactical video gets one label, H, A or
D, and we write X only outside the two halves. The final system reads nothing but the automatic
tracking. It turns each frame into geometry and motion features, scores them with two
gradient-boosted classifiers fitted on public Bundesliga matches (IDSSE, CC BY 4.0), then decodes
each half with Viterbi so that a possession has to earn both its start and its end. One classifier
estimates the state (H, A or D); the other estimates whether possession is changing hands.

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
They stay runnable as baselines. The HMM and the baseline trail the learned model on IDSSE
(Section 6), and we only checked the rule variants on partial Metrica anchors, where they did
worse than the baseline.

We trained temporal convolutional networks [6, 7] on the same external matches too, and on IDSSE
validation they beat the trees. They need a GPU, though, and in our runs CUDA and CPU probabilities
differed by up to about 0.001 with TF32 enabled; since the repository has to regenerate the
submitted file byte for byte, that settled it for this submission.

### 2.1 Features

All features come from tracking and are computed per half, so no window, interpolation or history
crosses half-time. Ball gaps of up to 25 frames get linear interpolation when the implied speed
stays under 45 m/s.

- Ball: observed or interpolated, frames since the last and until the next observation, speed,
  5-frame net motion, turning angle and acceleration.
- Team proximity: each team's nearest-player distance, which team was nearest before and after a
  missing stretch, and vote shares over ±5, ±15, ±25 and ±75 frames.
- Player–ball contact: for radii of 0.75, 1.5 and 2.5 m, how long ago each team was last that close
  and how soon it will be next; also second-nearest distance, approach speed, velocity alignment
  and team distances at offsets of ±3, ±10 and ±25 frames. The radii sit around the 0.5–1 m
  possession zones of [1], and the model decides how much each one matters.
- Context: player speeds, how many players are nearly still, tracked fraction, ball distance to
  the pitch edge.

The model never learns which club is which. Each training row also appears with the teams
swapped, and at inference we average the probabilities from both orientations.

### 2.2 Models and decoder

The state classifier is scikit-learn's `HistGradientBoostingClassifier`, a histogram method in
the style of [5], with 220 iterations, 31 leaves, learning rate 0.07, L2 of 8, seed 20261111 and a
single thread; short possessions and frames near transitions carry extra weight. It uses 93
features. Selection took the highest validation score, which was this v3 body; a
possession-balanced variant finished 0.0009 behind, far too close to call either way. A second
classifier with 160 iterations
predicts handovers, trained on frames within ±2 of a possession change in the training matches.
We export both as JSON trees and evaluate them in NumPy, which reproduces scikit-learn's
probabilities exactly.

Decoding runs log-space Viterbi over H, A and D [4], one half at a time. We count the initial and
transition probabilities on the three training matches, and the handover classifier adds a
potential to H↔A moves. That leaves two numbers to choose, the emission scale from
{0.3, 1, 3, 10} and the handover weight from {0, 0.5, 1}; mean spellF1 on two separate validation
matches, averaged over three input profiles, picked both. The winner was scale 0.3 with handover
weight 1.0, at a mean validation spellF1 of 0.6657. Without the handover potential the same body
scored 0.6448, so the handover classifier earns its place.

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

We didn't create or correct a single label. Training takes every fifth frame, runs it through
three input profiles (clean; synthetic ball dropout, a Markov process hiding 40% of frames in gaps
averaging 25 frames; dropout plus 0.15 m coordinate jitter) and adds the team-swapped copy, which
comes to 511,164 rows. None of it is from the provided match.

The Metrica sample repository has no explicit licence; its README only asks users to be
responsible with the data. We used it for a partial sanity check, never for training or for
choosing the final model, and the submission doesn't depend on it.

## 5. Possession rules implemented

The model learns the provider's mechanical definitions, D whenever the ball is out of play and
otherwise the team that owns it, so most of our line-drawing lives in the decoder.

Loose balls, deflections and aerial duels don't get a rule of their own. Switching teams costs
something in the transition matrix, and only sustained evidence or a confident handover score pays
that cost, so a touch or a ball passing close to a player doesn't flip the state. That's the
challenge's "possession stays with the last team that had it", and it agrees with [1], where a
touch counts only when the ball changes direction or speed.

We wrote no set-piece rules either: the D→H and D→A transitions, together with what happens right
after the restart, decide who restarts. For dead balls the model looks at ball position and
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
| Joint (final) | **0.720** | **0.604** | **0.594** |

Per match, the joint model scored 0.723 and 0.717 on clean input and never dropped below 0.586 on
the corrupted profiles; the other two models never got above 0.31. The organisers put the noise
band for spellF1 at about 0.04 on a full match, and the gap here is roughly ten times that.

The baseline's numbers come with a caveat that cuts against it. It calls D only when the ball
disappears, and in IDSSE the ball never really disappears, so its dead-ball F1 is 0.00 on clean
input. The HMM goes wrong the other way: it predicts about 200 possessions per match where the
provider has about 500, merging everything. Even the joint model predicts 417 and 437 possessions
against 511 and 500, which is the short-possession weakness from Section 7.

This round's selection used only the validation matches J03WQQ and J03WR9, where the chosen
configuration averaged 0.6657 over the three profiles. Earlier iterations had seen the test
matches, which is why we call the table a re-evaluation.

We have no accuracy figure for the provided match. The public 5-minute window served one purpose:
checking that the file was well formed and the frames aligned.

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

With more time we'd add formation-based set-piece triggers as dead-ball evidence and
Savitzky–Golay ball smoothing [1], try ball-trajectory inference for missing-ball stretches [3],
make the TCN deterministic on CPU and look for labelled automatic-tracking data from other
providers.

## 8. Reproduction

The repository ships the trained checkpoint (`models/supervised_joint/model.json`, SHA-256
`724475e5…3e85`), so regenerating the submission takes one command after installation:

```
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-team.txt && python -m pip install -e .
# data bundle: unpack tactical/ into data/ (or set MTAH_DATA_DIR)
python -m mtah.predict --config config/final.toml   # -> outputs/Goalorithm.csv
```

Rebuilding the checkpoint from the pinned IDSSE files first:

```
python tools/prepare_idsse.py                       # downloads/verifies the pinned IDSSE files
python tools/train_supervised.py --config config/exp_supervised_v3.toml --out-dir outputs/supervised_v3
python tools/train_boundaries.py
python tools/train_supervised.py --config config/exp_supervised_balanced.toml --out-dir outputs/supervised_balanced
python tools/select_joint.py
python -m mtah.predict --config config/exp_supervised_joint.toml        # -> outputs/Goalorithm.csv
python evaluation/validate_submission.py outputs/Goalorithm.csv
```

Python 3.11, with every dependency pinned using `==`. Seeds are fixed, training runs on one thread
and inference uses NumPy only. Two consecutive runs produced the same `Goalorithm.csv`, SHA-256
`2d063850f6b6be84780333f68000205a4677363ba85d30495b8298f445c25ab2`.

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
