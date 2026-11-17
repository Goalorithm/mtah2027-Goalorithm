# Goalorithm — Technical Document

More than a Hack 2027, Phase 1: Ball Possession. Team Goalorithm. 18 November 2026.

**Submitted file:** `Goalorithm.csv`, 199,319 rows, SHA-256 `05f57e08a7517defb3617e417cc914e57d9302f0e6c7191c04c4ce3c88b451c2`.

**In one paragraph.** The competition tracking is automatic: the organisers report the ball
missing on 41% of graded frames and about 20.5 tracked objects per frame. Models fitted on the
corrected tracking that public labelled games provide learn that provider's conventions (a frozen
ball means a stoppage, the ball is always there) and fail on automatic tracking. We therefore
take labelled games from other fixtures, re-measure them so that they behave like automatic
tracking (lost ball, smoothed ball track, lost players, re-synthesised stoppages), fit a temporal
convolutional network on them and decode its output into possessions with a Viterbi decoder.
Every model weight, every decoder setting and every selection is fitted on external matches only.
The model reads only the provided automatic tracking of BAR-ATH; it uses no video, no broadcast
recording and no Smart Tagging.

## 1. Approach

### 1.1 Re-measuring external games like automatic tracking (`tools/atd_sim.py`)

Each labelled external game is turned into several randomly measured copies before fitting:

- **Ball losses.** During live play the ball is hidden in stretches drawn from a renewal process
  that ignores the labels. At stoppages the ball is re-synthesised (it rolls on, is handled by
  whoever fetches it, is carried or kicked back to the restart spot, waits) and is then missing,
  visible or partly visible.
- **Smoothing.** The ball track passes through a zero-phase low-pass filter.
- **Players.** Players are dropped by tracklet and in shared outages.
- **Ranges, not fitted values.** Every parameter is drawn per copy from a range in
  `config/atd.toml`, so the network sees many tracking behaviours rather than one guess.
  Validation and test copies use the middle of each range.

The ranges were set from the organisers' published figures (ball missing on 41% of graded
frames, about 20.5 tracked objects) and from label-free descriptive statistics of the provided
tracking (lengths of ball gaps, smoothness of the ball track, how a tracked ball moves at
stoppages). No label, no video and no ground truth of the provided game was used, and no model
weight or decoder setting was fitted on the provided game.

### 1.2 Label sources and the possession definition

CHALLENGE.md defines possession by control: a touch without control changes nothing, and "in
phases where the ball is contested or loose, possession stays with the last team that had it".
SoccerTrack v2's provider marks team control and contested phases separately, so its contested
frames are given to the last team that had the ball. IDSSE and Metrica references are used as
the providers define them.

### 1.3 Model and decoder (`tools/atd_features.py`, `tools/train_atd.py`, `tools/ensemble_atd.py`)

- **Inputs:** 118 features per frame from the automatic ball and the players: distances and
  approach of each team to the ball over several windows, ball speed, how long the ball has been
  missing or resting, team shape and movement. Attack direction is read per half, so the features
  are symmetric in H and A.
- **Network:** a non-causal temporal convolutional network, 96 channels, dilations 1 to 128
  (about ±20 s of context), with one head for the state (H, A, D) and one for change events.
- **Fitting data:** 3 IDSSE games and 2 Metrica sample games (65% of samples) and 6 SoccerTrack v2
  matches (35%), each as several randomly measured copies.
- **Decoder:** Viterbi over the three states, transitions from the IDSSE and Metrica fitting
  references; emission scale and event weight chosen on validation games of the three sources,
  each source weighted equally.
- **Ensemble and refit:** five selection seeds (20261120-20261124), each trained for up to
  32 epochs, fix the refit length at 24 epochs and the decoder at emission scale 0.2,
  event weight 0.75. Five seeds are then refitted on fitting data plus the IDSSE/Metrica
  validation games, with both fixed; their probabilities are averaged before decoding.
  The SoccerTrack decoder-validation game stays outside fitting.

## 2. Assets used and dropped

| Asset | Used | How, or why not |
| --- | --- | --- |
| Tactical automatic tracking | Yes | Player and ball positions, speeds, team per track: the only input |
| Tactical video | No | Not used by the submitted model |
| Tactical homography | No | Tracking is already in pitch metres |
| Broadcast video, tracking and homography | No | Not used by the submitted model |
| Smart Tagging | No | Not used by the submitted model |
| Public 5-minute ground truth | No for this revision | Not read during training, selection, testing or CSV export |

## 3. Training data

Every labelled dataset is from other matches and openly available. None of it comes from the
provided game.

| Dataset | Labels and who produced them | Amount | Role | Licence / access |
| --- | --- | --- | --- | --- |
| IDSSE DFL/Sportec tracking (Bassek et al. 2025, doi:10.1038/s41597-025-04505-y) | Provider's ball status and possession channels (DFL/Sportec) | 7 games: 3 fitting, 2 validation, 2 test | Fit, select, test | CC BY 4.0 |
| Metrica Sports sample games 1-3 | Derived mechanically from the provider's events: dead ball where the corrected ball is blank, team of the latest owning event | 3 games: 2 fitting, 1 validation | Fit, select | Public repository, commit `e706dd5` |
| SoccerTrack v2 (Scott et al., arXiv:2508.01802) | Provider's per-frame possession state | 8 matches: 6 fitting, 1 decoder validation, 1 test | Fit, select, test | CC BY 4.0, Hugging Face (free account, terms accepted once) |

No human label of the provided game was created or used.

## 4. Possession rules implemented

- **Dead ball (D)** is the ball out of play, from the moment play stops until the restart is
  taken, as the providers mark it.
- **Team (H or A)** during live play is the team that last controlled the ball. Contested and
  loose balls stay with the last team that had the ball.
- **Touches without control** do not change the label unless the network sees sustained evidence
  for the other team.
- **Restarts.** The possession of the team taking the restart begins when the ball is back in play.
- **A missing ball is not a dead ball.** The model decides from player movement, how long the
  ball has been missing and the surrounding context.
- **X** is written only outside the two graded halves.

## 5. Validation

All figures use the organisers' scorer on games no fitting step used, measured like automatic
tracking (middle of every range). Higher is better.

| Check | Prior three-seed spellF1 | Five-seed spellF1 | Five-seed macroF1 |
| --- | ---: | ---: | ---: |
| Balanced external validation, selection ensembles | 0.7667 | 0.7697 | |
| IDSSE test games J03WMX, J03WN1 (2 games, 4 copies) | 0.7143 | 0.7111 | 0.9322 |
| SoccerTrack v2 test match 117092, contested frames to last owner | 0.7561 | 0.7653 | 0.9262 |

Selection validation by source (IDSSE / Metrica / SoccerTrack): previously
0.7435 / 0.7367 / 0.8200; now 0.7416 / 0.7315 /
0.8361. The balanced gain is 0.0030,
mainly from SoccerTrack; IDSSE and Metrica validation regress slightly. This small observed
gain is not a statistical-significance claim. The predeclared adoption threshold was >0.767,
using equally weighted external sources; neither test results nor BAR-ATH labels chose it.
The existing external test games have been scored before; they are not a new untouched holdout.

SoccerTrack match 117093 is decoder validation and is excluded from the test row above.
The public BAR-ATH ground truth was not read or scored for this revision. The BAR-ATH score
remains unknown. Passing code checks or repeated CPU export does not establish its accuracy.

## 6. Limitations

- **The score on BAR-ATH is not known to us.** All figures are on other games re-measured to
  resemble automatic tracking; behaviour the re-measurement does not capture is not covered.
- **Short possessions and duels.** About one possession in four is missed or wrongly bounded on
  IDSSE labels, mostly those under two seconds. When players of both teams are within reach of
  the ball, positions alone do not say who controlled it.
- **Label conventions differ by provider.** IDSSE records possession more finely than the
  competition's definition; the decoder is chosen on three sources weighted equally.
- **With more time:** a ball and player-contact model trained on footage of other matches, to
  decide duels; more external labelled games.

## 7. Reproduction

Python 3.11, dependencies pinned in `requirements-team.txt`, `requirements-temporal.txt` and
`requirements-validation.txt`. Unpack the tactical tracking of the bundle into `data/`.

From the released weights (no fitting):

```bash
pip install -r requirements-team.txt
pip install -r requirements-temporal.txt --index-url https://download.pytorch.org/whl/cpu
pip install -e .
python tools/restore_release.py
python tools/evaluate_atd.py --model-dir outputs/atds5_refit_ensemble --mode competition --name Goalorithm --no-public-check
```

This writes `outputs/Goalorithm.csv` and prints its SHA-256. The file is always computed on CPU
and the evaluator runs the prediction twice and refuses to write a file if the runs differ.

Refitting from the external data: `run_all_tech.cmd` (IDSSE and Metrica downloads), then
`run_atd.cmd`.
