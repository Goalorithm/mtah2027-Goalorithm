# Goalorithm ? More than a Hack 2027, Phase 1: Ball Possession

Team Goalorithm's entry to [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027):
label each frame H (Barcelona), A (Athletic), D (dead ball), or X (abstain).

Repository: [Goalorithm/mtah2027-Goalorithm](https://github.com/Goalorithm/mtah2027-Goalorithm).
The official source snapshot is pinned in [.github/upstream-source.json](.github/upstream-source.json).

## Current branch scope

The retained methods use tracking and Smart Tagging at inference. The baseline, rule methods,
and configured HMM remain runnable. Supervised trees and temporal models are fitted on external
matches through the pinned dataset adapters. Rebuild their checkpoints after source changes;
historical reports do not validate the current code or establish the competition score.

| Role | Retained implementation |
| --- | --- |
| Tech 1 | Configured HMM + Viterbi, bounded ball interpolation, external-model training and sequence decoding |
| Tech 2 | Confirmed outside-pitch evidence and tracking-supported Smart Tagging restarts |
| Tech 3 | Sustained team control, guarded abstention, interaction and handover features |

See [docs/HMM_V2.md](docs/HMM_V2.md),
[docs/SUPERVISED_IMPROVEMENT.md](docs/SUPERVISED_IMPROVEMENT.md),
[docs/INTERACTION_IMPROVEMENT.md](docs/INTERACTION_IMPROVEMENT.md),
[docs/TEMPORAL_IMPROVEMENT.md](docs/TEMPORAL_IMPROVEMENT.md), and
[docs/DOMAIN_IMPROVEMENT.md](docs/DOMAIN_IMPROVEMENT.md) for the retained research methods.
`run_all_tech.cmd`, followed by the optional temporal/domain workflows, rebuilds external models.

The video-ball pipeline, its dependent models and training of an HMM on BAR-ATH have been
excluded from this branch's current file tree. The publication guard reads
[.github/excluded-paths.txt](.github/excluded-paths.txt); run
`python tools/check_repository_content.py` before committing. Their local archives, generated
CSVs and submission drafts stay outside Git. This repository has a fresh history; excluded research from the prior team repository was never imported.

Run `python tools/check_release.py` for code checks and see
[docs/RELEASE_CHECK.md](docs/RELEASE_CHECK.md) for optional external-artifact verification.

## Reproduce the submission

Python **3.11**. The match data is gated and not in this repository (see [DATA.md](DATA.md));
unpack the `tactical/` folder of the bundle into `data/`, or point `MTAH_DATA_DIR` at it.

```bash
python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-team.txt
python -m pip install -e .
python -m mtah.predict --config config/final.toml      # -> outputs/Goalorithm.csv
```

`config/final.toml` loads the committed joint checkpoint in [models/](models/README.md), which
was trained only on external IDSSE matches; no download or training is needed. The command
validates the file with the organisers' checks and prints its sha256; the submitted file is
`2d063850f6b6be84780333f68000205a4677363ba85d30495b8298f445c25ab2`. To rebuild the checkpoint from the pinned IDSSE data first, follow
[models/README.md](models/README.md). The approach is described in the technical document,
[docs/Goalorithm.md](docs/Goalorithm.md).

On Windows machines where policy blocks `pip.exe`, keep the `python -m pip` form shown above.
`python -m mtah.predict` without `--config` still reproduces the organisers' `baseline_smooth50`.
CI checks code and baseline scoring on Windows and Linux; model accuracy requires separate evaluation.

Only three files of the bundle are read: `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt`
and `BAR-ATH_tactical_smart_tagging.xml` (the final model ignores the tagging; the HMM and rule
variants use it). The first run parses the tracking into `cache/`
(about a minute); later runs take seconds.

## Repository layout

```
src/mtah/            the pipeline
  load.py            tracking + Smart Tagging, cached
  features.py        per-frame features (ball, nearest players, speeds)
  dead_ball.py       which frames the ball is out of play
  possession.py      which team has the ball while it is alive
  predict.py         one command: data in, validated submission out
config/default.toml  every tunable number, in one place
tools/               setup check, timeline and frame viewers, full-match report, label converter
tests/               run with pytest; no match data needed
labels/              diagnostic window plan; same-game human annotation workflow retired
docs/                technical document and team guides
evaluation/ starter/ games.py   the organisers' code, unchanged (MIT licence)
```

## Team guides (Bahasa Indonesia)

- [docs/SETUP.md](docs/SETUP.md) — menyiapkan laptop dari nol
- [CONTRIBUTING.md](CONTRIBUTING.md) — alur kerja branch, PR, review
- [docs/RULE_UPDATE.md](docs/RULE_UPDATE.md) — aturan terbaru dan alur pseudo-label lokal
- [docs/ATURAN_POSSESSION.md](docs/ATURAN_POSSESSION.md) — arsip buku aturan possession tim

The organisers' own documents are kept as shipped: [CHALLENGE.md](CHALLENGE.md),
[DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md),
[RULES.md](RULES.md), [FAQ.md](FAQ.md).

## Licence

Code: MIT, building on the organisers' repository (see [LICENSE](LICENSE)). The match data is
licensed separately for this competition only and is not, and must never be, in this repository.
