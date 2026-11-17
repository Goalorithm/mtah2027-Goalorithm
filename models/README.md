# Committed model checkpoints

`supervised_joint/model.json` is the joint model (gradient-boosted state and handover classifiers
plus Viterbi decoder parameters). `config/final.toml` loads it on its own, and the final TCN
(second section below) uses it as its teacher.

- **Trained on:** IDSSE external matches only (DFL/Sportec tracking, CC BY 4.0; Bassek, Rein,
  Weber and Memmert, *Scientific Data* 12:195, 2025, doi:10.1038/s41597-025-04505-y). Training
  used J03WOH, J03WOY and J03WPY; selection used J03WQQ and J03WR9.
- **Never touched:** BAR-ATH tracking, video, ground truth or any human-produced data about it.
  The `provenance` block inside the file records this, together with source, config and data hashes.
- **Selected:** v3 body model, emission scale 0.3, handover weight 1.0 (mean validation spellF1
  0.6657 over two matches and three input profiles).
- **SHA-256:** `724475e5d1b8ddc4c05d2593972c49ae671892cbbca7bc706ee2dffa39393e85`

## Rebuilding it

```
python tools/prepare_idsse.py
python tools/train_supervised.py --config config/exp_supervised_v3.toml --out-dir outputs/supervised_v3
python tools/train_boundaries.py
python tools/train_supervised.py --config config/exp_supervised_balanced.toml --out-dir outputs/supervised_balanced
python tools/select_joint.py          # writes outputs/supervised_joint/model.json
```

Training is seeded and single-threaded. Our rebuild on 17 November 2026 reproduced the earlier
selection exactly, with the same choice and the same validation score to four decimals. The
`provenance` block also records hashes of the source files, so the bytes of `model.json` match only
when the source matches too; predictions depend only on the trees and decoder parameters.

# TCN checkpoint (final submission model)

`temporal/` holds the temporal convolutional network that `tools/predict_temporal.py` runs on top
of the joint model above, which acts as its teacher. These are the files `tools/evaluate_temporal.py`
expects, and it verifies their hashes and the hashes of the source files they were trained with.

| File | SHA-256 |
|---|---|
| `weights.pt` | `1292862f64284a70eb6a6ee89a670b966f84c6d1e48fce2ea22e1e3598f29086` |
| `model.json` | `8fddbbecc058e965a3baf43711d2dfa12627cafb4d7b1f329c18f41f6877a8be` |
| `parent.json` (same bytes as `supervised_joint/model.json`) | `724475e5d1b8ddc4c05d2593972c49ae671892cbbca7bc706ee2dffa39393e85` |
| `training_report.json` | `34baa052d82e704a8a7154ed8ce115c236842275f71b097f691682733ddcccb8` |
| `test_freeze.json` | `2b9ec91ed624d9cc7fc3e3d94fc4ed9d2aee905c8f8785d5d5c52c3aaed0cdd2` |

- **Trained on:** the same three IDSSE training matches, with `config/temporal.toml` (64 channels,
  dilations 1 to 32, 35 epochs, seed 20261114), on a laptop CPU in about 36 minutes.
- **Selected on:** the two IDSSE validation matches. Epoch 30 scored best; a 50/50 blend of TCN and
  tree probabilities for both the state and the handover heads, at emission scale 0.3, averaged
  0.7439 against 0.6657 for the trees alone.
- **Source pinning:** `model.json` records hashes of `src/mtah/*.py`, `config/temporal.toml`,
  `config/exp_supervised_joint.toml` and the training helpers. If any of them changes, loading
  fails on purpose and the TCN has to be retrained.

Rebuilding it, after the joint model above:

```
python -m pip install -r requirements-temporal.txt --index-url https://download.pytorch.org/whl/cpu
python tools/train_temporal.py        # writes outputs/temporal/
```
