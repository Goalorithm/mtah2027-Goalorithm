# Committed model checkpoint

`supervised_joint/model.json` is the final joint model (gradient-boosted state and handover
classifiers plus Viterbi decoder parameters) that `config/final.toml` loads.

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
