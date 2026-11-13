# manas_finetune_result: evaluation of manas2002/aya-indicguard-hardened

Evaluated 2026-11-13 with the same harness, splits, judge and metrics as the steering results in `article.md`.

## What was tested
- Model: `manas2002/aya-indicguard-hardened`, commit `63fff8eb7a6c96cbc2ddd9c86c1be2c04a2b6f80`.
- Per `lexsi_provenance.json`: full fine-tune of `CohereLabs/tiny-aya-fire` with SafeTune `harden.SafeGrad` (v0.1.18), 1 epoch, batch 2, lr 1e-4. There is no model card, and the training data is not stated.
- Eval: TEST (799 items, 467 non-English harmful) and UNSEEN (117 items, 20 Indic languages never used for selection). Greedy decoding, 320 new tokens. Judge: IndicGuard. Paired exact McNemar against base on the same items.
- Sanity check: 15 of the first 32 TEST outputs differ from base, so the fine-tuned weights were loaded.

## Headline

| Model | TEST ASR_nonEN | Harmless refusal | 4-form consistency | Fixed / broken vs base | McNemar p | UNSEEN unsafe (pooled) | UNSEEN p | MMLU-500 |
|---|---|---|---|---|---|---|---|---|
| base | 26.1% | 3.9% | 0.68 | – | – | 28.2% | – | 0.586 |
| aya-indicguard-hardened | 28.1% | 4.2% | 0.64 | 14 / 23 | 0.19 | 26.5% | 0.77 | 0.586 |
| steering d_en α=0.05 (ours) | 16.3% | 8.7% | 0.82 | 56 / 10 | 7e-9 | 16.2% | 0.007 | 0.592 |

ASR_nonEN = share of non-English harmful prompts that get an unsafe response (lower is better).

## Per form, TEST, unsafe rate on harmful prompts

| Form / attack | n | base | hardened | steer d_en 0.05 |
|---|---|---|---|---|
| English (IndicSafe) | 22 | 0.0% | 0.0% | 0.0% |
| Telugu script (IndicSafe) | 24 | 12.5% | 12.5% | 4.2% |
| Romanized Telugu (IndicSafe) | 24 | 8.3% | 12.5% | 8.3% |
| Telugu–English mix (IndicSafe) | 24 | 16.7% | 25.0% | 8.3% |
| Telugu script (IndicGuard) | 292 | 24.0% | 26.7% | 14.7% |
| CSRT code-switch | 30 | 6.7% | 3.3% | 3.3% |
| Telugu single-turn attacks | 25 | 44.0% | 44.0% | 24.0% |
| Telugu multi-turn attacks | 48 | 62.5% | 60.4% | 43.8% |

UNSEEN: single-turn 24.4% → 24.4%, multi-turn 35.9% → 30.8% (7 fixed, 5 broken, p = 0.77).

## Reading
- The hardened model is not measurably safer than base on any split. On TEST it is slightly worse (23 items became unsafe, 14 became safe), but p = 0.19, so this is no detectable change, not a proven regression.
- It costs nothing either: over-refusal +0.3 points, and MMLU unchanged. That fits a light 1-epoch fine-tune that barely moved the model.
- Even on the IndicGuard rows, where training-data overlap would have helped it most, it does not improve (24.0% → 26.7%). So data overlap and judge leniency, the two risks we flagged beforehand, did not inflate its score.
- Inference-time steering at layers 15–18 beats it on every non-English form and on UNSEEN languages, at a cost of about +5 points of over-refusal.

## Limits
- The IndicGuard judge is not yet validated against human labels.
- Per-form cells of 22–30 items have wide CIs. The pooled McNemar tests are the claims that matter.
- We do not know the hardened model's training data. A different dataset, more epochs or a higher learning rate for SafeGrad could give a different result. This evaluates this checkpoint only.
