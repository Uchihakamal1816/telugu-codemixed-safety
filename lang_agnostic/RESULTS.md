# RESULTS (overnight 2026-11-11, DEV split only; TEST/UNSEEN untouched)

DEV = 400 items: IndicSafe (16 harmful + 4 harmless per form × 4 forms), IndicGuard Telugu (145 harmful / 148 harmless), CSRT (20), IndicSafetyBench Telugu single-turn (7).
ASR_nonEN = unsafe rate on 220 non-English harmful items (Telugu script, romanized, mixed, CSRT, IndicGuard, Bench).
Judge = IndicGuard, not yet validated against human labels (judge_audit.csv still to build/label).
Refusal detector = phrase list (refusal_phrases.txt, unchecked) — known to undercount (see LITERATURE.md, Kachireddy 2026).

## Base
- ASR_nonEN 0.145; harmless refusal 0.030; MMLU-500 0.586; IndicSafe 4-form consistency 0.94.

## Diagnosis (pilot-sized: 16+4 per form on dev)
- H1 (English-only harmfulness probe, AUROC ≥ 0.85 at L ≤ 14 in every form): FAIL, narrowly.
  Telugu script 0.77 (L14), romanized 0.91 (L8), mixed 1.00 (L6). n is tiny; the Telugu-script result is the weak spot.
- Per-form harmful–harmless direction vs English (cosine, norm), train acts:

| layer | english | telugu_script | romanized | mixed |
|---|---|---|---|---|
| 9 | 1.00, 6.8 | 0.34, 1.9 | 0.25, 1.5 | 0.65, 5.3 |
| 12 | 1.00, 10.7 | 0.46, 3.4 | 0.17, 2.0 | 0.67, 7.7 |
| 15 | 1.00, 19.1 | 0.71, 6.7 | 0.17, 2.2 | 0.68, 10.0 |
| 18 | 1.00, 26.7 | 0.69, 10.5 | 0.12, 2.6 | 0.66, 14.4 |

  Reading: in romanized Telugu the harmful/harmless separation is both tiny (norm 2.6 vs 26.7) and nearly orthogonal to English.
  Telugu script and mixed separate in roughly the English direction but at 40–55% of the strength. This is evidence for
  "weak separation" (Wang et al. 2025) in script/mixed forms and for "non-transfer" (Zhang 2026) in romanized form.
  Caveat: romanized train rows are mostly extract_roman (IndicGuard-style prompts), not IndicSafe twins — possible confound.

## Unconditional steering, layers 15–18, vector = alpha × ||d_en[L]|| × unit(dir)

| method | ASR_nonEN | harmless refusal | MMLU | flips safe←unsafe / unsafe←safe | McNemar p |
|---|---|---|---|---|---|
| base | 0.145 | 0.030 | 0.586 | – | – |
| d_en 0.05 | 0.086 | 0.085 | 0.592 | 21 / 7 | 0.013 |
| d_en 0.10 | 0.055 | 0.140 | 0.576 | 26 / 5 | 0.0002 |
| d_en 0.15 | 0.036 | 0.323 | 0.516 | 30 / 5 | 2e-5 |
| d_multi 0.05 | 0.105 | 0.098 | 0.592 | 17 / 7 | 0.064 |
| d_multi 0.10 | 0.055 | 0.165 | 0.562 | 26 / 5 | 0.0002 |
| d_multi 0.15 | 0.036 | 0.329 | 0.498 | 28 / 3 | 5e-6 |
| d_multi_nolang 0.05 | 0.082 | 0.091 | 0.590 | | |
| d_multi_nolang 0.10 | 0.045 | 0.220 | 0.564 | | |
| d_multi_nolang 0.15 | 0.032 | 0.335 | 0.518 | | |

- No unconditional setting passes the §0.5 rule (over-refusal ≤ +3 pts). Smallest cost: d_en 0.05 (+5.5 pts).
- The safety/over-refusal trade-off is nearly identical across the three directions: at matched norm the direction choice matters little;
  the size of the push dominates. "Language-agnostic" directions did not beat the English one on dev.
- Earlier attempt at alpha 0.25 (d_en): ASR 0 but harmless refusal 0.61 — far too strong.

## CAST (gated)
- Fit (prompts cut to 1500 chars, candidate layers 8–18): condition at layer 18, threshold 0.11, F1 0.77.
- Gate firing rate on dev: IndicGuard harmful 0.77 / harmless 0.24; IndicSafe romanized harmful 0.88 / harmless 3 of 4, mixed 0.63 / 3 of 4, Telugu script 0.75 / 0 of 4; Bench 0 of 7. The gate leaks onto harmless prompts.
- Steering vector: d_multi, layers 15–18, same norm-matching as above.

| method | ASR_nonEN | harmless refusal | MMLU | flips | McNemar p |
|---|---|---|---|---|---|
| cast d_multi 0.1 | 0.077 | 0.104 | 0.586 | 22 / 6 | 0.004 |
| cast d_multi 0.2 | 0.059 | 0.152 | 0.584 | 25 / 5 | 0.0003 |
| cast d_multi 0.3 | 0.068 | 0.220 | 0.588 | 25 / 7 | 0.002 |

- CAST keeps MMLU at base level at every alpha; unconditional steering lost 3–9 pts at matched strength.
- Over-refusal is still above the +3 pt limit (+7 pts at alpha 0.1), because the gate fires on 24% of harmless prompts.
- Next lever is the gate, not the vector: fit it on more harmless non-English data and pick the threshold on over-refusal.
