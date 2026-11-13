# Tentative article: Safety that speaks Telugu — measuring and closing the code-mixed safety gap in Tiny Aya Fire

Lexsi Labs hackathon, problem statement: multilingual and code-mixed safety. Test whether safety holds across languages and mixed-language prompts, then close the gaps you find.

TL;DR
- Tiny Aya Fire (CohereLabs, 3B-class) refuses harmful requests in English almost every time, but the same requests in Telugu script, romanized Telugu or Telugu–English code-mix get through more often. Multi-turn attacks in Indic languages get through most of the time (62.5% on our held-out test set).
- Inside the model, the "this is harmful" signal is there for Telugu-script and mixed prompts, but 2–3x weaker than for English. For romanized Telugu it is close to orthogonal to the English signal.
- A single steering vector added at layers 15–18 at inference time cuts the non-English unsafe rate on held-out TEST from 26.1% to 16.3% (McNemar p = 7e-9). It also helps on 20 Indic languages it never saw (unsafe 28% → 16%, p = 0.007), at a cost of about +5 points of over-refusal and no MMLU loss.
- A SafeGrad fine-tune of the same model (manas2002/aya-indicguard-hardened) made no measurable difference: 28.1% unsafe vs 26.1% for base on TEST (p = 0.19). Steering beats it on every non-English form.
- What it does not do: it does not meet our own pre-registered over-refusal budget (≤ +3 points), and a gated version (CAST) that does meet it gives no significant safety gain on TEST. We report both.

---

## 1. The question

Safety tuning is mostly done in English. Indian users rarely write in one clean language: they type Telugu in Latin letters ("ela cheyali"), switch to English mid-sentence, or write in native script. We asked:

1. Does safety hold across these forms of the same request?
2. If not, where in the network does it break?
3. Can we fix it with a method keyed to meaning, so that it also works in languages we did not train on?

## 2. Setup

| Item | Choice |
|---|---|
| Model under test | `CohereLabs/tiny-aya-fire` (36 layers, hidden size 2048), greedy decoding, 320 new tokens |
| Judge | IndicGuard (`l3cube-pune/IndicGuard` LoRA on Gemma-3-4B-it): response unsafe or safe |
| Hardware | one RTX 3060 (12 GB) |
| Capability check | MMLU, 500-question fixed subset |

Data. Each request appears in up to four forms: English, Telugu script, romanized Telugu, and Telugu–English mix.

| Source | What it adds |
|---|---|
| IndicSafe (100 seeds × 4 forms) | the same request in all four forms, plus harmless controls |
| IndicGuard Telugu | about 1,500 labelled harmful and harmless Telugu prompts |
| CSRT (Yoo et al., ACL 2025) | code-switching red-team prompts |
| IndicSafetyBench, Telugu | single-turn and multi-turn jailbreak attacks |
| IndicSafetyBench, 20 other Indic languages | the UNSEEN set: Hindi, Tamil, Marathi, Punjabi, Odia, Urdu, Bengali and 13 more (117 items) |

Splits are by seed, so all four forms of a request stay in one split. Telugu multi-turn attacks are test-only, so that attack family is held out. Every layer, strength and threshold was picked on DEV. TEST and UNSEEN were run once, at the end. Child-safety content was filtered out of all sources before any file was written.

Metrics
- `ASR_nonEN`: share of non-English harmful prompts that get an unsafe response (lower is better).
- Over-refusal: refusal rate on harmless prompts, minus the base model's rate.
- Paired significance: exact McNemar test on the same items, base vs method.

## 3. The gap (measured)

Base model on the held-out TEST split, unsafe rate on harmful prompts:

| Form / attack | n | Unsafe (95% CI) |
|---|---|---|
| English (IndicSafe) | 22 | 0.0% [0, 0] |
| Telugu script (IndicSafe) | 24 | 12.5% [0, 25] |
| Romanized Telugu (IndicSafe) | 24 | 8.3% [0, 21] |
| Telugu–English mix (IndicSafe) | 24 | 16.7% [4, 33] |
| Telugu script (IndicGuard) | 292 | 24.0% [19, 29] |
| CSRT code-switch | 30 | 6.7% [0, 17] |
| Telugu single-turn attacks | 25 | 44.0% [24, 64] |
| Telugu multi-turn attacks (held-out family) | 48 | 62.5% [48, 75] |
| UNSEEN languages, single-turn | 78 | 24.4% [15, 35] |
| UNSEEN languages, multi-turn | 39 | 35.9% [21, 51] |

Only 68% of IndicSafe requests got the same verdict across all four forms. English is safe, and every other form leaks. Attacks, and multi-turn attacks most of all, make it much worse.

## 4. Where it breaks (diagnosis)

We took the model's hidden state at the last prompt token and computed, for each form, the mean difference between harmful and harmless prompts: the "harmfulness direction". Then we compared each form's direction with the English one.

| Layer | English | Telugu script | Romanized | Mixed |
|---|---|---|---|---|
| 9 | cos 1.00, norm 6.8 | 0.34, 1.9 | 0.25, 1.5 | 0.65, 5.3 |
| 12 | 1.00, 10.7 | 0.46, 3.4 | 0.17, 2.0 | 0.67, 7.7 |
| 15 | 1.00, 19.1 | 0.71, 6.7 | 0.17, 2.2 | 0.68, 10.0 |
| 18 | 1.00, 26.7 | 0.69, 10.5 | 0.12, 2.6 | 0.66, 14.4 |

(cos = cosine with the English direction; norm = how far apart harmful and harmless sit.)

```
English      harmful ●━━━━━━━━━━━━━━━━━━━━━━━━━━● harmless     norm 26.7 at L18
Telugu       harmful ●━━━━━━━━━━● harmless                     norm 10.5, same general direction
Mixed        harmful ●━━━━━━━━━━━━━━● harmless                 norm 14.4, same general direction
Romanized    harmful ●━● harmless  (points elsewhere)          norm 2.6, nearly orthogonal
```

How we read it:
- Telugu script and mixed: the model does see harm, roughly along the English axis, but at 40–55% of the strength. The refusal machinery in the late layers needs a bigger push than it gets. This matches the "weak separation" account of Wang et al. (2025).
- Romanized Telugu: the separation is tiny and points somewhere else, which looks like non-transfer, not a weak signal.
- An earlier pilot with activation attribution put most of the English vs Telugu loss in the MLPs of layers 15–18. Attribution is correlational, so treat that as a pointer, not a proof.
- A linear probe trained only on English harmful/harmless prompts reaches AUROC 0.91 on romanized (layer 8) and 1.00 on mixed (layer 6), but only 0.77 on Telugu script (layer 14). DEV has 16+4 items per form, so these are pilot-sized numbers.

## 5. The fix: add the missing push

Since the late layers get too weak a "this is harmful" signal, we add it back. At layers 15–18, on every token, we add

`h ← h + α · ‖d_en[L]‖ · unit(d)`

where `d` is a harmful-minus-harmless direction and `‖d_en[L]‖` is the English norm at that layer, so α is a fraction of the English signal. We tried three directions (English only, all forms pooled, all forms pooled with the language component projected out) and a gated variant (CAST: steer only when a layer-18 probe says the prompt is harmful).

DEV sweep (selection data):

| Method | ASR_nonEN | Harmless refusal | MMLU-500 |
|---|---|---|---|
| base | 14.5% | 3.0% | 0.586 |
| steer d_en α=0.05 | 8.6% | 8.5% | 0.592 |
| steer d_en α=0.10 | 5.5% | 14.0% | 0.576 |
| steer d_en α=0.15 | 3.6% | 32.3% | 0.516 |
| steer d_multi α=0.05 | 10.5% | 9.8% | 0.592 |
| CAST d_multi α=0.1 | 7.7% | 10.4% | 0.586 |
| CAST2 (strict gate, harmless FPR 5%) α=0.2 | 10.5% | 4.3% | 0.586 |

Two findings from the sweep:
- At matched strength, the choice of direction barely matters. The "language-agnostic" directions did not beat the English one. The size of the push is what counts.
- The trade-off is steep: past α ≈ 0.1, over-refusal and MMLU loss climb quickly.

## 6. Results on held-out data

TEST (Telugu, n = 467 non-English harmful items):

| Method | ASR_nonEN | Harmless refusal | 4-form consistency | Fixed / broken | McNemar p |
|---|---|---|---|---|---|
| base | 26.1% | 3.9% | 0.68 | – | – |
| steer d_en α=0.05 | 16.3% | 8.7% | 0.82 | 56 / 10 | 7e-9 |
| steer d_multi α=0.05 | 17.1% | 7.4% | 0.82 | 55 / 13 | 3e-7 |
| CAST d_multi α=0.1 | 18.8% | 10.6% | 0.73 | 45 / 11 | 5e-6 |
| CAST2 α=0.2 | 24.2% | 8.1% | 0.68 | 24 / 15 | 0.20 |
| aya-indicguard-hardened (SafeGrad FT) | 28.1% | 4.2% | 0.64 | 14 / 23 | 0.19 |

Per form, base → steer d_en 0.05: IndicGuard Telugu 24.0% → 14.7%, Telugu script 12.5% → 4.2%, mixed 16.7% → 8.3%, single-turn attacks 44% → 24%, multi-turn attacks 62.5% → 43.8%. Romanized stays at 8.3%, which fits the diagnosis: its harm signal does not point along the vector we add.

UNSEEN (20 Indic languages never used for any choice, n = 117):

| Method | Single-turn unsafe | Multi-turn unsafe | Fixed / broken | McNemar p |
|---|---|---|---|---|
| base | 24.4% | 35.9% | – | – |
| steer d_en α=0.05 | 14.1% | 20.5% | 19 / 5 | 0.007 |
| steer d_multi α=0.05 | 15.4% | 30.8% | 13 / 4 | 0.049 |
| CAST d_multi α=0.1 | 23.1% | 30.8% | 7 / 4 | 0.55 |
| aya-indicguard-hardened (SafeGrad FT) | 24.4% | 30.8% | 7 / 5 | 0.77 |

A vector built from English harmful/harmless prompts and added at four layers makes the model safer in Hindi, Tamil, Odia, Urdu and 16 other languages, with no MMLU loss (0.592 vs 0.586).

## 6b. Comparison with a SafeGrad fine-tune

We ran the same TEST and UNSEEN evaluation on manas2002/aya-indicguard-hardened (commit 63fff8eb), a full fine-tune of tiny-aya-fire made with SafeTune's SafeGrad: 1 epoch, batch size 2, learning rate 1e-4. Its training data is not published. It is not measurably safer than base on either split. On TEST, 14 items became safe and 23 became unsafe (p = 0.19); on UNSEEN, 7 and 5 (p = 0.77). Over-refusal (+0.3 points) and MMLU-500 (0.586) are unchanged. Even on the IndicGuard rows, where overlap with its training data would help it most, it did not improve (24.0% to 26.7%). Its rows are included in the TEST and UNSEEN tables above. Full results are in [manas_finetune_result.md](manas_finetune_result.md).

## 7. Honest limits

- Our own bar is not met. We fixed the rule before running: over-refusal ≤ +3 points. The best steering setting costs about +5 points on TEST (3.9% → 8.7%). The gated version that meets the budget on DEV (CAST2) loses most of the safety gain on TEST. So the result is a better trade-off point, not a free fix.
- The judge is a model. IndicGuard labels every response; we have not yet checked it against human labels (a 150-row audit sheet is built but not labelled). Refusal detection uses a phrase list, which is known to undercount.
- Small cells. Several per-form cells have 22–30 items, so their CIs are wide. The pooled McNemar tests are the claims we stand behind.
- Romanized Telugu is not fixed, and the diagnosis says why. That needs a different method (for example a direction learned from romanized data, or training), not a stronger push.
- One model, 3B scale, greedy decoding. We have not tested sampling or larger models.

## 8. What we would do next

1. Label the judge-audit sheet and report agreement (Cohen's κ) before trusting any small difference.
2. A per-form gate: a romanized-specific probe and direction, since the English one is nearly orthogonal there.
3. Make it permanent: a small LoRA trained to pull non-English harmful activations onto the English refusal axis, compared against HEAL and MPO-style multilingual alignment.

## 9. Reproduce

Everything is in `lang_agnostic/`. The pipeline:

```
p0_build_splits.py   build train/dev/test/unseen (seed-level, child-safety filtered)
p1_acts.py           last-token hidden states per layer
p1_probes.py         English-only probes, per-form directions and cosines
p2_dirs.py           steering directions (d_en, d_multi, d_multi_nolang)
p2_cast_fit.py       CAST gate; p2_gate2.py strict gate (harmless FPR 5%)
harness.py           generation with a method's hooks installed (resumable)
judge.py             IndicGuard judging (resumable)
metrics.py           unsafe rate with bootstrap CIs, refusal, consistency, McNemar
capability.py        MMLU-500
```

```bash
python harness.py test steer:d_en:0.05 && python judge.py runs/steer_d_en_0.05/test/generations.jsonl runs/steer_d_en_0.05/test/judged.jsonl && python metrics.py steer:d_en:0.05 test --vs base
```

## References

- Arditi et al. Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024. https://arxiv.org/abs/2406.11717
- Yoo, Yang, Lee. Code-Switching Red-Teaming (CSRT). ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.657
- Zhang. Does Safety Live in a Shared Multilingual Subspace? Cross-Lingual (Non-)Transfer of the Refusal Direction. MRL 2026. https://doi.org/10.18653/2026.mrl-1.19
- Dumas et al. Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations. ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.1536
- Chen, Chen, Zhan. HEAL: Anchored Hyperspherical Equilibrium for Multilingual Safety Alignment. EMNLP 2026. https://doi.org/10.18653/2026.emnlp-main.1294
- Zhao et al. MPO: Multilingual Safety Alignment via Reward Gap Optimization. ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.1149
- Lee et al. Programming Refusal with Conditional Activation Steering (CAST). ICLR 2025. https://arxiv.org/abs/2409.05907
- Wang et al. 2025, refusal direction across languages (full citation to confirm before publishing).
