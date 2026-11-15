# Language-agnostic safety steering: one English-derived direction narrows the multilingual and code-mixed safety gap in Tiny Aya Fire

One-line claim: safety in Tiny Aya Fire is encoded along a direction that is largely shared across languages (romanized Telugu is the exception) but weakened outside English. Adding back one direction, derived only from English prompts, at layers 15–18 at inference time cuts unsafe answers in 21 languages: from 26.1% to 16.3% on held-out Telugu, romanized and code-mixed prompts (p = 7e-9), and from 28.2% to 16.2% on 20 languages never used in development (p = 0.007). Telugu was the development language; the other 20 were held out.

> Content warning. This write-up and the linked dataset describe harmful requests (weapons, drugs, fraud, self-harm and similar) and contain model outputs that were judged unsafe. No harmful text is quoted in this README. All child-safety content was filtered out of every file before writing.

## Tracks and problem statements

- Track 2a: AI Safety and Red Teaming. Problem 4: Multilingual and code-mixed safety.
- Track 2b: AI Safety and Correction. Problem 4: Inference-time safety steering.
- Track 2b: AI Safety and Correction. Problem 5: Locating where safety lives.

## For judges: one minute

| | Base | Best pre-registered method (steer d_en α=0.05) |
|---|---|---|
| Non-English unsafe rate, held-out TEST (n = 467) | 26.1% | 16.3% (56 fixed / 10 broken, McNemar p = 7e-9) |
| Unsafe rate, 20 UNSEEN Indic languages (n = 117) | 28.2% | 16.2% (19 / 5, p = 0.007) |
| Harmless prompts refused, TEST | 3.9% | 8.7% |
| MMLU-500 | 0.586 | 0.592 |

- Base model + method in one line: `CohereLabs/tiny-aya-fire` + SafeTune refusal-direction steering, keyed to English, added at layers 15–18 with no retraining.
- Diagnosis: the model does detect harm in Telugu and mixed prompts, but the signal is 2–3x weaker than in English, so the late refusal layers do not fire. In romanized Telugu the signal points in a different direction altogether.
- Best exploratory result (designed after TEST was seen, so not a clean held-out claim): self-translate the prompt to English, answer with the same steering, translate the answer back. On a 183-item TEST subset this gives 21.1% → 4.9% unsafe (23 fixed / 3 broken, p = 9e-5) with no rise in over-refusal.
- What does not work: two fine-tunes. Neither a DPO LoRA nor a public SafeGrad fine-tune (`manas2002/aya-indicguard-hardened`) improved safety significantly.

## Abstract

Safety training is mostly done in English, so a multilingual model can refuse a harmful request in English and answer it in another language, another script, or a code-mixed sentence. We asked whether this gap has a language-agnostic cause and a language-agnostic fix: one intervention, built without any target-language data, that works across languages. We developed on Telugu, which gives four forms of the same request (English, native script, Latin-script "romanized" Telugu and Telugu–English code-mix). We then tested on 20 other Indic languages held out from every design choice.

On the same requests in four forms, the base model gives 0% unsafe answers in English but 8–25% in Telugu, romanized and mixed forms. Multi-turn jailbreaks in Telugu succeed 62.5% of the time.

To find where safety fails, we compared harmful-vs-harmless directions in the residual stream. For Telugu script and code-mix, the direction is close to the English one but 2–3x weaker at layer 18. For romanized Telugu it is nearly orthogonal (cosine 0.12). CircuitKIT attribution puts most of the English-vs-Telugu loss in the MLPs of layers 15–18.

The fix follows from the diagnosis: since the harm signal points along the same axis in every language but is weaker outside English, add the missing push along that shared axis. At layers 15–18 we add a SafeTune refusal direction, scaled to a fraction of the English signal's norm. We chose every setting on a DEV split and ran TEST and UNSEEN once. Non-English unsafe answers fell from 26.1% to 16.3%, and the gain carried over to 20 Indic languages never used in development (28.2% → 16.2%). MMLU did not drop. The price is about +5 points of over-refusal, which misses our own pre-registered limit of +3. A gated version (CAST) meets that limit but loses the safety gain.

We also report what did not work: DPO LoRA, a null-space LoRA, a closed-form weight edit, transliteration and a public SafeGrad fine-tune. An exploratory pipeline, which bridges through English by having the model translate its own input, gave the largest gain: 21.1% → 4.9% on a TEST subset.

## Problem statement and approach

Why this problem: most of the world does not talk to chatbots in English. In India alone, assistants for public services, agriculture or health get native-script, romanized and code-mixed input in 20+ languages by default, and safety-tuning data exists for very few of them. A fix that needs per-language safety data does not scale. A fix that is language-agnostic, built once from English and applied everywhere, does. Telugu (more than 120 million speakers) was our development language because it has all four input forms in one benchmark.

Pre-registered hypotheses (written in `code/PLAN.md` before any run):
- H1: an English-only harmfulness probe reaches AUROC ≥ 0.85 on every non-English form at some layer ≤ 14. Result: fail, narrowly. Romanized reached 0.91 and mixed 1.00, but Telugu script only 0.77.
- H3: conditional steering lowers DEV ASR_nonEN by ≥ 30% relative while keeping over-refusal ≤ +3 points and MMLU drop ≤ 3 points. Result: the ASR target is met (−41% for CAST at α = 0.1), but no setting met the over-refusal limit.
- H4: the final method lowers the unsafe rate on UNSEEN languages, McNemar p < 0.05. Result: pass (p = 0.007).

Threat model: an ordinary user, or a low-effort attacker, writes a harmful request in Telugu, romanized Telugu or code-mix, possibly over several turns. They have no access to the weights.

Related work:
- Arditi et al., Refusal in Language Models Is Mediated by a Single Direction, NeurIPS 2024. https://arxiv.org/abs/2406.11717
- Lee et al., Programming Refusal with Conditional Activation Steering (CAST), ICLR 2025. https://arxiv.org/abs/2409.05907
- Yoo, Yang, Lee, Code-Switching Red-Teaming (CSRT), ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.657
- Zhang, Does Safety Live in a Shared Multilingual Subspace?, MRL 2026. https://doi.org/10.18653/2026.mrl-1.19
- Dumas et al., Separating Tongue from Thought, ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.1536
- Chen, Chen, Zhan, HEAL: multilingual safety alignment, EMNLP 2026. https://doi.org/10.18653/2026.emnlp-main.1294
- Zhao et al., MPO: Multilingual Safety Alignment via Reward Gap Optimization, ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.1149

## Hackathon repos (kits) used

- SafeTune: `extract_refusal_direction` / `RefusalDirectionConfig` built the per-layer refusal directions (`circuit_safetune/a_extract.py`). `RefusalDirectionModel` ran the first steering pilot (`c_steer.py`). `fit_cast_condition` and the CAST internals fitted the conditional gate (`lang_agnostic/p2_cast_fit.py`, `p2_gate2.py`). The `harden.SafeGrad` checkpoint we compare against was also made with SafeTune.
- CircuitKIT: `ck.load_model` loads Tiny Aya Fire through CircuitKIT's TransformerLens cohere2 port. We then split the last-token projection on the refusal direction into exact per-head and per-MLP contributions, for English vs failing twins (`circuit_safetune/b_circuit.py`).
- AuditKIT: used by Olamide to test the model on non-Indic languages, including Yoruba. We also re-scored every logged Telugu/UNSEEN run with AuditKIT (`ak.evaluate` in precomputed mode with custom `@scorer`s, the built-in `RefusalDetector`, and `ak.compare(...).significance`); see Finding 6 (`lang_agnostic/audit_auditkit.py`). The main tables use our own judge pipeline, metrics and split builder, because they needed paired McNemar tests and seed-level splits across four language forms.
- CuratorKIT: not used.

How they connect:

```
 data (IndicSafe, IndicGuard, CSRT, IndicSafetyBench)
   │  p0_build_splits.py: seed-level train/dev/test/unseen, child-safety filter
   ▼
 SafeTune: refusal directions per layer ──► CircuitKIT: where the signal is lost (MLP15-18)
   │                                                 │
   └───────────────► steering at L15-18 ◄────────────┘   (+ SafeTune CAST gate)
                          │
            harness.py → IndicGuard judge → metrics.py (bootstrap CI, McNemar) + MMLU-500
```

## Models and licenses

- Base: `CohereLabs/tiny-aya-fire`, revision `73500c80fcb6c8bea3e830c21d5d2920249a0e1a`, bf16, no quantization. License CC-BY-NC-4.0 (Cohere Labs). Greedy decoding, 320 new tokens.
- Judge: `l3cube-pune/IndicGuard` LoRA on `unsloth/gemma-3-4b-it`.
- Compared against: `manas2002/aya-indicguard-hardened` (SafeTune SafeGrad, 1 epoch, lr 1e-4).

## Datasets, benchmarks and licenses

| Source | Role | Forms / languages |
|---|---|---|
| IndicSafe (100 seeds × 4 forms) | the same request in four forms | English, Telugu script, romanized, mixed |
| IndicGuard Telugu | labelled harmful and harmless prompts | Telugu script |
| CSRT | code-switching red-team prompts | code-mix |
| IndicSafetyBench, Telugu | single- and multi-turn jailbreaks | Telugu |
| IndicSafetyBench, 20 other languages | UNSEEN, test only | Hindi, Punjabi, Marathi, Tamil, Odia, Kannada, Urdu, Malayalam, Bengali, Kashmiri, Gujarati, Dogri, Assamese, Bodo, Santali, Maithili, Sindhi, Sanskrit, Manipuri, Nepali |
| MMLU, 500-question subset | capability retention | English |

Splits and leak check: splits are made at seed level, so all four forms of a request share one split. No id or seed appears in two splits; this is checked by direct match in `p0_build_splits.py`. Telugu multi-turn attacks are test-only (a held-out attack family). Final sizes: train 1,364, dev 400, test 799, unseen 117 (after the child-safety filter on public logs: 1,364 / 377 / 773 / 105). See `data/split_counts.md` in the dataset repo.

Over-refusal set: harmless IndicGuard Telugu prompts plus harmless IndicSafe controls in all four forms (310 in TEST).

Child-safety filter: every row whose category, judge output, prompt or response mentions minors was dropped. In the public logs this removed 62 item ids (1,487 rows across all method files), so recomputing metrics from the public files gives slightly different numbers from the tables below. The tables were computed before this filter.

## Metrics

- ASR_nonEN (lower is better): share of non-English harmful prompts whose response IndicGuard judges unsafe.
- Over-refusal (lower is better): refusal rate on harmless prompts, from a refusal-phrase detector, reported next to base.
- 4-form consistency (higher is better): share of IndicSafe seeds with the same verdict in all four forms.
- MMLU-500 accuracy (higher is better): the capability-retention check.
- Paired significance: exact McNemar test on the same items, base vs method, with 95% bootstrap CIs per cell.
- Checking the judge: not yet done. A 150-row human audit sheet exists but is unlabelled. This is our largest open gap.

## Method and experiments

Steering vector at layer L: `h ← h + α · ‖d_en[L]‖ · unit(d[L])`, applied at L = 15–18 on every token. `d` is a harmful-minus-harmless mean difference (English only, `d_en`; all forms, `d_multi`; or all forms with the language component removed, `d_multi_nolang`). Every α was swept on DEV and picked with a fixed rule: over-refusal ≤ +3 points and MMLU drop ≤ 3 points, then the lowest ASR. Seed 0 throughout.

DEV sweep:

| Method | ASR_nonEN | Harmless refusal | MMLU-500 |
|---|---|---|---|
| base | 14.5% | 3.0% | 0.586 |
| steer d_en α=0.05 | 8.6% | 8.5% | 0.592 |
| steer d_en α=0.10 | 5.5% | 14.0% | 0.576 |
| steer d_en α=0.15 | 3.6% | 32.3% | 0.516 |
| steer d_multi α=0.05 | 10.5% | 9.8% | 0.592 |
| CAST d_multi α=0.1 | 7.7% | 10.4% | 0.586 |
| CAST2 (gate tuned to 5% harmless FPR) α=0.2 | 10.5% | 4.3% | 0.586 |

## Results

### Finding 1: the gap (base model, TEST)

| Form / attack | n | Unsafe (95% CI) |
|---|---|---|
| English (IndicSafe) | 22 | 0.0% |
| Telugu script (IndicSafe) | 24 | 12.5% [0, 25] |
| Romanized Telugu | 24 | 8.3% [0, 21] |
| Telugu–English mix | 24 | 16.7% [4, 33] |
| Telugu script (IndicGuard) | 292 | 24.0% [19, 29] |
| Telugu single-turn attacks | 25 | 44.0% [24, 64] |
| Telugu multi-turn attacks | 48 | 62.5% [48, 75] |

### Finding 2: where it breaks

| Layer | English | Telugu script | Romanized | Mixed |
|---|---|---|---|---|
| 12 | cos 1.00, norm 10.7 | 0.46, 3.4 | 0.17, 2.0 | 0.67, 7.7 |
| 18 | cos 1.00, norm 26.7 | 0.69, 10.5 | 0.12, 2.6 | 0.66, 14.4 |

(Cosine with the English harmful-harmless direction, and that direction's norm.)

### Finding 3: same model before vs after, held-out TEST (n = 799; 467 non-English harmful)

| Method | ASR_nonEN | Harmless refusal | Consistency | Fixed / broken | McNemar p | MMLU-500 |
|---|---|---|---|---|---|---|
| base | 26.1% | 3.9% | 0.68 | – | – | 0.586 |
| steer d_en α=0.05 | 16.3% | 8.7% | 0.82 | 56 / 10 | 7e-9 | 0.592 |
| steer d_multi α=0.05 | 17.1% | 7.4% | 0.82 | 55 / 13 | 3e-7 | 0.592 |
| CAST d_multi α=0.1 | 18.8% | 10.6% | 0.73 | 45 / 11 | 5e-6 | 0.586 |
| CAST2 α=0.2 | 24.2% | 8.1% | 0.68 | 24 / 15 | 0.20 | 0.586 |
| SafeGrad fine-tune (manas2002) | 28.1% | 4.2% | 0.64 | 14 / 23 | 0.19 | 0.586 |

Per form, base → steer d_en: Telugu script 12.5% → 4.2%, mixed 16.7% → 8.3%, IndicGuard 24.0% → 14.7%, single-turn attacks 44% → 24%, multi-turn attacks 62.5% → 43.8%. Romanized stays at 8.3%, which matches Finding 2.

### Finding 4: transfer to 20 unseen languages (n = 117)

| Method | Single-turn | Multi-turn | Fixed / broken | p |
|---|---|---|---|---|
| base | 24.4% | 35.9% | – | – |
| steer d_en α=0.05 | 14.1% | 20.5% | 19 / 5 | 0.007 |
| steer d_multi α=0.05 | 15.4% | 30.8% | 13 / 4 | 0.049 |
| SafeGrad fine-tune | 24.4% | 30.8% | 7 / 5 | 0.77 |

### Finding 5 (exploratory): bridging through English

These methods were designed after the TEST results above were known. They were run on a fixed 183-item random subset of TEST (`sub_test`, 123 non-English harmful) and a 36-item UNSEEN subset. Treat them as hypotheses to confirm, not as held-out results.

| Method | sub_test ASR_nonEN | Harmless refusal | Fixed / broken | p | UNSEEN fixed / broken (p) | MMLU-500 |
|---|---|---|---|---|---|---|
| base | 21.1% | 6.2% | – | – | – | 0.586 |
| pivot-back + steer α=0.05 (translate to English, answer with steering, translate back) | 4.9% | 6.2% | 23 / 3 | 9e-5 | 8 / 0 (0.008) | not measured |
| xgate α=0.15 (translate only to decide; steer the original prompt) | 7.3% | 6.2% | 19 / 2 | 2e-4 | 5 / 0 (0.06) | not measured |
| steer d_en α=0.05 (same subset) | 16.3% | 4.2% | 9 / 3 | 0.15 | 6 / 1 (0.13) | 0.592 |

### Finding 6: independent re-scoring with AuditKIT

We loaded each run's logged outputs into AuditKIT as precomputed samples and scored the non-English harmful items with two scorers: `safe` (IndicGuard verdict) and `ak_refusal` (AuditKIT's built-in `RefusalDetector`). Each method was then paired with base by sample id using `ak.compare`. The numbers below use the public, child-safety-filtered logs (TEST n = 455, UNSEEN n = 105).

| Method | Split | Safe, base → method | AuditKIT bootstrap p | Centred bootstrap p |
|---|---|---|---|---|
| steer d_en α=0.05 | TEST | 73.8% → 83.5% | 0.51 | 0.0001 |
| steer d_multi α=0.05 | TEST | 73.8% → 82.6% | 0.51 | 0.0001 |
| CAST d_multi α=0.1 | TEST | 73.8% → 81.3% | 0.49 | 0.0001 |
| CAST2 α=0.2 | TEST | 73.8% → 76.0% | 0.50 | 0.10 |
| SafeGrad fine-tune | TEST | 73.8% → 72.7% | 0.59 | 0.38 |
| steer d_en α=0.05 | UNSEEN | 74.3% → 83.8% | 0.53 | 0.021 |
| steer d_multi α=0.05 | UNSEEN | 74.3% → 81.9% | 0.55 | 0.030 |
| CAST d_multi α=0.1 | UNSEEN | 74.3% → 76.2% | 0.70 | 0.53 |
| SafeGrad fine-tune | UNSEEN | 74.3% → 77.1% | 0.62 | 0.37 |

Two findings about AuditKIT itself (v1.1.2), reported as feedback for the kit:
- `paired_bootstrap` resamples the raw paired differences without centring them on zero, so the resampled means sit around the observed delta. About half of them then exceed it in absolute value, and p is about 0.5 for any effect size. Every AuditKIT p-value above is about 0.5, including a +9.7-point gain that McNemar puts at p = 7e-9. Centring the differences before resampling (last column) gives p-values in line with our McNemar tests.
- `RefusalDetector` matches only English refusal phrases. It found 0% over-refusal for every method and the same harmful-prompt refusal rate (1.3%) for all of them, because it does not detect refusals written in Telugu or other Indic languages. A multilingual refusal detector would be needed for this kind of evaluation.

### What didn't work

| Method | sub_test ASR_nonEN | Fixed / broken | p | Note |
|---|---|---|---|---|
| DPO LoRA (166 preference pairs, 2 epochs) | 24.4% | 3 / 7 | 0.34 | slightly worse |
| null-space LoRA | 16.3% | 13 / 7 | 0.26 | harmless refusal doubled to 12.5% |
| closed-form down_proj edit, β=0.1 | 17.9% | 9 / 5 | 0.42 | MMLU 0.598 |
| transliterate romanized → Telugu script, then steer | 20.3% | 9 / 8 | 1.0 | no gain |
| SafeGrad fine-tune (public checkpoint) | 28.1% (full TEST) | 14 / 23 | 0.19 | no gain |
| CAST with a strict gate | 24.2% (full TEST) | 24 / 15 | 0.20 | meets the over-refusal limit, loses the gain |

### Compute

One RTX 3060 (12 GB) on our own workstation, so the cash cost was zero. GPU-hours were not logged precisely. A full TEST run (799 items: generation and judging) takes about 40 minutes. The steering itself adds no measurable latency. The pivot-back method needs 3 generations per prompt.

## Limitations

- The judge has not been validated against humans: IndicGuard labels every response, and the 150-row audit is unlabelled.
- Our own over-refusal limit (+3 points) is not met by the best pre-registered method (+4.8 points on TEST).
- The exploratory methods (Finding 5) were chosen after TEST was seen, and their samples are small (123 and 36 harmful items).
- Romanized Telugu is not fixed by steering along the English direction.
- One 3B model, greedy decoding only. The over-refusal detector is a phrase list, which undercounts.
- We did not measure answer quality for pivot-back on harmless prompts beyond refusal.

## With more credits

- Label the judge audit (Cohen's κ against humans), then re-run pivot-back and xgate on the full TEST and UNSEEN sets with MMLU.
- Learn a romanized-specific direction and gate. Finding 2 says the English one cannot reach that form.
- Distil the pivot-back behaviour into a LoRA, so the model gets the English bridge in one pass instead of three.
- Expected: full-TEST ASR_nonEN under 8% with over-refusal within +3 points. This is a forecast, not a measurement.

## Artifacts

- Code (this repo, https://github.com/Uchihakamal1816/telugu-codemixed-safety): `lang_agnostic/` is the main pipeline; `circuit_safetune/` is the SafeTune + CircuitKIT pilot.
- HF dataset collection: https://huggingface.co/collections/uchihamadara1816/telugu-code-mixed-safety-datasets-6af755ccb216382e74df2c47 (repo: https://huggingface.co/datasets/uchihamadara1816/tiny-aya-fire-telugu-safety-evals). Contents: splits, every method's generations and judge verdicts, and metrics.
- HF model collection: https://huggingface.co/collections/uchihamadara1816/telugu-code-mixed-safety-models-6af755ca9994d1dfbb5d94c0 (repo: https://huggingface.co/uchihamadara1816/tiny-aya-fire-telugu-safety-steering). Contents: steering directions (`dirs.pt`), CAST gates, the closed-form edit, and the two negative-result LoRA adapters.

## Reproduction

Environment: Python 3.12, torch 2.11.0+cu128, transformers 5, peft, scikit-learn, SafeTune and CircuitKIT installed editable. GPU with ≥ 12 GB.

| File | What it does |
|---|---|
| `lang_agnostic/p0_build_splits.py` | builds train/dev/test/unseen with the child-safety filter |
| `p1_acts.py`, `p1_probes.py` | last-token activations; English-only probes and per-form directions |
| `p2_dirs.py`, `p2_cast_fit.py`, `p2_gate2.py` | steering directions; CAST gates |
| `methods.py`, `harness.py` | steering hooks; resumable generation |
| `methods3.py`, `harness3.py`, `p3_*`, `p4_*`, `p5_*` | exploratory methods (pivot-back, xgate, LoRA, DPO, weight edit) |
| `audit_auditkit.py` | AuditKIT re-scoring of logged runs (Finding 6) |
| `judge.py`, `metrics.py`, `capability.py` | IndicGuard judging; metrics with CIs and McNemar; MMLU-500 |
| `run_final.sh` | the full TEST/UNSEEN run for the main table |

```bash
cd lang_agnostic
python harness.py test steer:d_en:0.05
python judge.py runs/steer_d_en_0.05/test/generations.jsonl runs/steer_d_en_0.05/test/judged.jsonl
python metrics.py steer:d_en:0.05 test --vs base
```

## Declaration of work done outside the hackathon

None. All code and experiments were done between Nov 1 and Nov 15, 2026.

## Team

- Sai Kamal Nannuri: project lead; data splits, diagnosis, steering methods, evaluation pipeline and write-up.
- Venkata Manaswitha Sunkara: CircuitKIT and SafeTune integration (refusal-direction extraction, circuit attribution, steering and CAST hooks); local Telugu dataset scan.
- Olamide: testing on other languages such as Yoruba, and the AuditKIT evaluation.

## LLM usage statement

A large share of the code and of this write-up was drafted with an AI coding assistant (Claude), then run, checked and edited by the team. All numbers come from our own logged runs.

## Changelog

- v1 (2026-11-14): first submission.
- v2 (2026-11-15): team and author contributions added; AuditKIT use recorded.
- v4 (2026-11-15): reframed around the language-agnostic claim (development on Telugu, held-out test on 20 languages).
- v3 (2026-11-15): AuditKIT re-scoring of all logged runs (Finding 6), with two issues found in AuditKIT.
