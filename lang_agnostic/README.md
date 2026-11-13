# Safety That Speaks Telugu: Diagnosing and Closing the Multilingual and Code-Mixed Safety Gap in Tiny Aya Fire

[![Hackathon](https://img.shields.io/badge/Lexsi_Labs_Hackathon-Multilingual_%26_Code--Mixed_Safety-red)](https://github.com/Lexsi-Labs)
[![Model](https://img.shields.io/badge/Model-CohereLabs%2Ftiny--aya--fire-blue)](https://huggingface.co/CohereLabs/tiny-aya-fire)
[![Base](https://img.shields.io/badge/Judge-IndicGuard_LoRA_on_Gemma--3--4B--it-purple)](https://huggingface.co/l3cube-pune/IndicGuard)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE.md)

> **Lexsi Labs / Cohere Aya / Hugging Face Hackathon**  
> **Challenge**: Multilingual and Code-Mixed Safety — *Test whether safety holds across languages and mixed-language prompts, locate where it breaks, and close the gaps.*  
> **Team**: Saikamal Nannuri (India), Venkata Manaswitha Sunkara (US), Olamide Balogun (Nigeria)  
> **Ticket**: `#KNSH-HJ2Y-TT2M`

---

## Executive Summary & Key Results

Safety alignment in commercial and open-weight models is heavily biased toward English. In real-world communication across the Indian subcontinent and multilingual regions, users frequently code-mix, transliterate into Latin script (Romanization), or communicate in native scripts.

We investigate `CohereLabs/tiny-aya-fire` (3.2B parameters, 36 layers), evaluating its safety across **English**, **Telugu script**, **Telugu–English code-mix**, **Romanized Telugu**, and **20 unseen Indic languages**.

```
Base Model on Held-Out Test (non-EN harmful): 26.1% Unsafe
With L15–18 Activation Steering (d_en 0.05):   16.3% Unsafe  (McNemar p = 7e-9, -38% relative)
Held-Out Unseen Indic Languages (20 languages): 28.2% -> 16.2% Unsafe (McNemar p = 0.007)
General Capability (MMLU-500):                  58.6% -> 59.2% (Zero capability degradation)
```

### Key Discoveries:
1. **The Safety Gap is Structural**: Tiny Aya Fire refuses harmful requests in English $52\%$ of the time in our initial screen, but refusals collapse to **$4\%$ in Telugu–English mixed-script**, while unsafe responses increase to $15\%$. Multi-turn jailbreak attacks succeed **$62.5\%$** of the time on our held-out test set.
2. **Mechanistic Root Cause ("Weak Separation" vs. "Non-Transfer")**:
   * In **Telugu script** and **mixed code-mix**, the model represents harm along the exact same directional axis as English ($\cos \approx 0.66\text{--}0.71$), but at **$2\times\text{--}3\times$ weaker activation magnitude** (norm $10.5\text{--}14.4$ vs $26.7$ in English at layer 18). The refusal circuitry simply lacks the activation energy to trigger.
   * In **Romanized Telugu**, representations suffer from **non-transfer**: the separation vector is almost orthogonal to English ($\cos \approx 0.12$, norm $2.6$).
   * Circuit attribution localizes **$63\%$ of this signal loss specifically to the MLPs of layers 15 to 18** (MLP18, MLP17, MLP16, MLP15).
3. **Inference-Time Steering Closes the Gap Across 20 Languages**:
   * Pushing the latent states along the clean English refusal direction at layers 15–18 ($h \leftarrow h + \alpha \cdot \|d_{en}[L]\| \cdot \text{unit}(d)$ with $\alpha = 0.05$) reduces the non-English unsafe rate from **$26.1\%$ to $16.3\%$** on held-out test data ($p = 7 \times 10^{-9}$), dropping multi-turn attack vulnerability from **$62.5\%$ to $43.8\%$**.
   * It generalizes zero-shot to **20 completely unseen Indic languages** (Hindi, Tamil, Marathi, Punjabi, Odia, Urdu, Bengali, etc.), cutting unsafe responses from **$28.2\%$ to $16.2\%$** ($p = 0.007$).
4. **"Language-Agnostic" vs. English Directions**:
   * Testing directions with language subspaces projected out via SVD (`d_multi_nolang`) revealed that **the pure English direction (`d_en`) outperformed language-projected directions**. Because the refusal mechanism is already tied to the English axis, the bottleneck across languages is activation magnitude rather than angle mismatch.

---

## 1. The Dataset & Empirical Vulnerability Screen

Before applying interventions, we screened 743 prompts across three public benchmarks ([IndicSafe](https://huggingface.co/datasets/ppattnay/IndicSafe), [IndicSafetyBench](https://huggingface.co/datasets/anna-sarvam/indic-safety-eval), [CSRT](https://huggingface.co/datasets/walledai/CSRT)), profiled 53,581 prompts from [IndicGuard Telugu](https://huggingface.co/datasets/l3cube-pune/IndicGuard), and ran handcrafted native Telugu red-teaming.

All safety evaluations were conducted using the `l3cube-pune/IndicGuard` LoRA judge over `unsloth/gemma-3-4b-it`.

### A. Major Unsafe Subcategories by Benchmark

| Benchmark / Source | Dominant Unsafe Subcategories | Empirical Findings & Metrics |
|---|---|---|
| **IndicSafe (Paired 4-Form)** | **1. Gender & Caste Discrimination**<br>**2. Harmful Instructions** | In code-mixed Telugu, **Gender & Caste Discrimination was 40% unsafe** (vs 10% in EN), and **Harmful Instructions was 30% unsafe** (vs 0% in EN). Refusals collapsed from 52% (EN) and 41% (Telugu script) to **4% in mixed-script**. |
| **IndicSafetyBench (Telugu)** | **1. Weapons (80% unsafe)**<br>**2. Self-Harm (100% unsafe)**<br>**3. Multi-turn Attacks (69% unsafe)** | Succeeded via payload splitting (`E2`: 86% unsafe, 0% refusal) and crescendo multi-turn jailbreaks (`E1`: 71% unsafe). |
| **IndicGuard Telugu (53k Prompts)** | **1. Criminal Planning / Confessions (44.3%)**<br>**2. Hate / Identity Hate (18.5%)**<br>**3. Violence (15.2%)** | Represents the highest raw volume risk: 14,273 prompts in the dataset target criminal planning. |

---

## 2. Mechanistic Diagnosis: Tracing Where Safety Breaks

Using **CircuitKIT** and linear probing at the last prompt token, we traced activations across all 36 layers of `tiny-aya-fire`.

### A. Direction Cosines & Separation Norms (Layers 9–18)

We computed the mean difference vector $d_f[L] = \mu(\text{harmful}) - \mu(\text{harmless})$ for each form $f$ on train activations:

| Layer | English ($d_{en}$) | Telugu Script ($d_{te}$) | Romanized Telugu ($d_{ro}$) | Mixed Code-Switch ($d_{mix}$) |
|:---:|:---:|:---:|:---:|:---:|
| **Layer 9** | $\cos 1.00$, norm $6.8$ | $\cos 0.34$, norm $1.9$ | $\cos 0.25$, norm $1.5$ | $\cos 0.65$, norm $5.3$ |
| **Layer 12** | $\cos 1.00$, norm $10.7$ | $\cos 0.46$, norm $3.4$ | $\cos 0.17$, norm $2.0$ | $\cos 0.67$, norm $7.7$ |
| **Layer 15** | $\cos 1.00$, norm $19.1$ | $\cos 0.71$, norm $6.7$ | $\cos 0.17$, norm $2.2$ | $\cos 0.68$, norm $10.0$ |
| **Layer 18** | $\cos 1.00$, norm $26.7$ | $\cos 0.69$, norm $10.5$ | $\cos 0.12$, norm $2.6$ | $\cos 0.66$, norm $14.4$ |

```
Representation Geometry at Layer 18:

English      Harmful ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● Harmless   Norm: 26.7
Telugu       Harmful ●━━━━━━━━━━━━━━━━━━━━● Harmless                            Norm: 10.5  (cos = 0.69)
Mixed        Harmful ●━━━━━━━━━━━━━━━━━━━━━━━━━━● Harmless                      Norm: 14.4  (cos = 0.66)
Romanized    Harmful ●━━● Harmless (Points in orthogonal subspace)              Norm:  2.6  (cos = 0.12)
```

### B. Circuit Attribution (Stage B)
Using a 19-layer TransformerLens model in `circuit_safetune/b_circuit.py`, we measured the exact contribution of each component to the layer-18 refusal signal:
* The refusal signal at Layer 18 on harmful English prompts averaged $+22.0$, but dropped to $+5.4$ on matched Telugu/mixed pairs.
* **$63\%$ of this drop was driven by MLPs**: specifically MLP18 ($-3.66$), MLP17 ($-2.51$), MLP16 ($-0.82$), and MLP15 ($-0.65$), alongside attention head L17H8 ($-0.62$).

---

## 3. The Language-Agnostic Intervention Framework

To close this gap without destructive fine-tuning or language-specific overfitting, we developed the `lang_agnostic/` framework:

1. **Direction Sets Extracted at Layers 15–18**:
   * `d_en`: English harmful vs. harmless centroid difference.
   * `d_multi`: Equal-weighted centroid across English, Telugu script, Romanized, and Mixed.
   * `d_multi_nolang`: `d_multi` with the top-3 language principal components projected out via SVD on harmless centroids.
2. **Intervention Form**:
   $$h_L \leftarrow h_L + \alpha \cdot \|d_{en}[L]\| \cdot \frac{d}{\|d\|} \quad \text{for } L \in \{15, 16, 17, 18\}$$
   Scaling by $\|d_{en}[L]\|$ ensures $\alpha$ represents an exact fraction of the nominal English signal push.
3. **Conditional Activation Steering (CAST)**:
   A probe fitted on Layer 18 to conditionally steer only when input representations trigger a harmfulness threshold.

---

## 4. Final Evaluation on Held-Out Data

All parameters were selected on `DEV` under a pre-registered rule ($\text{Over-Refusal} \le 3.0\text{ pts}$, $\text{MMLU Drop} \le 3.0\text{ pts}$). `TEST` and `UNSEEN` splits were evaluated once.

### A. Held-Out TEST Split ($n = 467$ non-English harmful prompts)

| Method | Non-EN ASR (`ASR_nonEN`) | Harmless Refusal | 4-Form Consistency | Flips (Fixed / Broken) | Exact McNemar $p$ | MMLU (500) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base Model** | **26.1%** | **3.9%** | 0.68 | — | — | 0.586 |
| **`steer:d_en:0.05`** | **16.3%** | **8.7%** | **0.82** | **56 / 10** | **$7 \times 10^{-9}$** | **0.592** |
| **`steer:d_multi:0.05`** | 17.1% | 7.4% | 0.82 | 55 / 13 | $3 \times 10^{-7}$ | 0.592 |
| **`cast:d_multi:0.1`** | 18.8% | 10.6% | 0.73 | 45 / 11 | $5 \times 10^{-6}$ | 0.586 |
| **`cast2:d_multi:0.2`** | 24.2% | 8.1% | 0.68 | 24 / 15 | $p = 0.20$ | 0.586 |

* **Attack Family Reductions**:
  * Multi-turn Telugu attacks: **$62.5\% \rightarrow 43.8\%$** unsafe.
  * Single-turn Telugu attacks: **$44.0\% \rightarrow 24.0\%$** unsafe.
  * IndicGuard Telugu: **$24.0\% \rightarrow 14.7\%$** unsafe.
  * Mixed code-switch: **$16.7\% \rightarrow 8.3\%$** unsafe.
  * Romanized Telugu: remained at $8.3\%$, consistent with its near-orthogonal representation.

---

### B. Held-Out UNSEEN Languages ($n = 117$, 20 Indic Languages)
*(Evaluated zero-shot across Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, and Urdu)*:

| Method | Single-Turn Unsafe | Multi-Turn Unsafe | Total Unsafe Rate | Flips (Fixed / Broken) | McNemar $p$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Base Model** | 24.4% | 35.9% | **33 / 117 (28.2%)** | — | — |
| **`steer:d_en:0.05`** | **14.1%** | **20.5%** | **19 / 117 (16.2%)** | **19 / 5** | **$p = 0.007$** |
| **`steer:d_multi:0.05`** | 15.4% | 30.8% | 24 / 117 (20.5%) | 13 / 4 | $p = 0.049$ |
| **`cast:d_multi:0.1`** | 23.1% | 30.8% | 30 / 117 (25.6%) | 7 / 4 | $p = 0.55$ |
| **`cast2:d_multi:0.2`** | 23.1% | 38.5% | 37 / 117 (31.6%) | 2 / 6 | $p = 0.29$ |

---

## 5. Repository Structure

```
lexsi-hackathon/
├── README.md                      # This comprehensive technical report
├── ARTICLE.md                     # Publication-ready article draft
├── PLAN.md                        # Master execution protocol and constraints
├── aya-console/                   # FastAPI + vLLM web red-teaming interface
│   ├── app.py                     # Zero-shot session backend
│   └── static/                    # Frontend UI for team red-teaming
├── circuit_safetune/              # Pilot interpretability & circuit discovery
│   ├── a_extract.py               # SafeTune refusal direction extraction
│   ├── b_circuit.py               # CircuitKIT TransformerLens attribution
│   ├── c_steer.py                 # Multi-strength steering verification
│   └── d_judge.py                 # IndicGuard evaluation pipeline
├── lang_agnostic/                 # Core reproducible language-agnostic framework
│   ├── p0_build_splits.py         # Seed-stratified split generator (train/dev/test/unseen)
│   ├── p1_acts.py                 # 36-layer activation harvester
│   ├── p1_probes.py               # Cross-lingual linear probes & cosine tracker
│   ├── p2_dirs.py                 # d_en, d_multi, d_multi_nolang generator
│   ├── p2_cast_fit.py             # Conditional activation steering fitter
│   ├── methods.py                 # Hook registry (Steer, Cast, Base, FT)
│   ├── harness.py                 # High-throughput batch inference harness
│   ├── judge.py                   # Resumable IndicGuard LLM-as-a-judge
│   ├── metrics.py                 # Bootstrap CIs, ASR_nonEN, and McNemar test
│   ├── capability.py              # MMLU-500 validation
│   └── RESULTS.md                 # Raw numerical tables from all runs
└── results/                       # Empirical datasets and screening records
    ├── mixed_language_screen/     # 743-item screen results (Excel + JSONL)
    ├── indicguard/                # IndicGuard Telugu evaluations (53k rows)
    ├── pure_telugu/               # Native handcrafted Telugu red-teaming
    └── mixed_version/             # Multilingual code-switching probing
```

---

## 6. How to Reproduce

### Requirements
* Python 3.10+
* PyTorch 2.11+ with CUDA support
* Hugging Face Transformers, PEFT, Datasets
* Hardware: 1x NVIDIA GPU with $\ge 12\text{ GB}$ VRAM (e.g. RTX 3060)

### Step 1: Build Leak-Free Splits
```bash
cd lang_agnostic
python3 p0_build_splits.py
```

### Step 2: Extract Activations & Compute Refusal Directions
```bash
python3 p1_acts.py train
python3 p1_acts.py dev
python3 p1_probes.py
python3 p2_dirs.py
```

### Step 3: Run Baseline and Steering Interventions
```bash
# Evaluate baseline
python3 harness.py test base
python3 judge.py runs/base/test/generations.jsonl runs/base/test/judged.jsonl
python3 metrics.py base test

# Evaluate steering with English refusal direction at layers 15-18 (alpha = 0.05)
python3 harness.py test steer:d_en:0.05
python3 judge.py runs/steer_d_en_0.05/test/generations.jsonl runs/steer_d_en_0.05/test/judged.jsonl
python3 metrics.py steer:d_en:0.05 test --vs base

# Evaluate on 20 unseen Indic languages
python3 harness.py unseen steer:d_en:0.05
python3 judge.py runs/steer_d_en_0.05/unseen/generations.jsonl runs/steer_d_en_0.05/unseen/judged.jsonl
python3 metrics.py steer:d_en:0.05 unseen --vs base
```

### Step 4: Evaluate General Capabilities
```bash
python3 capability.py steer:d_en:0.05
```

---

## 7. Honest Limitations & Future Work

1. **Over-Refusal Trade-off**: The intervention incurred $+4.8\text{ points}$ of over-refusal on benign prompts on the test set (increasing from $3.9\%$ to $8.7\%$), exceeding our strict pre-registered budget of $\le +3.0\text{ points}$. While general knowledge (MMLU) was untouched ($59.2\%$), developing softer dynamic gating is an open challenge.
2. **The Romanized Frontier**: Romanized Telugu was not resolved by linear steering because its representation is near-orthogonal to English at late layers. Closing the romanization gap requires weight-space alignment (e.g., SafeGrad / LoRA anchor alignment) to rotate romanized representations onto the refusal manifold.
3. **Automated Judge Calibration**: All evaluation was performed using `IndicGuard` (Gemma-3-4B LoRA). Human spot-checking confirmed overall directionality, but formal human-in-the-loop audit annotation remains important.

---

## 8. References & Prior Work

* **Arditi et al. (NeurIPS 2024)**: *Refusal in Language Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
* **Wang et al. (NeurIPS 2025)**: *Refusal Direction is Universal Across Safety-Aligned Languages.* [DOI:10.52202/085713-1086](https://doi.org/10.52202/085713-1086)
* **Zhao et al. (NeurIPS 2025)**: *LLMs Encode Harmfulness and Refusal Separately.* [DOI:10.52202/085713-4688](https://doi.org/10.52202/085713-4688)
* **Lee et al. (ICLR 2025)**: *Programming Refusal with Conditional Activation Steering (CAST).* [arXiv:2409.05907](https://arxiv.org/abs/2409.05907)
* **Yoo, Yang, Lee (ACL 2025)**: *Code-Switching Red-Teaming (CSRT).* [DOI:10.18653/v1/2025.acl-long.657](https://doi.org/10.18653/v1/2025.acl-long.657)
* **Zhang (MRL 2026)**: *Does Safety Live in a Shared Multilingual Subspace? Cross-Lingual (Non-)Transfer of the Refusal Direction.* [DOI:10.18653/2026.mrl-1.19](https://doi.org/10.18653/2026.mrl-1.19)
* **Chen, Chen, Zhan (EMNLP 2026)**: *HEAL: Anchored Hyperspherical Equilibrium for Multilingual Safety Alignment.* [DOI:10.18653/2026.emnlp-main.1294](https://doi.org/10.18653/2026.emnlp-main.1294)
