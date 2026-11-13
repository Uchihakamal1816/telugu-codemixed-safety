# Related work found 2026-11-11 (overnight)

All were found through OpenAlex. Abstracts were read for the items marked (abs); the rest are known from title and venue only.

## Directly threatens the novelty of PLAN T2.4 (cross-lingual alignment LoRA)
- (abs) Chen, Chen, Zhan. *HEAL: Anchored Hyperspherical Equilibrium for Multilingual Safety Alignment.* EMNLP 2026. https://doi.org/10.18653/2026.emnlp-main.1294
  An auxiliary loss pulls non-English representations toward a stop-gradient English anchor, plus a hinge that keeps the languages distinct. This is essentially our T2.4 alignment term.
- (abs) Bu, Liu, Ren, Yang, Dai. *Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment.* arXiv 2602.16660, 2026.
  A multilingual consistency (MLC) loss that makes representations of multilingual prompt variants collinear. It needs no responses in the low-resource languages.
- (abs) Qin et al. *Multilingual Safety Alignment via Self-Distillation.* arXiv 2605.02971, 2026.
  Distils the English safety behaviour into low-resource languages without any response data.
- (abs) Krasnodębska, Kusa, Lipani. *Multilingual Refusal Alignment for Safer LLMs.* Findings ACL 2026. https://doi.org/10.18653/v1/2026.findings-acl.1537
  English-only alignment is not enough; DPO on multilingual data helps without hurting Global MMLU.
- (title) Zhao et al. *MPO: Multilingual Safety Alignment via Reward Gap Optimization.* ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.1149

Implication: an alignment LoRA alone is not novel. If we train one, cite HEAL and MLC and present it as a baseline, or as applying these ideas to romanized and code-mixed Telugu. Our distinct angle is (1) locating where the failure happens inside the model (layers 12–18, MLP15–18) and (2) a fix aimed at that place, tested on unseen languages.

## Does the refusal direction transfer across languages? (mixed evidence)
- Wang et al. 2025 (NeurIPS): yes, it is universal across safety-aligned languages.
- (abs) Zhang. *Does Safety Live in a Shared Multilingual Subspace? Cross-Lingual (Non-)Transfer of the Refusal Direction.* MRL 2026. https://doi.org/10.18653/2026.mrl-1.19
  Pre-registered. The English direction changed refusal in 0 of 9 other languages (Qwen2.5-7B, Llama-3.2-3B).
- (abs) Ivanova. *Is Refusal Direction Universal across Low-Resourced Languages?* BlackboxNLP 2026. https://doi.org/10.18653/2026.blackboxnlp-1.65
  Transfer is graded: directions from strongly misaligned languages fail (Qwen2.5-14B).

Implication: our T1.3 per-form direction cosines and the d_en vs d_multi comparison test this question directly, on a new model (Tiny Aya) and new forms (romanized, mixed). That is a publishable angle either way.

## Code-mixing, romanization and Indic languages
- Yoo, Yang, Lee. *Code-Switching Red-Teaming (CSRT).* ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.657 (our CSRT data comes from here)
- (abs) Kabir et al. *BanglaShield.* arXiv 2610.24288, 2026. Bangla in native script, romanized and code-mixed forms; LLM classifiers miss many unsafe native-script prompts. This is evidence that the judge can be biased by form, which supports the Gate 0 kappa check.
- (title) Pattnayak, Chowdhuri. *IndicJR: A Judge-Free Benchmark of Jailbreak Robustness in South Asian Languages.* EACL 2026 Industry. https://doi.org/10.18653/v1/2026.eacl-industry.50. Worth reading: a judge-free metric would sidestep our dependence on IndicGuard.
- (title) Kachireddy. *A Measurement Crisis in Multilingual LLM Safety: Heuristic Classifiers Systematically Undercount Reasoning-Model Refusals in Indic Languages.* SSRN 2026. This is directly relevant to our phrase-based refusal detector.

## Background
- Yong, Menghini, Bach. *Low-Resource Languages Jailbreak GPT-4.* arXiv 2310.02446, 2023.
- Deng, Zhang, Pan, Bing. *Multilingual Jailbreak Challenges in LLMs.* arXiv 2310.06474, 2023.
- (title) Dumas et al. *Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations.* ACL 2025. https://doi.org/10.18653/v1/2025.acl-long.1536. This is the method precedent for T1.4 patching.
- (title, could not open: Springer login) Li et al. *TransAegis: Cross-Lingual Representation Steering-Based Low Resource Language Jailbreak Defense.* LNCS 2026. https://doi.org/10.1007/978-3-032-38410-2_19. This is the closest prior work to our T2.2/T2.3 steering; its abstract still needs reading.
