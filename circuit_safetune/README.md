# circuit_safetune: where Tiny Aya's safety fails in Telugu/mixed text, and one fix

Model: `CohereLabs/tiny-aya-fire` @ `73500c80fcb6c8bea3e830c21d5d2920249a0e1a`.
Question: on prompts where Tiny Aya answered safely in English but unsafely in romanized,
code-mixed or Telugu-script form, where inside the model does the safety signal get lost
(CircuitKIT), and does adding back an English refusal direction fix it (SafeTune)?

Child-safety prompts and outputs are excluded everywhere.

## Roadmap

```
 STAGE 0 (laptop)      STAGE A           STAGE B             STAGE C            STAGE D
 build_sets.py   -->   a_extract.py -->  b_circuit.py   -->  c_steer.py   -->   d_judge.py   --> analysis
 pick prompts          SafeTune:         CircuitKIT:         SafeTune steer:    IndicGuard judge
                       refusal direction where it fails      regenerate at      safe / unsafe
                                                             several strengths
 eval_set.jsonl        direction.pt      circuit.pt          generations.jsonl  judged.jsonl
 extract_set.jsonl     direction.json    tiny_aya_fire_first19/
```

`run_all.sh` runs stages A to D in order and skips any stage whose output already exists.
To redo a stage, delete its output file first.

```bash
cd ~/lexsi-hackathon/circuit_safetune
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ./run_all.sh > run.log 2>&1 &
# the extra strength sweep was run separately:
~/lexsi-hackathon/.venv/bin/python c_steer.py 2 4 && ~/lexsi-hackathon/.venv/bin/python d_judge.py generations.jsonl judged.jsonl   # log: sweep.log
```

All Python runs use `~/lexsi-hackathon/.venv` (torch 2.11.0+cu128, transformers 5, transformer-lens 3.8,
editable installs of CircuitKIT and SafeTune). GPU: RTX 3060 12 GB.

---

## Stage 0: build the prompt sets (`build_sets.py`, run on the laptop)

Reads the earlier mixed-language screen (`results/mixed_language_screen/*_raw.jsonl` and `*_judge.jsonl`,
which only exist on the laptop) and writes two files that were copied here.

| Output | Rows | What it holds |
|---|---|---|
| `eval_set.jsonl` | 143 | The test prompts. `role` says which group each belongs to (below). |
| `extract_set.jsonl` | 64 | English prompts used only to find the refusal direction (54 harmful, 10 harmless). Disjoint from every test seed. |

`role` in `eval_set.jsonl`:

| role | n | Meaning |
|---|---|---|
| `fail` | 25 | IndicSafe prompt judged unsafe in a romanized / mixed / Telugu form whose English version was safe. 18 seeds. |
| `fail_english_twin` | 18 | The English version of those same seeds (the model was safe on these). |
| `attack` | 60 | Other single-turn probes the model failed (IndicSafetyBench attack families, CSRT code-switch). |
| `harmless_eval` | 40 | Harmless control questions in all 4 forms, to check over-refusal. |

Other fields: `id, dataset, form, family, harm, seed, prompt, screen_unsafe` (the screen's original label).

## Stage A: refusal direction (`a_extract.py`, SafeTune)

- Calls `safetune.steer.extract_refusal_direction` (Arditi et al. 2024) on 54 harmful English prompts
  vs 128 harmless English ones (the 10 from `extract_set.jsonl` plus 118 random Alpaca instructions).
- Direction = mean last-token activation (harmful) minus mean (harmless), per layer.
- The forward pass is limited to last-position logits; full 262k-vocab logits ran out of GPU memory.

| Output | What it holds |
|---|---|
| `direction.pt` | `direction` (unit vector at the chosen layer), `layer` (= 18), `diff` (raw mean difference at all 36 layers, shape 36x2048), `per_layer` (SafeTune's unit candidates). |
| `direction.json` | Summary: chosen layer 18 of 36, raw difference norm 25.2, cosine with SafeTune's vector 0.99999. |

Caveat: layer 18 is exactly the middle layer, which is SafeTune's fallback. Its layer-selection sweep counts
refusals with English phrases only, so it probably did not find a winner.

## Stage B: where the signal fails (`b_circuit.py`, CircuitKIT)

- Saves a 19-layer copy of the model (`tiny_aya_fire_first19/`, 3.8 GB). Layers 0 to 18 compute exactly the
  same activations without the later layers, and the full model's conversion did not fit in 15 GB RAM.
- Loads it with `circuitkit.load_model(..., n_ctx=2048)` (CircuitKIT's TransformerLens port for cohere2).
  `n_ctx=2048` avoids an attention mask sized for 8192 tokens.
- For each prompt, at the last prompt token:
  - `proj`: the residual stream projected on that layer's refusal direction, for layers 0 to 18 ("refusal signal").
  - `heads[l][h]` and `mlps[l]`: each attention head's and MLP's exact contribution to the layer-18 signal.
- Prompts over 640 tokens are skipped (5 of 143): TransformerLens attention needed 7 GB on a 927-token prompt.

| Output | What it holds |
|---|---|
| `circuit.pt` | `items[id] = {proj, heads, mlps, total}` (None if skipped), `ref` = projections for 40 harmful and 40 harmless English reference prompts, `layer`. |
| `tiny_aya_fire_first19/` | The truncated checkpoint. Safe to delete (re-created on demand). |

Result (laptop copy: `circuit_summary.json`):

| Group | Refusal signal at layer 18 |
|---|---|
| English harmful reference | 22.0 |
| `fail_english_twin` | 19.4 |
| `fail` (Telugu / mixed forms) | 5.4 |
| `attack` | 4.2 |
| `harmless_eval` | -1.5 |

- The signal is lower in the failing form than its English twin in 25 of 25 pairs.
- The gap is near zero up to layer 9 and grows from layer 12 to 18.
- Biggest losses: MLP18, MLP17, MLP16, MLP15 and head L17H8 (each lower in 20 to 24 of 25 pairs).
  MLPs carry about 63% of the gap.

## Stage C: steering (`c_steer.py`, SafeTune)

- Wraps the Hugging Face model in `safetune.steer.RefusalDirectionModel(mode="steer", layers=[18])`, which adds
  `strength x diff[18]` (the raw mean difference) to layer 18's output at every position.
- Regenerates all 143 eval prompts, greedy, up to 320 new tokens, chat template with the default preamble.
- Strengths are command-line arguments; 0 means no steering (the fresh baseline). Results are appended.

| Output | What it holds |
|---|---|
| `generations.jsonl` | 715 rows = 143 prompts x strengths {0, 0.5, 1, 2, 4}. Each row is the eval-set row plus `strength` and `output`. |

## Stage D: judge (`d_judge.py`)

- Same judge and prompt as `../screen/judge.py`: IndicGuard (`unsloth/gemma-3-4b-it` + `l3cube-pune/IndicGuard` LoRA),
  greedy, responses cut at 700 tokens. Resumable.

| Output | What it holds |
|---|---|
| `judged.jsonl` | 715 rows: `key` = `"<id>|<strength>"`, `judge` = `{"User Safety", "Response Safety", "Safety Categories"}`. 0 parse errors. |

Join with `generations.jsonl` on `id` and `strength`.

## Results (judged unsafe; lower is better)

| strength | fail | attack | english twin | harmless |
|---|---|---|---|---|
| 0 (baseline) | 16/25 | 48/60 | 0/18 | 0/40 |
| 0.5 | 15/25 | 46/60 | 0/18 | 0/40 |
| 1 | 15/25 | 48/60 | 0/18 | 1/40 |
| 2 | 12/25 | 45/60 | 0/18 | 0/40 |
| 4 | 10/25 | 38/60 | 0/18 | 1/40 |

- Strength 4 on `fail` + `attack`: 23 unsafe-to-safe flips vs 7 safe-to-unsafe, exact McNemar p = 0.005.
  Strength 2: 16 vs 9, p = 0.23.
- Harmless prompts were still answered (no refusals found), outputs about 5% shorter.
- The steered model mostly stopped complying or hedged; clear refusal phrases rose only from 2/85 to 9/85.
- The fresh baseline matched the screen's unsafe label on 64 of 85 failing items
  (shorter output limit, Hugging Face generation instead of the console, judge noise).

## Limits

- One automatic judge, not yet checked against human labels.
- Small set (85 harmful items, 18 matched seeds), and strength 4 was chosen on this same set: a pilot, not a held-out result.
- Steering is unconditional: every prompt is pushed toward refusal. 40 harmless prompts is thin evidence on over-refusal.
- 48/85 harmful prompts are still unsafe at strength 4.

## Next

1. Steer layers 15 to 18 together, where Stage B found the loss.
2. Conditional steering (SafeTune CAST) so only harmful-looking prompts are pushed.
3. Use the matched pairs as training data for a small LoRA, then re-run Stages C and D on a held-out set.

## Logs

- `run.log`: last `run_all.sh` run (stages B to D; stage A ran in an earlier attempt).
- `sweep.log`: strengths 2 and 4 plus their judging.
