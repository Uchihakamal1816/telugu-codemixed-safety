"""SafeTune: extract Tiny Aya Fire's refusal direction from English contrast prompts (Arditi et al.)."""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetune.steer import RefusalDirectionConfig, extract_refusal_direction
import os
from common import MODEL, REV, extraction_prompts, roman_extraction_prompts
TAG = os.environ.get("TAG", "")  # TAG=_roman: romanized-Telugu contrast set, layer fixed to the English run's layer
harmful, harmless = roman_extraction_prompts() if TAG == "_roman" else extraction_prompts()
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
_fwd = model.forward
def _last_logits_only(*a, **k):  # 262k-vocab logits at every position do not fit in 12 GB; only the last is used
    k.setdefault("logits_to_keep", 1); return _fwd(*a, **k)
model.forward = _last_logits_only
direction, layer, per_layer = extract_refusal_direction(model, tok, harmful, harmless, RefusalDirectionConfig(pick_layer=int(os.environ["PICK_LAYER"])) if "PICK_LAYER" in os.environ else RefusalDirectionConfig())
# Raw mean-difference norm at each layer, for activation-addition strength (Arditi adds the unnormalised diff).
def last_hidden(prompts):
    acc = None
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(0)
        with torch.no_grad(): hs = model(**ids, output_hidden_states=True).hidden_states
        v = torch.stack([h[0, -1].float() for h in hs[1:]])
        acc = v if acc is None else acc + v
    return acc / len(prompts)
diff = last_hidden(harmful) - last_hidden(harmless)
torch.save({"direction": direction.float().cpu(), "layer": layer, "diff": diff.cpu(),
            "per_layer": {k: v.float().cpu() for k, v in per_layer.items()}}, f"direction{TAG}.pt")
cos = torch.nn.functional.cosine_similarity(diff[layer], direction.float().to(diff.device), dim=0).item()
json.dump({"layer": layer, "n_layers": diff.shape[0], "diff_norm": diff[layer].norm().item(), "cos_with_safetune": cos,
           "n_harmful": len(harmful), "n_harmless": len(harmless)}, open(f"direction{TAG}.json", "w"), indent=1)
print(open(f"direction{TAG}.json").read())
