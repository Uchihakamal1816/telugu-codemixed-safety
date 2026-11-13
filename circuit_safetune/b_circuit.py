"""CircuitKIT: where in Tiny Aya Fire does the refusal signal fail for mixed/romanized prompts?
Loads the model through CircuitKIT's TransformerLens cohere2 port, then decomposes the last-token
residual's projection on the refusal direction into per-head and per-MLP contributions (exact, linear)."""
import json, torch, circuitkit as ck
from common import MODEL, REV, jl
D = torch.load("direction.pt"); L = D["layer"]
from transformers import AutoConfig, AutoModelForCausalLM
# Only layers 0..L are analysed, and they do not depend on later layers, so a truncated model is exact
# and keeps the TransformerLens conversion inside the server's 15 GB RAM / 12 GB GPU.
cfg = AutoConfig.from_pretrained(MODEL, revision=REV); cfg.num_hidden_layers = L + 1; cfg.layer_types = cfg.layer_types[: L + 1]
import os
from transformers import AutoTokenizer
LOCAL = f"tiny_aya_fire_first{L + 1}"
if not os.path.exists(f"{LOCAL}/config.json"):
    hf = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, config=cfg, dtype=torch.bfloat16, device_map={"": "cpu"})
    hf.save_pretrained(LOCAL); AutoTokenizer.from_pretrained(MODEL, revision=REV).save_pretrained(LOCAL); del hf
model = ck.load_model(LOCAL, dtype="bfloat16", device="cuda", n_ctx=2048)
assert model.cfg.n_layers == L + 1
tok = model.tokenizer; nl = model.cfg.n_layers  # = L + 1
R = torch.nn.functional.normalize(D["diff"].float(), dim=-1).cuda()     # per-layer unit refusal directions
r = R[L]
names = lambda n: n.endswith("hook_resid_post") or n.endswith("hook_z") or n.endswith("hook_mlp_out") or n == "hook_embed"
MAX_TOKENS = 640
def analyse(prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda()
    if ids.shape[1] > MAX_TOKENS: return None  # TL's attention materialises ~T^2*d_model floats; >640 tokens OOMs on 12 GB
    with torch.no_grad(): _, c = model.run_with_cache(ids, names_filter=names)
    proj = [(c[f"blocks.{l}.hook_resid_post"][0, -1].float() @ R[l]).item() for l in range(nl)]
    heads, mlps = [], []
    for l in range(L + 1):
        z = c[f"blocks.{l}.attn.hook_z"][0, -1].float()                          # [head, d_head]
        out = torch.einsum("hd,hdm->hm", z, model.blocks[l].attn.W_O.float())
        heads.append((out @ r).tolist()); mlps.append((c[f"blocks.{l}.hook_mlp_out"][0, -1].float() @ r).item())
    return {"proj": proj, "heads": heads, "mlps": mlps, "total": proj[L]}
res = {}
for e in jl("eval_set.jsonl"):
    res[e["id"]] = analyse(e["prompt"])
harm, harmless = __import__("common").extraction_prompts()
ref = {k: [a["proj"] for a in map(analyse, ps[:40]) if a] for k, ps in (("harmful_en", harm), ("harmless_en", harmless))}
torch.save({"items": res, "ref": ref, "layer": L}, "circuit.pt"); print("circuit done", len(res), "skipped_long", sum(v is None for v in res.values()))
