"""T2.3 condition fit -> runs/cast_cond.pt + gate firing on dev per form."""
import json, torch, collections as C
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetune.steer import fit_cast_condition
from safetune.interventions.steer.cast import _collect_pooled_hidden, _cast_similarity, _condition_met
from common import MODEL, REV, jl
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
_f = model.forward
def _fw(*a, **k):
    k.setdefault("logits_to_keep", 1); return _f(*a, **k)
model.forward = _fw
tr = [i for i in jl("data/train.jsonl") if not i["multi_turn"] and i["form"] in ("english", "telugu_script", "romanized_derived", "mixed_derived")]
H = [i["turns"][0][:1500] for i in tr if i["label"] == "harmful"]; B = [i["turns"][0][:1500] for i in tr if i["label"] == "harmless"]
print("fit on", len(H), len(B), flush=True)
cond = fit_cast_condition(model, H, B, tok, candidate_layers=[8, 10, 12, 14, 16, 18], pool="last", device="cuda")
torch.save(cond, "runs/cast_cond.pt")
print("layer", cond.condition_layer, "thr", cond.threshold, "cmp", cond.comparator, "f1", getattr(cond, "f1", None), flush=True)
dv = [i for i in jl("data/dev.jsonl") if not i["multi_turn"]]
acts = _collect_pooled_hidden(model, tok, [i["turns"][0][:1500] for i in dv], [cond.condition_layer], "last", chat_template=True)
A = acts[cond.condition_layer] if isinstance(acts, dict) else acts
fire = C.defaultdict(list)
for it, h in zip(dv, A):
    fire[f"{it['source']}/{it['form']}/{it['label']}"].append(int(_condition_met(_cast_similarity(h, cond.condition_vector), cond.threshold, cond.comparator)))
r = {k: [sum(v) / len(v), len(v)] for k, v in sorted(fire.items())}
json.dump({"layer": cond.condition_layer, "threshold": cond.threshold, "comparator": cond.comparator, "fire_rate_dev": r}, open("runs/cast_gate.json", "w"), indent=1)
for k, v in r.items(): print(k, v)
