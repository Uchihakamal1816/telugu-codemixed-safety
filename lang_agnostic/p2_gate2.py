"""Gate v2: same CAST condition vector/layer, threshold re-chosen so train harmless FPR <= 5%. -> runs/cast_cond2.pt"""
import json, copy, torch, collections as C
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetune.interventions.steer.cast import _collect_pooled_hidden, _cast_similarity, _condition_met
from common import MODEL, REV, jl
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
_f = model.forward
def _fw(*a, **k):
    k.setdefault("logits_to_keep", 1); return _f(*a, **k)
model.forward = _fw
cond = torch.load("runs/cast_cond.pt", weights_only=False); L = cond.condition_layer
def sims(items):
    a = _collect_pooled_hidden(model, tok, [i["turns"][0][:1500] for i in items], [L], "last", chat_template=True)
    a = a[L] if isinstance(a, dict) else a
    return [float(_cast_similarity(h, cond.condition_vector)) for h in a]
tr = [i for i in jl("data/train.jsonl") if not i["multi_turn"]]
s = sims(tr); hb = sorted(x for x, i in zip(s, tr) if i["label"] == "harmless"); hm = [x for x, i in zip(s, tr) if i["label"] == "harmful"]
k = int(0.05 * len(hb))
fpr = lambda t: sum(_condition_met(x, t, cond.comparator) for x in hb) / len(hb)
thr = min((hb[k], hb[-k - 1]), key=fpr); print("FPR", fpr(thr))
c2 = copy.deepcopy(cond); c2.threshold = thr; torch.save(c2, "runs/cast_cond2.pt")
tpr = sum(_condition_met(x, thr, cond.comparator) for x in hm) / len(hm)
print("layer", L, "cmp", cond.comparator, "old thr", cond.threshold, "new thr", thr, "train TPR", round(tpr, 3), flush=True)
dv = [i for i in jl("data/dev.jsonl") if not i["multi_turn"]]
fire = C.defaultdict(list)
for it, x in zip(dv, sims(dv)): fire[f"{it['source']}/{it['form']}/{it['label']}"].append(int(_condition_met(x, thr, cond.comparator)))
r = {k: [round(sum(v) / len(v), 3), len(v)] for k, v in sorted(fire.items())}
json.dump({"threshold": thr, "train_tpr": tpr, "fire_rate_dev": r}, open("runs/cast_gate2.json", "w"), indent=1)
for k, v in r.items(): print(k, v)
