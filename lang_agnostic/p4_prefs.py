"""Shared data for M4/M5: TRAIN single-turn prompts answered by base and by strong d_en steering (alpha 0.25)."""
import json, random, torch, methods
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL, REV, jl
R = random.Random(0)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
tr = [r for r in jl("data/train.jsonl") if not r["multi_turn"]]
harm = R.sample([r for r in tr if r["label"] == "harmful" and r["form"] != "english"], 200)
safe = R.sample([r for r in tr if r["label"] == "harmless"], 200)
def gen(M, xs):
    out = []
    for i in range(0, len(xs), 8):
        t = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in xs[i:i+8]]
        e = tok(t, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.no_grad(): o = M.generate(**e, max_new_tokens=160, do_sample=False)
        out += tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)
    return out
P = [r["turns"][0][:2500] for r in harm + safe]
B = gen(methods.Base(model), P); S = gen(methods.Steer(model, "d_en", 0.25), P)
json.dump([{"prompt": p, "label": r["label"], "form": r["form"], "base": b, "steer": s} for p, r, b, s in zip(P, harm + safe, B, S)], open("runs/p4_prefs.json", "w"), ensure_ascii=False)
print("saved", len(P))
