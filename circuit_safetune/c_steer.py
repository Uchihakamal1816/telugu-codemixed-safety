"""SafeTune steer: activation addition of the English refusal direction at one layer; regenerate the eval set."""
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetune.steer import RefusalDirectionModel
from common import MODEL, REV, jl
import os
TAG = os.environ.get("TAG", "")
D = torch.load(f"direction{TAG}.pt"); L = D["layer"]; vec = D["diff"][L]          # raw mean difference
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
items = jl("eval_set.jsonl"); MAX_NEW = 320
def gen(prompts, B=8):
    outs = []
    for i in range(0, len(prompts), B):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i+B]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.no_grad(): o = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(i, len(prompts), flush=True)
    return outs
with open(f"generations{TAG}.jsonl", "a", encoding="utf8") as f:
    for s in [float(x) for x in sys.argv[1:]]:
        h = RefusalDirectionModel(model, vec.to(0), mode="steer", strength=s, layers=[L]).install() if s else None
        for e, o in zip(items, gen([e["prompt"] for e in items])):
            f.write(json.dumps({**e, "strength": s, "output": o}, ensure_ascii=False) + "\n")
        f.flush()
        if h: h.remove()
print("steer done")
