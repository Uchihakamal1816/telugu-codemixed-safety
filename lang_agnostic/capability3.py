"""capability.py <method>: MMLU-500 letter-logit accuracy with the method's hooks."""
import json, os, sys, random, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL, REV
import methods3
name = sys.argv[1]; tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
M = methods3.get(name, model, tok)
ds = load_dataset("cais/mmlu", "all", split="test"); idx = random.Random(0).sample(range(len(ds)), 500)
L = [tok.encode(" " + c, add_special_tokens=False)[-1] for c in "ABCD"]
cap = {}
def fwd(**kw):
    # run one decoding step through the method's generate to keep hooks/gates identical
    o = M.generate(**kw, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True); return o.scores[0]
ok = 0
for n, i in enumerate(idx):
    q = ds[i]; p = q["question"] + "\n" + "\n".join(f"{c}. {t}" for c, t in zip("ABCD", q["choices"])) + "\nAnswer with the letter only."
    t = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) + "Answer:"
    enc = tok(t, return_tensors="pt", add_special_tokens=False).to(0)
    with torch.no_grad(): s = fwd(**enc)[0]
    ok += int(torch.stack([s[l] for l in L]).argmax().item() == q["answer"])
d = f"runs/{name.replace(':', '_')}"; os.makedirs(d, exist_ok=True)
json.dump({"mmlu500": ok / 500}, open(f"{d}/capability.json", "w")); print(name, "mmlu500", ok / 500)
