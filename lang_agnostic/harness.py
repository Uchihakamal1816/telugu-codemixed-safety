"""Generate for a split with a method. Resumable. usage: harness.py <split> <method> [--limit N] [--filter src1,src2]"""
import json, os, sys, time, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL, REV, jl
import methods
ap = argparse.ArgumentParser(); ap.add_argument("split"); ap.add_argument("method"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--filter", default=""); ap.add_argument("--ids", default="")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
M = methods.get(a.method, model)
items = jl(f"data/{a.split}.jsonl")
if a.filter: items = [i for i in items if i["source"] in a.filter.split(",")]
if a.ids: keep = set(open(a.ids).read().split()); items = [i for i in items if i["id"] in keep]
if a.limit: items = items[:a.limit]
d = f"runs/{a.method.replace(':', '_')}/{a.split}"; os.makedirs(d, exist_ok=True); out = f"{d}/generations.jsonl"
done = {json.loads(l)["id"] for l in open(out, encoding="utf8")} if os.path.exists(out) else set()
todo = [i for i in items if i["id"] not in done]; print("todo", len(todo), flush=True)
MAX_NEW = 320
def gen(chats, B=8):
    outs = []
    for i in range(0, len(chats), B):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats[i:i+B]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.no_grad(): o = M.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        outs += tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs
t0 = time.time()
single = [i for i in todo if not i["multi_turn"]]; multi = [i for i in todo if i["multi_turn"]]
with open(out, "a", encoding="utf8") as f:
    for s in range(0, len(single), 32):
        chunk = single[s:s+32]
        for it, o in zip(chunk, gen([[{"role": "user", "content": it["turns"][0]}] for it in chunk])):
            f.write(json.dumps({**it, "method": a.method, "output": o}, ensure_ascii=False) + "\n")
        f.flush(); print(s + len(chunk), len(single), f"{time.time()-t0:.0f}s", flush=True)
    for it in multi:
        chat = []
        for t in it["turns"]:
            chat.append({"role": "user", "content": t}); chat.append({"role": "assistant", "content": gen([chat], B=1)[0]})
        f.write(json.dumps({**it, "method": a.method, "output": chat[-1]["content"], "chat": chat}, ensure_ascii=False) + "\n"); f.flush()
n = len(todo); print(f"done {n} items, {(time.time()-t0)/max(n,1):.2f} s/item", flush=True)
