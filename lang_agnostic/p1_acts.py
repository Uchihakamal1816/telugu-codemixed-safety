"""T1.1: last-prompt-token hidden states, all 36 layers, fp16 -> acts/<split>.pt"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL, REV, jl
split = sys.argv[1]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
items = [i for i in jl(f"data/{split}.jsonl") if not i["multi_turn"]]
X = torch.zeros(len(items), 36, 2048, dtype=torch.float16)
for n, it in enumerate(items):
    ids = tok.apply_chat_template([{"role": "user", "content": it["turns"][0][:6000]}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(0)
    with torch.no_grad(): hs = model(**ids, output_hidden_states=True, logits_to_keep=1).hidden_states
    X[n] = torch.stack([h[0, -1] for h in hs[1:]]).half().cpu()
    if n % 200 == 0: print(n, len(items), flush=True)
torch.save({"X": X, "meta": [{k: it[k] for k in ("id", "source", "form", "label", "seed", "language")} for it in items]}, f"acts/{split}.pt")
print("saved", X.shape)
