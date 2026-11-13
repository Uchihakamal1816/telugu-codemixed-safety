"""B (TLI-style): LoRA on layers 1-4 trained so non-English last-token states at L8/L12 match the paired English prompt's
base states (cosine), plus a retain term keeping English states unchanged. Pairs: IndicSafe TRAIN seeds only."""
import random, torch, collections as C
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from common import MODEL, REV, jl
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0})
model.config.use_cache = False
S = C.defaultdict(dict)
for r in jl("data/train.jsonl"):
    if r["source"] == "IndicSafe": S[r["seed"]][r["form"]] = r["turns"][0]
pairs = [(v[f], v["english"]) for v in S.values() if "english" in v for f in ("telugu_script", "romanized_derived", "mixed_derived") if f in v]
eng = [v["english"] for v in S.values() if "english" in v]
print("pairs", len(pairs), "english", len(eng))
TL = [8, 12]
def enc(t): return tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(0)
def states(t):
    return [h[0, -1].float() for i, h in enumerate(model(**enc(t), output_hidden_states=True).hidden_states) if i in TL]
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], layers_to_transform=[1, 2, 3, 4]))
model.print_trainable_parameters()
with torch.no_grad(), model.disable_adapter():
    tgt = {t: states(t) for t in set(eng)}
def cosloss(a, b): return sum(1 - torch.nn.functional.cosine_similarity(x, y, 0) for x, y in zip(a, b)) / len(a)
with torch.no_grad(): print("init cos-loss", sum(cosloss(states(x), tgt[e]).item() for x, e in pairs) / len(pairs))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4); R = random.Random(0); ACC = 8
for ep in range(3):
    R.shuffle(pairs); tot = 0
    for i, (x, e) in enumerate(pairs):
        l = cosloss(states(x), tgt[e])
        e2 = R.choice(eng); l = l + 1.0 * cosloss(states(e2), tgt[e2])   # retain English
        (l / ACC).backward(); tot += l.item()
        if (i + 1) % ACC == 0: opt.step(); opt.zero_grad()
    print("epoch", ep, "loss", tot / len(pairs), flush=True)
with torch.no_grad(): print("final cos-loss (train pairs)", sum(cosloss(states(x), tgt[e]).item() for x, e in pairs) / len(pairs))
model.save_pretrained("runs/p3_tli_lora"); print("saved")
