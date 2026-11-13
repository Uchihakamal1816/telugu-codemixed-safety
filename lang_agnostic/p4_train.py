"""p4_train.py dpo|nlora. M4 dpo: LoRA DPO (beta 0.1, ref = adapter off). M5 nlora: SFT LoRA on down_proj L15-18, LoRA input masked to top-1% safety neurons."""
import json, random, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from common import MODEL, REV
mode = sys.argv[1]; R = random.Random(0)
PH = [p.strip().lower() for p in open("refusal_phrases.txt", encoding="utf8") if p.strip()]
ref = lambda o: len(o.strip()) < 20 or any(p in o[:200].lower() for p in PH)
D = json.load(open("runs/p4_prefs.json"))
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}); model.config.use_cache = False
def ids(p, a):
    pi = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"][-512:]
    ai = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"][:200]
    x = torch.tensor([pi + ai], device=0); lab = x.clone(); lab[0, :len(pi)] = -100; return x, lab
def logp(m, x, lab):
    k = int((lab != -100).sum()) + 1
    lg = m(input_ids=x, logits_to_keep=k).logits[:, :-1].float(); t = lab[:, -(k - 1):]; msk = t != -100
    return (torch.log_softmax(lg, -1).gather(-1, t.clamp(min=0)[..., None])[..., 0] * msk).sum()
if mode == "dpo":
    pairs = [(d["prompt"], d["steer"], d["base"]) for d in D if d["label"] == "harmful" and ref(d["steer"]) and not ref(d["base"])]
    nb = [(d["prompt"], d["base"], d["steer"]) for d in D if d["label"] == "harmless" and ref(d["steer"]) and not ref(d["base"])]
    print("harm pairs", len(pairs), "boundary pairs", len(nb)); pairs += nb
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "o_proj", "down_proj"], layers_to_transform=list(range(8, 24))))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads(); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5); ACC = 8
    for ep in range(2):
        R.shuffle(pairs); tot = acc = 0
        for i, (p, c, r) in enumerate(pairs):
            xc, lc = ids(p, c); xr, lr_ = ids(p, r)
            with torch.no_grad(), model.disable_adapter(): rc, rr = logp(model, xc, lc), logp(model, xr, lr_)
            z = 0.1 * ((logp(model, xc, lc) - rc) - (logp(model, xr, lr_) - rr))
            l = -torch.nn.functional.logsigmoid(z); (l / ACC).backward(); tot += l.item(); acc += float(z > 0)
            if (i + 1) % ACC == 0: opt.step(); opt.zero_grad()
        print("epoch", ep, "loss", tot / len(pairs), "pref-acc", acc / len(pairs), flush=True)
    model.save_pretrained("runs/p4_dpo_lora")
else:
    data = [(d["prompt"], d["steer"]) for d in D if d["label"] == "harmful" and ref(d["steer"])] + [(d["prompt"], d["base"]) for d in D if d["label"] == "harmless"]
    look = sum(1 for d in D if d["label"] == "harmless" and ref(d["steer"]) and not ref(d["base"]))
    print("sft ex", len(data), "of which boundary look-alikes", look)
    gap = torch.load("runs/p4_swe.pt")["gap"]
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["down_proj"], layers_to_transform=[15, 16, 17, 18]))
    for L in (15, 16, 17, 18):
        A = model.base_model.model.model.layers[L].mlp.down_proj.lora_A["default"].weight
        mask = torch.zeros(A.shape[1], device=A.device, dtype=A.dtype); mask[gap[L].topk(int(0.01 * len(gap[L]))).indices] = 1
        with torch.no_grad(): A.mul_(mask)
        A.register_hook(lambda g, mask=mask: g * mask)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0); ACC = 8
    for ep in range(2):
        R.shuffle(data); tot = 0
        for i, (p, a) in enumerate(data):
            x, lab = ids(p, a); l = model(input_ids=x, labels=lab).loss; (l / ACC).backward(); tot += l.item()
            if (i + 1) % ACC == 0: opt.step(); opt.zero_grad()
        print("epoch", ep, "loss", tot / len(data), flush=True)
    model.save_pretrained("runs/p4_nlora_lora")
print("saved")
