"""M1 swe: closed-form ridge edit of down_proj at L15-18. Keys = last-token down_proj inputs on TRAIN single-turn prompts.
Non-EN harmful -> +||d_en[L]||*unit(d_en[L]); harmless + English -> 0. lambda picked on DEV separation. Also saves English
harmful-harmless neuron gaps (for M5's safety-neuron mask)."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL, REV, jl
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
LAYERS = [15, 16, 17, 18]; buf = {}
for L in LAYERS:
    model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: buf.__setitem__(L, a[0][0, -1].float().cpu()))
def keys(split):
    its = [r for r in jl(f"data/{split}.jsonl") if not r["multi_turn"]]; K = {L: [] for L in LAYERS}
    for it in its:
        e = tok.apply_chat_template([{"role": "user", "content": it["turns"][0][:6000]}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(0)
        with torch.no_grad(): model(**e, logits_to_keep=1)
        for L in LAYERS: K[L].append(buf[L])
    y = torch.tensor([float(r["label"] == "harmful" and r["form"] != "english") for r in its])
    en = torch.tensor([r["form"] == "english" for r in its]); h = torch.tensor([r["label"] == "harmful" for r in its])
    return {L: torch.stack(v) for L, v in K.items()}, y, en, h
Kt, yt, ent, ht = keys("train"); Kd, yd, _, _ = keys("dev"); print("keys", Kt[15].shape, Kd[15].shape, flush=True)
E = torch.load("runs/dirs.pt")["d_en"]; out = {"dW": {}, "gap": {}}
for L in LAYERS:
    u = E[L].float() / E[L].norm(); v = u * E[L].norm()
    K = Kt[L].cuda(); G = K @ K.T; n = len(K); best = None
    for f in (1e-3, 1e-2, 1e-1, 1.0):
        lam = f * G.trace() / n
        a = torch.linalg.solve(G + lam * torch.eye(n, device=0), yt.cuda())   # coefficient per train key (scalar target along v)
        w = a @ K                                                               # (m,) : dW = v w^T
        pd = (Kd[L].cuda() @ w).cpu()
        sep = pd[yd == 1].mean() - pd[yd == 0].mean(); leak = pd[yd == 0].abs().mean()
        print(f"L{L} lam*{f}: dev push harmful-nonEN {pd[yd==1].mean():.3f} other {pd[yd==0].mean():.3f} |other| {leak:.3f}", flush=True)
        score = sep - leak
        if best is None or score > best[0]: best = (score, f, w)
    print(f"L{L} pick lam*{best[1]}")
    out["dW"][L] = torch.outer(v, best[2].cpu()).to(torch.bfloat16)
    out["gap"][L] = (Kt[L][ent & ht].mean(0) - Kt[L][ent & ~ht].mean(0)).abs()
torch.save(out, "runs/p4_swe.pt"); print("saved")
