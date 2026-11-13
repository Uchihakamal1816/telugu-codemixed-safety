"""Calibrate on DEV: AUROC (non-EN harmful vs harmless) of last-token vs span-max L18 probe, on original and deletor-rewritten prompts. thr = 90th pct of DEV harmless."""
import json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL, REV, jl
from methods5 import Scorer, Deletor, isref
from sklearn.metrics import roc_auc_score
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map={"": 0}).eval()
D = [d for d in jl("data/dev.jsonl") if not d["multi_turn"]]
D = [d for d in D if d["label"] == "harmless" or d["form"] != "english"]
S = Scorer(model, tok); y = np.array([d["label"] == "harmful" for d in D])
txt = [d["turns"][0] for d in D]; dl = Deletor(model, tok)(txt)
res = {}
for src, T in [("orig", txt), ("del", dl)]:
    sc = [S.scores(t) for t in T]
    for k in ["last", "span"]:
        v = np.array([s[k] for s in sc])
        if src == "del": v = np.where([isref(t) for t in T], 1e9, v)
        thr = float(np.percentile(v[~y], 90)); fin = np.clip(v, -1e3, 1e3)
        res[f"{src}_{k}"] = {"auroc": float(roc_auc_score(y, fin)), "thr": thr, "tpr": float((v[y] > thr).mean()), "fpr": float((v[~y] > thr).mean())}
        print(src, k, res[f"{src}_{k}"], flush=True)
res["deletor_refused"] = {"harmful": float(np.mean([isref(t) for t, l in zip(dl, y) if l])), "harmless": float(np.mean([isref(t) for t, l in zip(dl, y) if not l]))}
print(res["deletor_refused"]); json.dump(res, open("runs/p5_cal.json", "w"), indent=1)
json.dump([{"in": a, "out": b} for a, b in list(zip(txt, dl))[:20]], open("runs/p5_del_examples.json", "w"), ensure_ascii=False, indent=1)
