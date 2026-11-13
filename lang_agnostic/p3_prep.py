"""P3 prep: seed-0 stratified subsets of TEST/UNSEEN, per-form dirs (L15-18), upstream gate probes (L8, L11)."""
import json, random, collections as C, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from common import jl
R = random.Random(0)
def pick(xs, k): xs = sorted(xs, key=lambda r: r["id"]); return R.sample(xs, min(k, len(xs)))
te = jl("data/test.jsonl"); un = jl("data/unseen.jsonl"); ids = []
# IndicSafe: whole seeds so the 4 forms stay paired
seeds = C.defaultdict(list)
for r in te:
    if r["source"] == "IndicSafe": seeds[(r["label"], r["seed"])].append(r)
hs = sorted(s for l, s in seeds if l == "harmful"); ns = sorted(s for l, s in seeds if l == "harmless")
for s in R.sample(hs, 12): ids += [r["id"] for r in seeds[("harmful", s)]]
for s in ns: ids += [r["id"] for r in seeds[("harmless", s)]]
g = lambda **kw: [r for r in te if all(r[k] == v for k, v in kw.items())]
ids += [r["id"] for r in pick(g(source="IndicGuard", label="harmful"), 48) + pick(g(source="IndicGuard", label="harmless"), 24)
        + pick(g(source="CSRT"), 15) + pick(g(source="IndicSafetyBench", multi_turn=False), 12) + pick(g(source="IndicSafetyBench", multi_turn=True), 12)]
uids = [r["id"] for r in pick([r for r in un if not r["multi_turn"]], 24) + pick([r for r in un if r["multi_turn"]], 12)]
open("data/sub_test.ids", "w").write("\n".join(ids)); open("data/sub_unseen.ids", "w").write("\n".join(uids))
print("subset test", len(ids), "unseen", len(uids))
# per-form directions
tr = torch.load("acts/train.pt"); X = tr["X"].float(); meta = tr["meta"]
form = np.array([m["form"] for m in meta]); y = np.array([m["label"] == "harmful" for m in meta])
FORMS = ["english", "telugu_script", "romanized_derived", "mixed_derived"]
dirs = {f: {L: X[torch.tensor((form == f) & y), L].mean(0) - X[torch.tensor((form == f) & ~y), L].mean(0) for L in range(15, 19)} for f in FORMS}
# upstream gate probes, form-balanced, threshold set on DEV harmless (FPR <= 10%)
dv = torch.load("acts/dev.pt"); Xd = dv["X"].float(); yd = np.array([m["label"] == "harmful" for m in dv["meta"]])
fd = np.array([m["form"] for m in dv["meta"]])
w = np.zeros(len(y)); sel = np.isin(form, FORMS)
for f in FORMS:
    for c in (0, 1):
        m = sel & (form == f) & (y == c); w[m] = 1 / max(m.sum(), 1)
gates = {}
from sklearn.metrics import roc_auc_score
for L in (8, 11, 18):
    mu, sd = X[:, L].mean(0), X[:, L].std(0) + 1e-4
    clf = LogisticRegression(C=0.05, max_iter=3000).fit(((X[:, L] - mu) / sd)[sel].numpy(), y[sel], sample_weight=w[sel])
    s = clf.decision_function(((Xd[:, L] - mu) / sd).numpy())
    thr = float(np.quantile(s[~yd], 0.90))
    auc = {f: round(roc_auc_score(yd[fd == f], s[fd == f]), 3) for f in FORMS if len(set(yd[fd == f])) == 2}
    tpr = {f: round(float((s[(fd == f) & yd] > thr).mean()), 2) for f in FORMS}
    print(f"gate L{L} dev AUROC {auc} TPR@FPR10 {tpr}")
    gates[L] = {"mu": mu, "sd": sd, "w": torch.tensor(clf.coef_[0], dtype=torch.float32), "b": float(clf.intercept_[0]), "thr": thr}
torch.save({"dirs": dirs, "gates": gates}, "runs/p3.pt")
for f in FORMS: print(f, {L: round(torch.nn.functional.cosine_similarity(dirs[f][L], dirs['english'][L], 0).item(), 2) for L in range(15, 19)})
# materialise subset splits + reuse existing base / steer d_en generations (same items, same greedy decoding)
import os, shutil
for sp, src, keep in [("sub_test", "test", set(ids)), ("sub_unseen", "unseen", set(uids))]:
    with open(f"data/{sp}.jsonl", "w", encoding="utf8") as f:
        for r in jl(f"data/{src}.jsonl"):
            if r["id"] in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for m in ("base", "steer_d_en_0.05"):
        d = f"runs/{m}/{sp}"; os.makedirs(d, exist_ok=True)
        G = [r for r in jl(f"runs/{m}/{src}/generations.jsonl") if r["id"] in keep]
        J = [r for r in jl(f"runs/{m}/{src}/judged.jsonl") if r["key"].rsplit("|", 1)[0] in keep]
        for fn, rows in (("generations", G), ("judged", J)):
            with open(f"{d}/{fn}.jsonl", "w", encoding="utf8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(m, sp, len(G), len(J))
