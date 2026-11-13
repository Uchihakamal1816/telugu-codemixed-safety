"""T1.2 + T1.3: per-layer harmfulness probes (E: English-only, M: form-balanced), AUROC per form on dev; per-form direction cosines."""
import json, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tr, dv = torch.load("acts/train.pt"), torch.load("acts/dev.pt")
def arr(d): return d["X"].float().numpy(), np.array([m["label"] == "harmful" for m in d["meta"]]), np.array([m["form"] for m in d["meta"]])
Xt, yt, ft = arr(tr); Xd, yd, fd = arr(dv)
FORMS = ["english", "telugu_script", "romanized_derived", "mixed_derived"]
# English-only probe uses english rows; M uses the 4 forms, sample-weighted so each form x label cell has equal weight
selE = ft == "english"; selM = np.isin(ft, FORMS)
w = np.zeros(len(yt))
for f in FORMS:
    for y in (0, 1):
        c = selM & (ft == f) & (yt == y); w[c] = 1.0 / max(c.sum(), 1)
res = {"E": {}, "M": {}}
evalforms = FORMS + ["code-switch", "single-turn"]
for L in range(36):
    mu, sd = Xt[:, L].mean(0), Xt[:, L].std(0) + 1e-4
    Zt, Zd = (Xt[:, L] - mu) / sd, (Xd[:, L] - mu) / sd
    for name, sel, sw in [("E", selE, None), ("M", selM, w[selM])]:
        clf = LogisticRegression(C=0.05, max_iter=2000).fit(Zt[sel], yt[sel], sample_weight=sw)
        s = clf.decision_function(Zd); r = {}
        for f in FORMS:
            m = fd == f
            if len(set(yd[m])) == 2: r[f] = roc_auc_score(yd[m], s[m])
        # IndicGuard dev rows are telugu_script too; report them separately
        ig = np.array([mm["source"] == "IndicGuard" for mm in dv["meta"]])
        r["IndicGuard"] = roc_auc_score(yd[ig], s[ig])
        res[name][L] = r
json.dump(res, open("runs/diagnose_probes.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, name in zip(ax, ["E", "M"]):
    for f in FORMS + ["IndicGuard"]:
        ys = [res[name][L].get(f, np.nan) for L in range(36)]; a.plot(range(36), ys, label=f)
    a.axhline(0.85, ls="--", c="gray"); a.set_title(f"probe {name}"); a.set_xlabel("layer")
ax[0].set_ylabel("dev AUROC"); ax[1].legend(fontsize=8); plt.tight_layout(); plt.savefig("runs/diagnose_probes.png", dpi=120)
# H1
h1 = {f: max([(res["E"][L].get(f, 0), L) for L in range(15)]) for f in FORMS[1:]}
print("H1 best (AUROC, layer) at L<=14 per non-English form, probe E:", h1, "PASS" if all(v[0] >= .85 for v in h1.values()) else "FAIL")
# T1.3 per-form direction cosines
D = {}
for f in FORMS:
    m = ft == f; D[f] = Xt[m & yt].mean(0) - Xt[m & ~yt].mean(0)
tab = {}
for L in [9, 12, 15, 18]:
    tab[L] = {f: {"cos_en": float(np.dot(D[f][L], D["english"][L]) / np.linalg.norm(D[f][L]) / np.linalg.norm(D["english"][L])), "norm": float(np.linalg.norm(D[f][L]))} for f in FORMS}
json.dump({"H1": {k: list(v) for k, v in h1.items()}, "dir_cos": tab}, open("runs/diagnose_dirs.json", "w"), indent=1)
for L, v in tab.items(): print(L, {f: (round(x["cos_en"], 3), round(x["norm"], 1)) for f, x in v.items()})
