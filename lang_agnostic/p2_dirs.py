"""T2.1: d_en, d_multi, d_multi_nolang at layers 15-18 (raw, unnormalized) -> runs/dirs.pt"""
import torch
tr = torch.load("acts/train.pt"); X = tr["X"].float(); meta = tr["meta"]
FORMS = ["english", "telugu_script", "romanized_derived", "mixed_derived"]
form = [m["form"] for m in meta]; harm = torch.tensor([m["label"] == "harmful" for m in meta])
def mask(f): return torch.tensor([x == f for x in form])
out = {"d_en": {}, "d_multi": {}, "d_multi_nolang": {}}
for L in range(15, 19):
    df = {f: X[mask(f) & harm, L].mean(0) - X[mask(f) & ~harm, L].mean(0) for f in FORMS}
    out["d_en"][L] = df["english"]
    dm = torch.stack(list(df.values())).mean(0); out["d_multi"][L] = dm
    mus = torch.stack([X[mask(f) & ~harm, L].mean(0) for f in FORMS]); mus = mus - mus.mean(0)
    U = torch.linalg.svd(mus, full_matrices=False).Vh[:3]
    out["d_multi_nolang"][L] = dm - (U.T @ (U @ dm))
    print(L, {k: round(v[L].norm().item(), 2) for k, v in out.items()})
torch.save(out, "runs/dirs.pt")
