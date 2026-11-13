"""Method registry. name -> object with .ctx() (unconditional hooks) or .generate (gated)."""
import contextlib, torch
LAYERS = [15, 16, 17, 18]
def _layers(model): return model.model.layers
class Base:
    def __init__(self, model): self.model = model
    def generate(self, **kw): return self.model.generate(**kw)
class Steer:
    """Adds alpha * raw diff vector at each layer (unnormalized), every position."""
    def __init__(self, model, dirset, alpha):
        self.model = model; A = torch.load("runs/dirs.pt"); D, E = A[dirset], A["d_en"]
        # every direction scaled to d_en's per-layer norm, so alpha means the same push for all sets
        self.vecs = {L: (D[L] / D[L].norm() * E[L].norm() * alpha).to(0, torch.bfloat16) for L in LAYERS}
    def generate(self, **kw):
        hs = []
        for L, v in self.vecs.items():
            def hook(_m, _i, out, v=v):
                if isinstance(out, tuple): return (out[0] + v,) + out[1:]
                return out + v
            hs.append(_layers(self.model)[L].register_forward_hook(hook))
        try: return self.model.generate(**kw)
        finally:
            for h in hs: h.remove()
class Cast:
    def __init__(self, model, dirset, alpha, COND="runs/cast_cond.pt"):
        from safetune.steer import CASTModel
        _f = model.forward
        def _fw(*a, **k):
            k.setdefault("logits_to_keep", 1); return _f(*a, **k)
        model.forward = _fw
        A = torch.load("runs/dirs.pt"); D, E = A[dirset], A["d_en"]; cond = torch.load(COND, weights_only=False)
        self.m = CASTModel(model, {L: (D[L] / D[L].norm() * E[L].norm()).to(0, torch.bfloat16) for L in LAYERS}, condition=cond, alpha=alpha)
    def generate(self, **kw): return self.m.generate(**kw)
def get(name, model):
    p = name.split(":")
    if p[0] in ("base", "ft"): return Base(model)  # ft:<name> = fine-tuned checkpoint via MODEL_PATH, no hooks
    if p[0] == "steer": return Steer(model, p[1], float(p[2]))
    if p[0] == "cast": return Cast(model, p[1], float(p[2]))
    if p[0] == "cast2": return Cast(model, p[1], float(p[2]), "runs/cast_cond2.pt")
    raise ValueError(name)
