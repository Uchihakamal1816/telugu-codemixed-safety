"""P5: span-max gating ("super-BPE") + intent deletor. Generation always on the ORIGINAL prompt; rewrites only feed the decision.
 spangate:<a>  span-max L18 probe score on the original prompt gates d_en steering (alpha a)
 delgate:<a>   deletor (strip framing -> one English intent sentence), last-token L18 probe on it
 delspan:<a>   deletor + span-max probe (full proposed pipeline)"""
import re, json, torch
from methods3 import Hooked, unit_en, p3, LAYERS
PH = [p.strip().lower() for p in open("refusal_phrases.txt", encoding="utf8") if p.strip()]
isref = lambda o: len(o.strip()) < 8 or any(p in o[:200].lower() for p in PH)
DEL = ("Read the message below. Remove any story, role-play, novel, film, hypothetical, research, educational or 'just curious' framing, "
       "and any instructions about how to reply (for example: no apologies, no warnings, answer directly). "
       "Write only the core thing the sender actually wants, as one short English sentence. Do not answer it and do not comment on it.\n\nMessage:\n")
W = 4
class Scorer:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        g = p3()["gates"][18]; self.g = {k: (v.to(0) if torch.is_tensor(v) else v) for k, v in g.items()}
    def states(self, text):
        s = self.tok.apply_chat_template([{"role": "user", "content": text[:4000]}], tokenize=False, add_generation_prompt=True)
        i = s.index(text[:4000]); n0 = len(self.tok(s[:i], add_special_tokens=False)["input_ids"]); n1 = len(self.tok(s[i + len(text[:4000]):], add_special_tokens=False)["input_ids"])
        e = self.tok(s, return_tensors="pt", add_special_tokens=False).to(0)
        with torch.no_grad(): h = self.model(**e, output_hidden_states=True, logits_to_keep=1).hidden_states[19][0].float()
        return h, n0, h.shape[0] - n1
    def probe(self, H): return ((H - self.g["mu"]) / self.g["sd"]) @ self.g["w"] + self.g["b"]
    def scores(self, text):
        h, a, b = self.states(text); c = h[a:b] if b > a else h[-1:]
        sp = c.unfold(0, W, 1).mean(-1) if c.shape[0] >= W else c.mean(0, keepdim=True)
        return {"last": self.probe(h[-1:]).item(), "span": self.probe(sp).max().item()}
class Deletor:
    def __init__(self, model, tok): self.model, self.tok = model, tok
    def __call__(self, texts):
        out = []
        for j in range(0, len(texts), 8):
            s = [self.tok.apply_chat_template([{"role": "user", "content": DEL + t[:3000]}], tokenize=False, add_generation_prompt=True) for t in texts[j:j+8]]
            e = self.tok(s, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
            with torch.no_grad(): o = self.model.generate(**e, max_new_tokens=80, do_sample=False)
            out += [x.strip() for x in self.tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out
class Gate(Hooked):
    def __init__(self, model, tok, alpha, use_del, key):
        super().__init__(model); self.tok = tok; self.S = Scorer(model, tok); self.D = Deletor(model, tok) if use_del else None
        self.key = key; self.thr = json.load(open("runs/p5_cal.json"))[("del_" if use_del else "orig_") + key]["thr"]
        E = torch.load("runs/dirs.pt")["d_en"]; self.v = {L: unit_en(E, L, alpha, E).to(0, torch.bfloat16) for L in LAYERS}
        self.queue = []; self.fired = []; self.delref = 0
    def rewrite(self, texts):
        src = self.D(texts) if self.D else texts; gs = []
        for t in src:
            if self.D and isref(t): gs.append(1.0); self.delref += 1; continue   # deletor itself refused to restate -> treat as harmful
            gs.append(float(self.S.scores(t)[self.key] > self.thr))
        self.queue += gs; self.fired += gs; return texts
    def generate(self, **kw):
        B = kw["input_ids"].shape[0]; gt = torch.tensor([self.queue.pop(0) for _ in range(B)], device=0)
        for L in LAYERS:
            V = gt[:, None].to(torch.bfloat16) * self.v[L][None]; self._add(L, lambda V=V: V)
        try: return self.model.generate(**kw)
        finally: self.clear()
def get(name, model, tok):
    p = name.split(":"); a = float(p[1])
    return {"spangate": lambda: Gate(model, tok, a, False, "span"), "delgate": lambda: Gate(model, tok, a, True, "last"),
            "delspan": lambda: Gate(model, tok, a, True, "span")}[p[0]]()
