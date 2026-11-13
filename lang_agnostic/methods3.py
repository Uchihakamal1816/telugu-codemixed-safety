"""P3 methods (literature-inspired). Names:
 twostage:<gateL>:<alpha>  - A: upstream harm probe at layer gateL gates d_en steering at L15-18 (per row)
 formsteer:<alpha>         - D: script-routed direction (Telugu script -> d_te, Latin -> d_ro, mixed -> d_mix) at L15-18
 tli / tli_steer:<alpha>   - B: early-layer LoRA (L1-4) contrastively aligned to English; optionally + d_en steering
 path                      - C: LoRA on MLP15-18 only (the attributed pathway), refusal SFT + retain
 pivot / pivot_steer:<a>   - E: self-translate prompt to English, then answer (RomanSetu-style English bridge)"""
import re, torch, methods
LAYERS = [15, 16, 17, 18]
P3 = None
def p3():
    global P3
    if P3 is None: P3 = torch.load("runs/p3.pt", weights_only=False)
    return P3
def unit_en(d, L, alpha, E): return (d[L] / d[L].norm() * E[L].norm() * alpha)
class Hooked:
    """per-row steering: self.rowvec(L) -> (B,H) or None"""
    def __init__(self, model): self.model = model; self.hs = []
    def _add(self, L, fn):
        def hook(_m, _i, out):
            v = fn(); h = out[0] if isinstance(out, tuple) else out
            if v is None: return out
            h = h + v[:, None, :].to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        self.hs.append(self.model.model.layers[L].register_forward_hook(hook))
    def clear(self):
        for h in self.hs: h.remove()
        self.hs = []
class TwoStage(Hooked):
    def __init__(self, model, gL, alpha):
        super().__init__(model); A = torch.load("runs/dirs.pt"); E = A["d_en"]
        self.v = {L: unit_en(E, L, alpha, E).to(0, torch.bfloat16) for L in LAYERS}
        g = p3()["gates"][gL]; self.gL = gL; self.g = {k: (v.to(0) if torch.is_tensor(v) else v) for k, v in g.items()}
        self.fired = []
    def generate(self, **kw):
        state = {"gate": None}
        def gate_hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:  # prefill: decide from last prompt token
                x = (h[:, -1].float() - self.g["mu"]) / self.g["sd"]
                state["gate"] = ((x @ self.g["w"] + self.g["b"]) > self.g["thr"]).float()
                self.fired += state["gate"].tolist()
            return out
        self.hs.append(self.model.model.layers[self.gL].register_forward_hook(gate_hook))
        for L in LAYERS:
            self._add(L, lambda L=L: None if state["gate"] is None else state["gate"][:, None] * self.v[L][None])
        try: return self.model.generate(**kw)
        finally: self.clear()
TE = re.compile(r"[ఀ-౿]"); LAT = re.compile(r"[A-Za-z]")
EN_SW = set("the a an is are was were to of and or in on for with how what why who which can i you it this that do does be my your should".split())
def is_english(text):
    w = re.findall(r"[a-z']+", text.lower()); return bool(w) and sum(x in EN_SW for x in w) / len(w) >= 0.15
def route(text):
    t, l = len(TE.findall(text)), len(LAT.findall(text))
    if t == 0: return "romanized_derived" if l else "english"
    if l / (t + l) > 0.2: return "mixed_derived"
    return "telugu_script"
class FormSteer(Hooked):
    def __init__(self, model, tok, alpha):
        super().__init__(model); self.tok = tok; D = p3()["dirs"]; E = torch.load("runs/dirs.pt")["d_en"]
        # English/Latin text without Telugu -> d_ro would hit English too; English rows get d_en
        self.v = {f: {L: unit_en(D[f], L, alpha, E).to(0, torch.bfloat16) for L in LAYERS} for f in D}
        self.v["english"] = {L: unit_en(E, L, alpha, E).to(0, torch.bfloat16) for L in LAYERS}
        self.routes = []
    def generate(self, **kw):
        texts = self.tok.batch_decode(kw["input_ids"], skip_special_tokens=True)
        fs = [route(t) for t in texts]; self.routes += fs
        for L in LAYERS:
            V = torch.stack([self.v[f][L] for f in fs]); self._add(L, lambda V=V: V)
        try: return self.model.generate(**kw)
        finally: self.clear()
class Pivot:
    """rewrite(): translate each user turn to English with the same model, then answer it."""
    def __init__(self, model, tok, inner):
        self.model, self.tok, self.inner = model, tok, inner
    def rewrite(self, texts):
        if len(texts) > 8: return sum((self.rewrite(texts[i:i+8]) for i in range(0, len(texts), 8)), [])
        chats = [[{"role": "user", "content": "Translate the following text into English. Output only the translation, nothing else.\n\n" + t[:3000]}] for t in texts]
        s = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        enc = self.tok(s, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.no_grad(): o = self.model.generate(**enc, max_new_tokens=256, do_sample=False)
        return [x.strip() for x in self.tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    def generate(self, **kw): return self.inner.generate(**kw)
class PivotBack(Pivot):
    """v2: translate prompt to English, answer with steering, translate the answer back to the user's language."""
    def __init__(self, model, tok, inner):
        super().__init__(model, tok, inner); self.queue = []
    def rewrite(self, texts):
        en = Pivot.rewrite(self, texts)
        self.queue += list(texts)  # harness generates rows in exactly this order (single-turn chunks, then turn by turn)
        return en
    def generate(self, **kw):
        out = self.inner.generate(**kw); L = kw["input_ids"].shape[1]
        prompts = self.tok.batch_decode(kw["input_ids"], skip_special_tokens=False)
        ans = self.tok.batch_decode(out[:, L:], skip_special_tokens=True)
        res = []
        for p, a in zip(prompts, ans):
            o = self.queue.pop(0); f = route(o)
            other = TE.search(o) is None and len(LAT.findall(o)) < 0.5 * len(o.replace(" ", ""))
            if not other and TE.search(o) is None and is_english(o): res.append(a); continue
            lang = {"telugu_script": "Telugu, written in Telugu script", "romanized_derived": "Telugu, written in Latin letters (Romanized Telugu, not Hindi)",
                    "mixed_derived": "Telugu-English code-mixed, like the user's message"}.get(f)
            if other:  # other Indic scripts (unseen languages): name the script, never show the user text (it gets echoed)
                import unicodedata
                sc = [unicodedata.name(ch, "").split(" ")[0] for ch in o if ord(ch) > 0x900 and unicodedata.name(ch, "")]
                sc = max(set(sc), key=sc.count).title() if sc else "the user's"
                sc = {"Arabic": "Urdu (Arabic script)", "Gurmukhi": "Punjabi (Gurmukhi script)", "Oriya": "Odia", "Meetei": "Manipuri (Meetei Mayek script)",
                      "Devanagari": "the Indian language the user wrote in, using Devanagari script", "Bengali": "Bengali-script language the user wrote in"}.get(sc, sc + " language")
                lang = sc + ". Do not use Telugu or English"
            c = [{"role": "user", "content": "Translate the assistant reply below into " + lang + ". Translate only the reply. Output only the translated reply.\n\nAssistant reply:\n" + a}]
            e = self.tok(self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to(0)
            with torch.no_grad(): t = self.model.generate(**e, max_new_tokens=400, do_sample=False)
            res.append(self.tok.decode(t[0, e["input_ids"].shape[1]:], skip_special_tokens=True))
        enc = self.tok(res, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to(0)
        pad = torch.full((enc.shape[0], L), self.tok.pad_token_id, device=0)
        return torch.cat([pad, enc], 1)

# ---------------- P4 ----------------
class SWE:
    """M1: closed-form down_proj edit (runs/p4_swe.pt) scaled by beta."""
    def __init__(self, model, beta):
        for L, dW in torch.load("runs/p4_swe.pt")["dW"].items():
            W = model.model.layers[L].mlp.down_proj.weight; W.data += (beta * dW.float()).to(W.dtype).to(W.device)
        self.model = model
    def generate(self, **kw): return self.model.generate(**kw)
class XGate(Hooked):
    """M2: self-translate to English for the decision only; L18 probe on the translation gates strong d_en steering of the ORIGINAL prompt."""
    def __init__(self, model, tok, alpha):
        super().__init__(model); self.tok = tok; self.tr = Pivot(model, tok, None); self.queue = []; self.fired = []
        E = torch.load("runs/dirs.pt")["d_en"]; self.v = {L: unit_en(E, L, alpha, E).to(0, torch.bfloat16) for L in LAYERS}
        g = p3()["gates"][18]; self.g = {k: (v.to(0) if torch.is_tensor(v) else v) for k, v in g.items()}
    def score(self, texts):
        s = [self.tok.apply_chat_template([{"role": "user", "content": t[:6000]}], tokenize=False, add_generation_prompt=True) for t in texts]
        e = self.tok(s, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        with torch.no_grad(): h = self.model(**e, output_hidden_states=True, logits_to_keep=1).hidden_states[19][:, -1].float()
        return (((h - self.g["mu"]) / self.g["sd"]) @ self.g["w"] + self.g["b"] > self.g["thr"]).float().tolist()
    def rewrite(self, texts):
        en = self.tr.rewrite(texts); gs = []
        for i in range(0, len(en), 8): gs += self.score(en[i:i+8])
        self.queue += gs; self.fired += gs; return texts
    def generate(self, **kw):
        B = kw["input_ids"].shape[0]; gt = torch.tensor([self.queue.pop(0) for _ in range(B)], device=0)
        for L in LAYERS:
            V = gt[:, None].to(torch.bfloat16) * self.v[L][None]; self._add(L, lambda V=V: V)
        try: return self.model.generate(**kw)
        finally: self.clear()
class Translit:
    """M3: Latin-script non-English input -> Telugu script (by the model), then d_en steering."""
    def __init__(self, model, tok, alpha):
        self.model, self.tok = model, tok; self.inner = methods.Steer(model, "d_en", alpha); self.n = 0
    def rewrite(self, texts):
        idx = [i for i, t in enumerate(texts) if TE.search(t) is None and len(LAT.findall(t)) > 0.5 * len(t.replace(" ", "")) and not is_english(t)]
        out = list(texts)
        for j in range(0, len(idx), 8):
            ch = idx[j:j+8]
            s = [self.tok.apply_chat_template([{"role": "user", "content": "Rewrite this Romanized Telugu text in Telugu script. Do not translate it and do not answer it; only change the script. Output only the rewritten text.\n\n" + texts[i][:2000]}], tokenize=False, add_generation_prompt=True) for i in ch]
            e = self.tok(s, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
            with torch.no_grad(): o = self.model.generate(**e, max_new_tokens=300, do_sample=False)
            for i, t in zip(ch, self.tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)): out[i] = t.strip() or texts[i]
        self.n += len(idx); return out
    def generate(self, **kw): return self.inner.generate(**kw)
def load_lora(model, path):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, path).merge_and_unload()
def get(name, model, tok):
    p = name.split(":")
    if p[0] == "twostage": return TwoStage(model, int(p[1]), float(p[2]))
    if p[0] == "formsteer": return FormSteer(model, tok, float(p[1]))
    if p[0] == "tli": return methods.Base(load_lora(model, "runs/p3_tli_lora"))
    if p[0] == "tli_steer": return methods.Steer(load_lora(model, "runs/p3_tli_lora"), "d_en", float(p[1]))
    if p[0] == "path": return methods.Base(load_lora(model, "runs/p3_path_lora"))
    if p[0] == "pivot": return Pivot(model, tok, methods.Base(model))
    if p[0] == "swe": return SWE(model, float(p[1]))
    if p[0] == "xgate": return XGate(model, tok, float(p[1]))
    if p[0] == "translit": return Translit(model, tok, float(p[1]))
    if p[0] == "dpo": return methods.Base(load_lora(model, "runs/p4_dpo_lora"))
    if p[0] == "nlora": return methods.Base(load_lora(model, "runs/p4_nlora_lora"))
    if p[0] == "path2": return methods.Base(load_lora(model, "runs/p3_path2_lora"))
    if p[0] == "pivotback_steer": return PivotBack(model, tok, methods.Steer(model, "d_en", float(p[1])))
    if p[0] == "pivot_steer": return Pivot(model, tok, methods.Steer(model, "d_en", float(p[1])))
    if p[0] in ("spangate", "delgate", "delspan"):
        import methods5; return methods5.get(name, model, tok)
    return methods.get(name, model)
