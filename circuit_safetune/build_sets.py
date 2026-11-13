"""Build the analysis sets from the mixed-language screen: matched failures, attack probes,
over-refusal controls, and a disjoint English extraction set. Child-safety rows are dropped."""
import json, random, collections as C, ast
SRC = "../results/mixed_language_screen/"
raw = {}
for l in open(SRC + "mixed_language_screen_raw.jsonl", encoding="utf8"):
    r = json.loads(l); raw[r["id"]] = r
J = {json.loads(l)["id"]: json.loads(l) for l in open(SRC + "mixed_language_screen_judge.jsonl", encoding="utf8")}
def turns(r):
    t = r["turns"]; return ast.literal_eval(t) if isinstance(t, str) else t
def unsafe(i): return str(J[i]["judge"].get("Response Safety", "")).lower() == "unsafe"
def minor(i): return "minor" in str(J[i]["judge"].get("Safety Categories", "")).lower() or "child" in str(raw[i].get("harm", "")).lower()
HARMLESS = lambda h: h.startswith("Harmless Control")
rows = [r for i, r in raw.items() if i in J and not minor(i) and len(turns(r)) == 1]
seeds = C.defaultdict(dict)
for r in rows:
    if r["dataset"] == "IndicSafe": seeds[r["seed"]][r["form"]] = r
out = []
def add(r, role, pair=None):
    out.append({"id": r["id"], "role": role, "pair": pair, "dataset": r["dataset"], "form": r["form"],
                "family": r["family"], "harm": r["harm"], "seed": r["seed"], "prompt": turns(r)[0],
                "screen_unsafe": unsafe(r["id"])})
flip_seeds = set()
for s, d in seeds.items():
    e = d.get("english")
    if not e or unsafe(e["id"]) or HARMLESS(e["harm"]): continue
    for f, r in d.items():
        if f != "english" and unsafe(r["id"]):
            add(r, "fail", s); flip_seeds.add(s)
    if s in flip_seeds: add(e, "fail_english_twin", s)
for r in rows:  # other single-turn attack probes the model failed
    if r["dataset"] != "IndicSafe" and unsafe(r["id"]): add(r, "attack")
harmless = sorted(s for s, d in seeds.items() if HARMLESS(next(iter(d.values()))["harm"]))
random.Random(0).shuffle(harmless)
h_eval, h_extract = set(harmless[: len(harmless) // 2]), set(harmless[len(harmless) // 2:])
for s in h_eval:
    for r in seeds[s].values(): add(r, "harmless_eval", s)
ext = []
for s, d in seeds.items():
    e = d.get("english")
    if not e or s in flip_seeds or s in h_eval: continue
    h = e["harm"]
    if h.startswith("Tricky"): continue
    ext.append({"label": "harmless" if HARMLESS(h) else "harmful", "harm": h, "seed": s, "prompt": turns(e)[0]})
with open("eval_set.jsonl", "w", encoding="utf8") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
with open("extract_set.jsonl", "w", encoding="utf8") as f:
    for o in ext: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(C.Counter(o["role"] for o in out), C.Counter(o["label"] for o in ext))
print("flip seeds", len(flip_seeds), "| attack by family", C.Counter(o["family"] for o in out if o["role"] == "attack"))
