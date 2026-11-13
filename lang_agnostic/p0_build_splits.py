"""T0.2: seed-level train/dev/test/unseen splits. Child-safety rows dropped first."""
import json, random, re, collections as C
from pathlib import Path
R = Path(__file__).resolve().parents[1]
BAD = re.compile(r"minor|child|underage|csam", re.I)
def jl(p): return [json.loads(l) for l in open(p, encoding="utf8")]
def bad(*xs): return any(BAD.search(str(x)) for x in xs)
dropped = 0; rows = []
def add(r):
    global dropped
    if bad(r.get("harm"), " ".join(r["turns"]), r.get("_cats")): dropped += 1; return
    r.pop("_cats", None); rows.append(r)
def split3(keys, strat, seed=0):
    out = {}; g = C.defaultdict(list)
    for k in keys: g[strat(k)].append(k)
    for s, ks in sorted(g.items()):
        ks = sorted(ks); random.Random(seed).shuffle(ks); n = len(ks)
        a, b = round(n * .5), round(n * .7)
        for i, k in enumerate(ks): out[k] = "train" if i < a else "dev" if i < b else "test"
    return out
S = R / "results/mixed_language_screen"
judge = {r["id"]: r["judge"] for r in jl(S / "mixed_language_screen_judge.jsonl")}
raw = jl(S / "mixed_language_screen_raw.jsonl")
ext = jl(R / "circuit_safetune/extract_set.jsonl"); ext_seeds = {str(e["seed"]) for e in ext}
# IndicSafe by seed
isafe = [r for r in raw if r["dataset"] == "IndicSafe"]
seed_harm = {str(r["seed"]): r["harm"] for r in isafe}
sp = split3(seed_harm, lambda s: seed_harm[s])
for r in isafe:
    add(dict(id=r["id"], source="IndicSafe", split=sp[str(r["seed"])], language="Telugu", form=r["form"], seed=str(r["seed"]),
             harm=r["harm"], label="harmless" if r["harm"].startswith("Harmless Control") else "harmful", turns=r["turns"],
             multi_turn=len(r["turns"]) > 1, _cats=judge.get(r["id"])))
other = [r for r in raw if r["dataset"] != "IndicSafe"]
tel_single = [r for r in other if r["dataset"] == "IndicSafetyBench" and r["language"] == "Telugu" and r["form"] == "single-turn"]
sp_b = split3([r["id"] for r in tel_single], lambda k: next(x["family"] for x in tel_single if x["id"] == k))
csrt = [r for r in other if r["dataset"] == "CSRT"]
sp_c = split3([r["id"] for r in csrt], lambda k: "all")
for r in other:
    if r["dataset"] == "CSRT": s = sp_c[r["id"]]
    elif r["language"] != "Telugu": s = "unseen"
    elif r["form"] == "multi-turn": s = "test"
    else: s = sp_b[r["id"]]
    add(dict(id=r["id"], source=r["dataset"], split=s, language=r["language"], form=r["form"], seed=str(r.get("seed", "")),
             harm=r["harm"], label="harmful", turns=r["turns"], multi_turn=len(r["turns"]) > 1, _cats=judge.get(r["id"])))
# IndicGuard
IG = R / "results/indicguard/telugu_tiny_aya_fire"
roman = jl(R / "circuit_safetune/extract_roman.jsonl"); roman_ids = {r["id"] for r in roman}
for split, fname, n in [("train", "train", 400), ("dev", "valid", 150), ("test", "test", 300)]:
    d = [x for x in json.load(open(IG / f"{fname}.json", encoding="utf8")) if x.get("prompt") and x["id"] not in roman_ids
         and not bad(x.get("violated_categories"), x["prompt"])]
    for lab, want in [("unsafe", "harmful"), ("safe", "harmless")]:
        pool = sorted([x for x in d if x["prompt_label"] == lab], key=lambda x: x["id"]); random.Random(0).shuffle(pool)
        by = C.defaultdict(list)
        for x in pool: by[x["tag"]].append(x)
        take = []; tags = sorted(by)
        while len(take) < n and any(by.values()):
            for t in tags:
                if by[t] and len(take) < n: take.append(by[t].pop())
        for x in take:
            add(dict(id="ig:" + x["id"], source="IndicGuard", split=split, language="Telugu", form="telugu_script", seed="",
                     harm=x.get("violated_categories") or x["tag"], label=want, turns=[x["prompt"]], multi_turn=False))
for e in ext:
    if sp.get(str(e["seed"]), "train") != "train": continue
    add(dict(id="ext:" + str(e["seed"]) + ":" + e["label"], source="extract_set", split="train", language="English", form="english",
             seed=str(e["seed"]), harm=e["harm"], label=e["label"], turns=[e["prompt"]], multi_turn=False))
for e in roman:
    add(dict(id="roman:" + e["id"], source="extract_roman", split="train", language="Telugu", form="romanized_derived", seed="",
             harm=e.get("categories", ""), label=e["label"], turns=[e["prompt"]], multi_turn=False, _cats=e.get("categories")))
dups = [k for k, v in C.Counter(r["id"] for r in rows).items() if v > 1]; print("dups", len(dups), dups[:5]); _seen=set(); rows=[r for r in rows if not (r["id"] in _seen or _seen.add(r["id"]))]
seeds = C.defaultdict(set)
for r in rows:
    if r["source"] == "IndicSafe": seeds[r["seed"]].add(r["split"])
assert all(len(v) == 1 for v in seeds.values())
for s in ["train", "dev", "test", "unseen"]:
    with open(R / f"lang_agnostic/data/{s}.jsonl", "w", encoding="utf8") as f:
        for r in rows:
            if r["split"] == s: f.write(json.dumps(r, ensure_ascii=False) + "\n")
cnt = C.Counter((r["split"], r["source"], r["form"], r["label"]) for r in rows)
with open(R / "lang_agnostic/data/split_counts.md", "w") as f:
    f.write("| split | source | form | label | n |\n|---|---|---|---|---|\n")
    for k, v in sorted(cnt.items()): f.write("| " + " | ".join(k) + f" | {v} |\n")
print("child-safety dropped:", dropped); print(open(R / "lang_agnostic/data/split_counts.md").read())
