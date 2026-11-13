import json, random
MODEL, REV = "CohereLabs/tiny-aya-fire", "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"
def jl(p): return [json.loads(l) for l in open(p, encoding="utf8")]
def extraction_prompts():
    ext = jl("extract_set.jsonl")
    harmful = [e["prompt"] for e in ext if e["label"] == "harmful"]
    harmless = [e["prompt"] for e in ext if e["label"] == "harmless"]
    from datasets import load_dataset
    a = load_dataset("tatsu-lab/alpaca", split="train").filter(lambda x: x["input"] == "")
    idx = random.Random(0).sample(range(len(a)), 128 - len(harmless))
    harmless += [a[i]["instruction"] for i in idx]
    rng = random.Random(1); rng.shuffle(harmful); rng.shuffle(harmless)
    return harmful, harmless

def roman_extraction_prompts():
    rows = jl("extract_roman.jsonl")
    return [r["prompt"] for r in rows if r["label"] == "harmful"], [r["prompt"] for r in rows if r["label"] == "harmless"]
