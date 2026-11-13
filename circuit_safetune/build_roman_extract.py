"""Romanized-Telugu extraction set: IndicGuard Telugu prompts (labelled safe/unsafe), transliterated with one
scheme for both sides so the direction cannot pick up writing style. No overlap with the eval datasets.
Child-safety rows are dropped. Needs `indic-transliteration` on PYTHONPATH."""
import json, random, re, sys, unicodedata
from indic_transliteration import sanscript
SRC = "../results/indicguard/telugu_tiny_aya_fire/"
rows = [r for s in ("train", "valid", "test") for r in json.load(open(f"{SRC}{s}.json", encoding="utf8"))]
def ok(r):
    p, cats = r.get("prompt") or "", str(r.get("violated_categories") or "")
    return p and 20 <= len(p) <= 300 and "minor" not in cats.lower() and re.search(r"[ఀ-౿]", p)
def romanize(t):  # IAST, then strip diacritics and lowercase: matches the screen's casual style ("nepal rajadhani emiti?")
    t = sanscript.transliterate(t, sanscript.TELUGU, sanscript.IAST).replace("\u200c", "").replace("\u200d", "")
    return "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c)).lower()
CORE = ("Violence", "Criminal Planning", "Guns", "Controlled", "Hate", "Harassment", "Suicide", "Fraud", "Malware", "Threat", "Illegal")
harm = [r for r in rows if ok(r) and r.get("prompt_label") == "unsafe" and any(c in str(r.get("violated_categories")) for c in CORE)]
safe = [r for r in rows if ok(r) and r.get("prompt_label") == "safe" and not r.get("violated_categories")]
rng = random.Random(0); harm, safe = rng.sample(harm, 128), rng.sample(safe, 128)
with open("extract_roman.jsonl", "w", encoding="utf8") as f:
    for lab, rs in (("harmful", harm), ("harmless", safe)):
        for r in rs:
            f.write(json.dumps({"label": lab, "id": r["id"], "categories": r.get("violated_categories"), "prompt": romanize(r["prompt"])}, ensure_ascii=False) + "\n")
print("harmful", len(harm), "harmless", len(safe))
for lab, rs in (("harmless", safe[:3]),): [print(lab, "|", romanize(r["prompt"])[:90]) for r in rs]
