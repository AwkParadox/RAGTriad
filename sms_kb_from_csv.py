import csv, re, html, pathlib, json, textwrap, random

CSV_FILE = pathlib.Path("analysisdataset.csv")   # your 600-row file
MIN_CHARS = 0                                    # ignore ultra-short rows
MAX_ROWS  = 10000                                # keep first N rows (or None)
TRAIN_RATIO = 0.9                                # % of rows used for knowledge base
RANDOM_SEED = 42                                 # reproducible split
TEST_OUTPUT = pathlib.Path("sms_test_messages.json")

# ------------------------------------------------------------------
# 1. Read CSV once
# ------------------------------------------------------------------
rows = []
with CSV_FILE.open(newline='', encoding='latin-1') as f:
    for r in csv.DictReader(f):
        txt = html.unescape(r.get("MainText") or r.get("Fulltext") or "")
        if txt and len(txt) >= MIN_CHARS:
            rows.append(txt.strip())
if MAX_ROWS:
    rows = rows[:MAX_ROWS]

if len(rows) < 2:
    raise RuntimeError("Need at least two rows to perform a 90/10 split.")

rng = random.Random(RANDOM_SEED)
rng.shuffle(rows)
split_idx = max(1, int(len(rows) * TRAIN_RATIO))
if split_idx >= len(rows):
    split_idx = len(rows) - 1

train_rows = rows[:split_idx]
test_rows = rows[split_idx:]

# ------------------------------------------------------------------
# 2. Turn every SMS into a “pattern” dictionary
# ------------------------------------------------------------------
phishing_patterns = []
for sms in train_rows:
    # very small automatic label (you can improve this later)
    if any(k in sms.lower() for k in ("urgent","suspend","locked","immediately")):
        pat = "Urgent action required"
    elif any(k in sms.lower() for k in ("won","reward","gift","prize")):
        pat = "Prize or reward claims"
    elif any(k in sms.lower() for k in ("bit.ly","tinyurl","short.link")):
        pat = "Suspicious links"
    else:
        pat = "General smishing attempt"

    phishing_patterns.append({
        "pattern": pat,
        "description": "Real-world SMS extracted from public smishing dataset.",
        "example": sms #textwrap.shorten(sms, width=120, placeholder="…")
    })

test_sms_messages = test_rows

# ------------------------------------------------------------------
# 3. Export the list under the exact name the RAG code imports
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(json.dumps(phishing_patterns, indent=2, ensure_ascii=False))

    OUT_FILE = pathlib.Path("sms_kb.json")   # any name / path you like
    OUT_FILE.write_text(
        json.dumps(phishing_patterns, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    TEST_OUTPUT.write_text(
        json.dumps(test_sms_messages, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✅  Wrote {len(phishing_patterns)} patterns → {OUT_FILE.resolve()}")
    print(f"✅  Wrote {len(test_sms_messages)} test SMS → {TEST_OUTPUT.resolve()}")