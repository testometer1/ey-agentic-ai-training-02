#!/usr/bin/env python3
"""Relationship-manager client review from one month of activity (OpenRouter).

Uses the same wealth-desk client as earlier problems (Sarah Chen / RSK-PROF-C01).

  export OPENROUTER_API_KEY=...
  python3 rm_review_demo.py
  python3 rm_review_demo.py july_activity.csv
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "july_activity.csv")

SYSTEM = """You are a relationship manager writing to the named client.
Write exactly two paragraphs. No bullets, no heading, no signature block.
Use only figures, dates, product names, and events that appear in the activity extract.
Do not recommend trades, promise returns, or offer products.
If the extract is incomplete, say what is missing instead of filling gaps.
Address the client by name. Stay factual about sensitive items (large debit, fees, missed payment, risk profile)."""


def load_env():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(os.getcwd(), ".env"), os.path.join(here, ".env"), os.path.join(here, "..", ".env")]:
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def chat(messages, key, model):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.15, "max_tokens": 450}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "livedemo",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"network error: {e}")
    return data["choices"][0]["message"]["content"].strip()


def source_tokens(text):
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    nums = {n.replace(",", "") for n in nums}
    codes = set(re.findall(r"[A-Z]{2,}(?:-[A-Z0-9]+)+", text))
    names = set(re.findall(r"\b(?:Sarah Chen|Priya Nair|Vanguard|VTI|First National Private Bank|Chen Advisory LLP)\b", text, re.I))
    events = []
    for pat, label in [
        (r"missed[- ]payment", "missed payment"),
        (r"margin-loan|margin loan", "margin-loan interest"),
        (r"custody fee", "custody fee"),
        (r"wire handling fee", "wire handling fee"),
        (r"outgoing wire|house-deposit|HOUSE-DEP", "outgoing house-deposit wire"),
        (r"risk profile", "risk profile review"),
        (r"payroll", "payroll credit"),
        (r"dividend", "dividend"),
        (r"cash sweep", "cash sweep interest"),
    ]:
        if re.search(pat, text, re.I):
            events.append(label)
    return nums, codes, names, events


def facts_in_draft(draft):
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", draft)
    codes = re.findall(r"[A-Z]{2,}(?:-[A-Z0-9]+)+", draft)
    names = re.findall(
        r"\b(?:Sarah Chen|Priya Nair|Vanguard|VTI|First National Private Bank|Chen Advisory LLP|"
        r"July|August|June|USD)\b",
        draft,
        re.I,
    )
    events = []
    for pat, label in [
        (r"missed[- ]payment|not collected|insufficient cash", "missed payment"),
        (r"margin[- ]loan", "margin-loan interest"),
        (r"custody", "custody fee"),
        (r"wire (handling )?fee|handling fee", "wire handling fee"),
        (r"wire|HOUSE-DEP|house[- ]deposit", "outgoing house-deposit wire"),
        (r"risk profile|conservative|RSK-PROF", "risk profile review"),
        (r"payroll", "payroll credit"),
        (r"dividend", "dividend"),
        (r"sweep", "cash sweep interest"),
        (r"property tax", "property tax"),
        (r"sell|sold|shares", "VTI sale"),
    ]:
        if re.search(pat, draft, re.I):
            events.append(label)
    items = []
    for n in nums:
        items.append(("number", n.replace(",", "")))
    for c in codes:
        items.append(("code", c))
    for n in names:
        items.append(("name", n))
    for e in events:
        items.append(("event", e))
    return items


def in_source(kind, value, nums, codes, names, events, source):
    src = source.lower()
    if kind == "number":
        if value in nums:
            return True
        # allow 185000 written as 185,000 already stripped
        return value in src.replace(",", "")
    if kind == "code":
        return value in codes or value in source
    if kind == "name":
        return value.lower() in src
    if kind == "event":
        return value in events or value.lower() in src
    return False


load_env()
key = os.environ.get("OPENROUTER_API_KEY", "").strip()
model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
if not key:
    sys.exit("set OPENROUTER_API_KEY (or put it in a .env file)")

path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
path = os.path.abspath(path)
source = open(path, encoding="utf-8").read()
print("source:", path)
print("\n--- activity extract ---\n" + source)
print(f"\ncalling {model} ...")

draft = chat(
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Write the two-paragraph July review from this extract:\n\n" + source},
    ],
    key,
    model,
)

paras = [p.strip() for p in re.split(r"\n\s*\n", draft) if p.strip()]
print("\n--- two-paragraph review ---\n" + draft)
print(f"\nparagraph count: {len(paras)} (need 2)")

nums, codes, names, events = source_tokens(source)
print("\n--- fact check ---")
ok = True
seen = set()
for kind, value in facts_in_draft(draft):
    keyf = (kind, value.lower() if kind != "number" else value)
    if keyf in seen:
        continue
    seen.add(keyf)
    # skip generic calendar words that appear in period
    if kind == "name" and value.lower() in {"july", "usd"}:
        mark = "in source"
    else:
        hit = in_source(kind, value, nums, codes, names, events, source)
        mark = "in source" if hit else "NOT in source"
        if not hit:
            ok = False
    print(f"  [{kind}] {value}: {mark}")

if len(paras) != 2:
    ok = False
    print("  structure: NOT two paragraphs")

print("\n--- send? ---")
if ok:
    print("I would send this draft as-is: every checked figure and named event is in the extract, and it is two paragraphs.")
else:
    print("I would not send this draft as-is: edit or regenerate until every number and named event is in the source and the body is exactly two paragraphs.")
