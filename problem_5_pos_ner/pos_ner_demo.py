#!/usr/bin/env python3
"""POS counts + NER on 5 domain samples, then a gold vs predicted check.

Product codes are supposed to fail on the default model — that's the point.
Accuracy is counted per entity, not per sample.
"""

import importlib.util
import subprocess
import sys
from collections import Counter

MODEL = "en_core_web_md"


def install(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg])


def load_nlp():
    import spacy

    try:
        return spacy.load(MODEL)
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", MODEL])
        return spacy.load(MODEL)


def match(gold, pred):
    leftover = list(gold)
    invented = []
    for p in pred:
        hit = None
        pt, pl = p["text"].lower().strip(), p["label"].upper()
        for i, g in enumerate(leftover):
            gt, gl = g["text"].lower().strip(), g["label"].upper()
            overlap = pt == gt or pt in gt or gt in pt
            if overlap and pl == gl:
                hit = i
                break
        if hit is None:
            invented.append(p)
        else:
            leftover.pop(hit)
    return leftover, invented


install("spacy", "spacy==3.7.5")
nlp = load_nlp()

samples = [
    {
        "title": "London ops incident",
        "text": (
            "On 18 August 2026 Sarah Chen from the London operations desk stopped a live agent after it entered a tight "
            "tool-call loop. Ticket AGNT-LOOP-42 was opened at 09:15 in Splunk. The planner asked the executor to retry a "
            "failed query without observing the new error. Logs showed forty repeated calls against the billing service at "
            "Amazon. Production question answering for Goldman Sachs conservative-profile clients was blocked for forty "
            "seven minutes and the estimated operations cost was £47,000. Guardrails now block more than three identical "
            "tool calls in one loop. The desk will monitor the run for twenty four hours before returning the agent to "
            "full production traffic in London and the backup region."
        ),
        "gold": [
            {"text": "18 August 2026", "label": "DATE"},
            {"text": "Sarah Chen", "label": "PERSON"},
            {"text": "London", "label": "GPE"},
            {"text": "AGNT-LOOP-42", "label": "PRODUCT"},
            {"text": "Splunk", "label": "ORG"},
            {"text": "Amazon", "label": "ORG"},
            {"text": "Goldman Sachs", "label": "ORG"},
            {"text": "£47,000", "label": "MONEY"},
        ],
    },
    {
        "title": "retrieval note",
        "text": (
            "Engineers in San Francisco rebuilt the OpenAI embedding pipeline after retrieval quality dropped on European "
            "policy chunks. The Pinecone vector database in California stores those embeddings under index EMB-IDX-EU-9. "
            "Microsoft Azure hosts the search failover for clients in Paris and Dublin. If retrieval fails, the model "
            "hallucinates account numbers instead of citing a chunk. Cosine similarity ranking was restored on 12 July 2026 "
            "after a stale key was rotated. The rebuild cost USD 150,000 and took thirty six hours. RAG pipelines now "
            "retrieve context before the answer is written, and the San Francisco team owns on-call for the index "
            "during European business hours and the first overnight failover test."
        ),
        "gold": [
            {"text": "San Francisco", "label": "GPE"},
            {"text": "OpenAI", "label": "ORG"},
            {"text": "Pinecone", "label": "ORG"},
            {"text": "California", "label": "GPE"},
            {"text": "EMB-IDX-EU-9", "label": "PRODUCT"},
            {"text": "Microsoft Azure", "label": "ORG"},
            {"text": "Paris", "label": "GPE"},
            {"text": "Dublin", "label": "GPE"},
            {"text": "12 July 2026", "label": "DATE"},
            {"text": "USD 150,000", "label": "MONEY"},
        ],
    },
    {
        "title": "NY risk memo",
        "text": (
            "The risk desk in New York reviewed conservative, moderate, and aggressive profiles for Goldman Sachs clients "
            "on Monday morning. Product code RSK-PROF-C01 still maps a conservative investor to low leverage. The committee "
            "never approved high leverage for new accounts after a failed recommendation of $2.4 million in notional. "
            "EY partners in New York asked for a written explanation of each profile score. Sentiment on the trading floor "
            "was negative, but the moderate profile remains useful for clients who sit between cautious and aggressive. "
            "The next review is scheduled for 1 September 2026 in the New York office on Madison Avenue after the close."
        ),
        "gold": [
            {"text": "New York", "label": "GPE"},
            {"text": "Goldman Sachs", "label": "ORG"},
            {"text": "Monday", "label": "DATE"},
            {"text": "RSK-PROF-C01", "label": "PRODUCT"},
            {"text": "$2.4 million", "label": "MONEY"},
            {"text": "EY", "label": "ORG"},
            {"text": "1 September 2026", "label": "DATE"},
            {"text": "Madison Avenue", "label": "FAC"},
        ],
    },
    {
        "title": "guardrail change",
        "text": (
            "Change TOOL-GRD-3 was deployed to production in London at 22:00 on 18 August 2026. The executor now logs every "
            "tool call and rejects unsafe retries against the billing API. Amazon CloudWatch alarms fire when more than "
            "three identical calls occur in one loop. The estimated avoided loss is $80,000 per incident. Operators in "
            "Singapore and London confirmed the new rule during a tabletop exercise. Documentation lives in Confluence and "
            "the on-call rota is owned by the production reliability team. No customer funds were moved and no live trades "
            "were placed during the rollout window. The desk will review the alarm noise after seven days of production use."
        ),
        "gold": [
            {"text": "TOOL-GRD-3", "label": "PRODUCT"},
            {"text": "London", "label": "GPE"},
            {"text": "18 August 2026", "label": "DATE"},
            {"text": "Amazon CloudWatch", "label": "ORG"},
            {"text": "$80,000", "label": "MONEY"},
            {"text": "Singapore", "label": "GPE"},
            {"text": "Confluence", "label": "ORG"},
        ],
    },
    {
        "title": "follow-up",
        "text": (
            "A joint note from London and New York on 20 August 2026 asked Pinecone and OpenAI support to confirm that "
            "index EMB-IDX-EU-9 had been rebuilt. Goldman Sachs still wants conservative routing only. The advisory desk "
            "paused a $1.1 million aggressive recommendation until retrieval quality is scored above the internal bar. "
            "Sarah Chen will present findings in Chicago on 3 September 2026. Ticket AGNT-LOOP-42 remains linked to change "
            "TOOL-GRD-3. The legal review sits with EY in New York. Operators must not skip the observe step before the "
            "next production action, even when the planner is confident. The follow-up call is booked for Friday afternoon."
        ),
        "gold": [
            {"text": "London", "label": "GPE"},
            {"text": "New York", "label": "GPE"},
            {"text": "20 August 2026", "label": "DATE"},
            {"text": "Pinecone", "label": "ORG"},
            {"text": "OpenAI", "label": "ORG"},
            {"text": "EMB-IDX-EU-9", "label": "PRODUCT"},
            {"text": "Goldman Sachs", "label": "ORG"},
            {"text": "$1.1 million", "label": "MONEY"},
            {"text": "Sarah Chen", "label": "PERSON"},
            {"text": "Chicago", "label": "GPE"},
            {"text": "3 September 2026", "label": "DATE"},
            {"text": "AGNT-LOOP-42", "label": "PRODUCT"},
            {"text": "TOOL-GRD-3", "label": "PRODUCT"},
            {"text": "EY", "label": "ORG"},
        ],
    },
]

missed_all = []
invented_all = []
gold_n = Counter()
miss_n = Counter()

print(f"{'sample':<28} words  NOUN VERB ADJ  PROPN")
print("-" * 55)

for i, s in enumerate(samples, 1):
    n = len(s["text"].split())
    doc = nlp(s["text"])
    pos = Counter(t.pos_ for t in doc if not t.is_space)
    print(f"{i}. {s['title']:<25} {n:<6} {pos.get('NOUN',0):<4} {pos.get('VERB',0):<4} {pos.get('ADJ',0):<4} {pos.get('PROPN',0)}")

    pred = [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents]
    print(f"  entities ({len(pred)}):")
    for e in pred:
        print(f"    {e['text']!r:35} {e['label']:<8} {e['start']}-{e['end']}")

    missed, invented = match(s["gold"], pred)
    for g in s["gold"]:
        gold_n[g["label"]] += 1
    for g in missed:
        miss_n[g["label"]] += 1
        missed_all.append((i, g))
    for p in invented:
        invented_all.append((i, p))

print("\nmissed (gold, not predicted / wrong label):")
for i, g in missed_all:
    print(f"  sample {i}: {g['text']!r} [{g['label']}]")

print("\ninvented (predicted, not in gold):")
for i, p in invented_all:
    print(f"  sample {i}: {p['text']!r} [{p['label']}]")

print("\nmiss rate by gold type:")
worst, worst_rate = None, -1
for lab, n in sorted(gold_n.items()):
    m = miss_n.get(lab, 0)
    rate = m / n
    print(f"  {lab:<10} gold={n} missed={m} {rate:.0%}")
    if rate > worst_rate:
        worst, worst_rate = lab, rate

print(f"\n{worst} is worst on this text ({worst_rate:.0%} miss) — internal product codes aren't in the default NER model.")
