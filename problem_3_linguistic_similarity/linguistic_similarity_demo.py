#!/usr/bin/env python3
"""Compare Jaccard overlap vs spaCy document vectors on 10 sentence pairs.

Use en_core_web_md — the small model has no real vectors.
A pair is a 'trap caught' if abs(vector - jaccard) >= 0.30.
"""

import importlib.util
import subprocess
import sys

MODEL = "en_core_web_md"
TRAP = 0.30


def install(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg])


def load_nlp():
    import spacy

    try:
        nlp = spacy.load(MODEL)
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", MODEL])
        nlp = spacy.load(MODEL)
    if nlp.vocab.vectors.size == 0:
        sys.exit(f"{MODEL} has no vectors. Don't use en_core_web_sm for this.")
    return nlp


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def root_token(doc):
    for t in doc:
        if t.dep_ == "ROOT":
            return f"{t.text} ({t.pos_})"
    return "?"


def parse_line(doc):
    return "  ".join(f"{t.text}:{t.dep_}->{t.head.text}" for t in doc if not t.is_space)


install("spacy", "spacy==3.7.5")  # 3.8+ needs python 3.10
nlp = load_nlp()
print(f"loaded {MODEL}, vectors={nlp.vocab.vectors.size}")

# 1-3 are the deliberate traps (same words / different meaning, or paraphrase)
pairs = [
    ("trap-lexical", "The risk agent approved the aggressive trade.", "The risk agent rejected the aggressive trade."),
    ("trap-lexical", "The retrieval agent is ready for production.", "The retrieval agent is not ready for production."),
    ("trap-paraphrase", "The vector database stores embeddings for retrieval.", "Our search index keeps dense representations so similar policy chunks can be found."),
    ("agree-similar", "Agents plan act and observe in a loop.", "Agents plan, act, and observe using a control loop."),
    ("agree-similar", "Embeddings help retrieve similar documents from storage.", "Retrieval uses embeddings to find related documents."),
    ("agree-different", "Conservative clients prefer low risk investment profiles.", "The vector database indexes chunks for fast retrieval search."),
    ("agree-different", "Guardrails block dangerous tool calls in production.", "Moderate risk sits between conservative and aggressive."),
    ("agree-similar", "Production systems must log every tool call.", "Production guardrails reject unsafe API tool calls."),
    ("agree-similar", "The system acts only after it observes new state.", "Observing state in the loop helps decide the next action."),
    ("trap-lexical", "The desk recommended a conservative profile for the client.", "The desk recommended an aggressive profile for the client."),
]

rows = []
for i, (kind, a, b) in enumerate(pairs, 1):
    da, db = nlp(a), nlp(b)
    lex = jaccard({t.text.lower() for t in da if t.is_alpha}, {t.text.lower() for t in db if t.is_alpha})
    vec = da.similarity(db)
    gap = abs(vec - lex)
    rows.append((gap, i, kind, a, b, lex, vec, root_token(da), root_token(db), parse_line(da), parse_line(db)))
    print(f"\nPair {i} [{kind}]")
    print("  A ROOT:", root_token(da))
    print("  A", parse_line(da))
    print("  B ROOT:", root_token(db))
    print("  B", parse_line(db))

rows.sort(reverse=True)

print("\n#  jaccard  vector   gap    trap?  kind")
print("-" * 70)
for gap, i, kind, a, b, lex, vec, *_ in rows:
    flag = "YES" if gap >= TRAP else "no"
    print(f"{i:<3} {lex:.3f}    {vec:.3f}   {gap:.3f}  {flag:<5} {kind}")
    print(f"    A: {a}")
    print(f"    B: {b}")

print(f"\ntrap caught if |vector - jaccard| >= {TRAP:.2f}")
top = rows[:2]
print(
    f"\nBiggest disagreement: pair {top[0][1]} (gap {top[0][0]:.3f}) "
    f"and pair {top[1][1]} (gap {top[1][0]:.3f})."
)
print(
    "Jaccard drops to 0 as soon as the wording changes, even on paraphrases. "
    "Averaged spaCy vectors stay high on one-word swaps like approved/rejected."
)
