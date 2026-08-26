#!/usr/bin/env python3
"""Grounded Q&A over HR policy extracts (retrieve then generate).

Same OpenRouter path as problems 10-12. Retrieval uses the Chroma helper
from problem 16 so policy chunks sit in a vector collection, not a keyword grep.

  export OPENROUTER_API_KEY=...
  python3 policy_rag_demo.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
POL = os.path.join(HERE, "policies")
P16 = os.path.join(HERE, "..", "problem_16_ticket_search")
sys.path.insert(0, P16)

import subprocess

try:
    import chromadb
    import sklearn  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb", "numpy", "scikit-learn"])
    import chromadb

from ticket_store import Embedder, load_env

SYSTEM = """You answer HR questions for employees.
Use only the retrieved policy excerpts below.
If they do not contain the answer, say you don't know and that the policy set does not cover it.
Do not guess from general HR knowledge.
Name the document id and section heading you used when you do answer."""


def chat(messages, key, model):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 350}).encode()
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


def load_policies():
    docs = []
    for name in sorted(os.listdir(POL)):
        if not name.endswith(".txt"):
            continue
        text = open(os.path.join(POL, name), encoding="utf-8").read()
        docs.append((name, text))
    return docs


def chunk_policy(name, text):
    chunks = []
    current = []
    section = "preamble"
    for line in text.splitlines():
        if line.startswith("Section "):
            if current:
                chunks.append((section, "\n".join(current).strip()))
            section = line.strip()
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append((section, "\n".join(current).strip()))
    out = []
    for i, (section, body) in enumerate(chunks):
        if len(body) < 40:
            continue
        out.append(
            {
                "id": f"{name}#{i}",
                "doc": name,
                "section": section,
                "text": body,
            }
        )
    return out


def ingest():
    import shutil

    persist = os.path.join(HERE, ".chroma")
    if os.path.isdir(persist):
        shutil.rmtree(persist)
    client = chromadb.PersistentClient(path=persist)
    col = client.get_or_create_collection(name="hr_policies", metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for name, text in load_policies():
        for ch in chunk_policy(name, text):
            ids.append(ch["id"])
            docs.append(ch["text"])
            metas.append({"document": ch["doc"], "section": ch["section"]})
    embedder = Embedder(HERE)
    embedder.fit(docs)
    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embedder(docs))
    return col


def retrieve(col, question, n=4):
    embedder = Embedder(HERE)
    res = col.query(query_embeddings=embedder([question]), n_results=n, include=["documents", "metadatas", "distances"])
    rows = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        rows.append({"text": doc, "meta": meta, "distance": float(dist)})
    return rows


def answer(col, question, key, model):
    rows = retrieve(col, question)
    ctx = []
    for r in rows:
        ctx.append(
            f"[{r['meta']['document']} / {r['meta']['section']}]\n{r['text']}"
        )
    block = "\n\n".join(ctx)
    print("retrieved:")
    for r in rows:
        print(f"  - {r['meta']['document']} | {r['meta']['section']} | dist={r['distance']:.3f}")
    text = chat(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Question: {question}\n\nPolicy excerpts:\n{block}",
            },
        ],
        key,
        model,
    )
    return text, rows


load_env()
key = os.environ.get("OPENROUTER_API_KEY", "").strip()
model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
if not key:
    sys.exit("set OPENROUTER_API_KEY")

col = ingest()
print("policy chunks in Chroma:", col.count())

in_scope = "How many vacation days do new hires get in their first year?"
out_scope = "Does the company offer pet insurance and a gym membership stipend?"

print("\n=== IN SCOPE ===")
print("Q:", in_scope)
a1, _ = answer(col, in_scope, key, model)
print("\nA:\n" + a1)

print("\n=== OUT OF SCOPE ===")
print("Q:", out_scope)
a2, _ = answer(col, out_scope, key, model)
print("\nA:\n" + a2)

low = a2.lower()
refuses = any(
    s in low
    for s in [
        "don't know",
        "do not know",
        "does not cover",
        "does not grant",
        "not cover",
        "not in the policy",
        "no policy",
        "not offer",
        "does not offer",
    ]
)
print("\n--- check ---")
print("in-scope cites a document:", "POL-HR-VAC" in a1 or "vacation" in a1.lower())
print("out-of-scope refuses instead of inventing:", refuses)
if not refuses:
    sys.exit("out-of-scope answer looks like a guess")
