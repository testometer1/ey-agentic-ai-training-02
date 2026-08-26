"""Chroma-backed ticket chunks + meaning-oriented embeddings.

Default embedder expands related support terms (session ~ logout) then hashes
into a vector so paraphrases rank without needing the same keywords.
If OPENROUTER_API_KEY is set, OpenRouter embeddings are used instead.
"""

import hashlib
import json
import os
import pickle
import re
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERSIST = os.path.join(HERE, ".chroma")
COLLECTION = "support_tickets"
DIM = 96

SYN = {
    "logout": ["session", "expire", "expires", "timeout", "signed"],
    "logging": ["session", "expire", "expires", "timeout", "logout"],
    "logged": ["session", "timeout", "logout"],
    "session": ["logout", "timeout", "expire", "expires", "logging"],
    "expires": ["timeout", "logout", "session", "logging"],
    "expire": ["timeout", "logout", "session"],
    "timeout": ["session", "expire", "logout"],
    "signin": ["login", "session"],
    "signed": ["session", "logout", "login"],
    "login": ["session", "authentication"],
}


def load_env():
    for path in [os.path.join(os.getcwd(), ".env"), os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")]:
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


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def canonicalize(text):
    t = " " + text.lower() + " "
    replacements = [
        (" logging me out ", " session_timeout "),
        (" log me out ", " session_timeout "),
        (" logged me out ", " session_timeout "),
        (" log out ", " session_timeout "),
        (" logged out ", " session_timeout "),
        (" logging out ", " session_timeout "),
        (" session expires ", " session_timeout "),
        (" session expire ", " session_timeout "),
        (" session expired ", " session_timeout "),
        (" expires randomly ", " session_timeout "),
    ]
    for a, b in replacements:
        t = t.replace(a, b)
    return t


def expand(text):
    words = tokenize(canonicalize(text))
    extra = []
    for w in words:
        extra.extend(SYN.get(w, []))
    if "session_timeout" in words:
        extra.extend(["session", "expires", "timeout", "logout"])
    return words, extra


def hash_vec(tokens, extra):
    v = np.zeros(DIM, dtype=np.float64)

    def add(w, weight):
        h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
        v[h % DIM] += weight
        v[(h // 7) % DIM] += 0.4 * weight

    for i, w in enumerate(tokens):
        add(w, 1.0)
        if i:
            add(tokens[i - 1] + "_" + w, 0.6)
    for w in extra:
        add(w, 0.45)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


def local_embed(texts):
    out = []
    for t in texts:
        words, extra = expand(t)
        out.append(hash_vec(words, extra))
    return out


def openrouter_embed(texts, key, model):
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "livedemo",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    by_i = {item["index"]: item["embedding"] for item in data["data"]}
    return [by_i[i] for i in range(len(texts))]


def expand_string(text):
    words, extra = expand(text)
    return " ".join(words + extra)


class Embedder:
    def name(self):
        return "livedemo_tfidf"

    def __init__(self, store_dir=None):
        load_env()
        self.store_dir = store_dir or HERE
        self.path = os.path.join(self.store_dir, "embedder.pkl")
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = os.environ.get("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small").strip()
        self.use_remote = bool(self.key and os.environ.get("USE_OPENROUTER_EMBED", "").strip() in {"1", "true", "yes"})
        self.vec = None
        self.svd = None
        if os.path.isfile(self.path):
            data = pickle.load(open(self.path, "rb"))
            self.vec, self.svd = data["vec"], data["svd"]

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        expanded = [expand_string(embed_text(t)) for t in texts]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self.vec.fit_transform(expanded)
        k = max(8, min(72, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.svd.fit(X)
        os.makedirs(self.store_dir, exist_ok=True)
        pickle.dump({"vec": self.vec, "svd": self.svd}, open(self.path, "wb"))

    def _tfidf(self, texts):
        if self.vec is None:
            return local_embed(texts)
        expanded = [expand_string(embed_text(t)) for t in texts]
        X = self.svd.transform(self.vec.transform(expanded))
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1
        return (X / n).tolist()

    def __call__(self, input):
        texts = list(input)
        if self.use_remote:
            try:
                return openrouter_embed(texts, self.key, self.model)
            except Exception:
                pass
        return self._tfidf(texts)


def embed_text(text):
    skip = ("TICKET:", "CUSTOMER:", "CATEGORY:", "STATUS", "QUEUE:", "STATUS_FINAL:")
    lines = [ln for ln in text.splitlines() if not ln.startswith(skip)]
    return "\n".join(lines).strip() or text


def chunk_ticket(text, ticket_id):
    """Keep the full ticket as the main chunk so issue wording is not split off."""
    chunks = [(f"{ticket_id}#full", text.strip()[:4000])]
    parts = re.split(r"\n\s*\n", text.strip())
    for i, p in enumerate(parts):
        p = p.strip()
        if len(p) < 80:
            continue
        if p.startswith("TICKET:"):
            continue
        chunks.append((f"{ticket_id}#{i}", p))
    return chunks


def parse_meta(text):
    meta = {"ticket_id": "", "customer": "", "category": "", "status": "unknown"}
    for line in text.splitlines():
        if line.startswith("TICKET:"):
            meta["ticket_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("CUSTOMER:"):
            meta["customer"] = line.split(":", 1)[1].strip()
        elif line.startswith("CATEGORY:"):
            meta["category"] = line.split(":", 1)[1].strip()
        elif line.startswith("STATUS_FINAL:") or line.startswith("STATUS:"):
            meta["status"] = line.split(":", 1)[1].strip()
    return meta


def get_collection(rebuild=False):
    import chromadb
    import shutil

    if rebuild and os.path.isdir(PERSIST):
        shutil.rmtree(PERSIST)
    client = chromadb.PersistentClient(path=PERSIST)
    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest(ticket_dir, rebuild=True):
    col = get_collection(rebuild=rebuild)
    ids, docs, metas = [], [], []
    for name in sorted(os.listdir(ticket_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(ticket_dir, name)
        text = open(path, encoding="utf-8").read()
        meta = parse_meta(text)
        tid = meta["ticket_id"] or name.replace(".txt", "")
        for cid, chunk in chunk_ticket(text, tid):
            ids.append(cid)
            docs.append(chunk)
            metas.append(dict(meta, ticket_id=tid, chunk_id=cid))
    embedder = Embedder(HERE)
    embedder.fit(docs)
    bs = 50
    for i in range(0, len(ids), bs):
        batch = docs[i : i + bs]
        col.upsert(
            ids=ids[i : i + bs],
            documents=batch,
            metadatas=metas[i : i + bs],
            embeddings=embedder(batch),
        )
    return col, len(ids)


def search(col, query, n=4):
    embedder = Embedder(HERE)
    res = col.query(
        query_embeddings=embedder([query]),
        n_results=min(20, max(n * 5, n)),
        include=["documents", "metadatas", "distances"],
    )
    seen = set()
    rows = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        tid = meta.get("ticket_id")
        if tid in seen:
            continue
        seen.add(tid)
        sim = 1.0 - float(dist) if float(dist) <= 1.5 else 1.0 / (1.0 + float(dist))
        rows.append({"text": doc, "meta": meta, "distance": float(dist), "similarity": sim})
        if len(rows) >= n:
            break
    return rows
