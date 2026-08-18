#!/usr/bin/env python3
"""24 domain sentences -> TF-IDF -> KMeans (k=4) -> PCA scatter."""

import os
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "scikit-learn", "matplotlib"])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "The control loop plans then acts then observes",
    "Planning happens before the next action in the loop",
    "Observing state in the loop helps decide the next action",
    "A planning loop can retry after a failed action",
    "The system acts only after it observes new state",
    "The observe plan act cycle keeps the loop grounded",
    "The vector database stores embeddings for retrieval",
    "Embeddings support retrieval of similar documents from storage",
    "A vector database indexes chunks for fast retrieval search",
    "Retrieval uses embeddings to find related context",
    "Similarity search ranks documents in the vector store",
    "RAG pipelines run retrieval on chunks before the model answers",
    "Risk desks evaluate conservative moderate and aggressive profiles",
    "Conservative clients prefer low risk investment profiles",
    "Aggressive profiles accept higher risk for higher return",
    "Moderate risk sits between conservative and aggressive",
    "The risk desk recommends a profile after scoring",
    "Portfolio risk depends on the chosen client profile",
    "Tool calls query external APIs during a live run",
    "Production systems must log every tool call",
    "Unsafe tool use can harm the live service",
    "The executor selects a tool for the given task",
    "Guardrails block dangerous tool calls in production",
    "Production guardrails reject unsafe API tool calls",
]

K = 4
vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(sentences)
names = vec.get_feature_names_out()
labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X)

themes = {
    "retrieval": ("retriev", "embed", "vector", "chunk", "similar", "rag"),
    "risk": ("risk", "conservative", "aggressive", "moderate", "profile", "portfolio"),
    "tools": ("tool", "production", "guardrail", "unsafe", "api", "log"),
    "loop": ("plan", "observe", "act", "loop", "cycle", "state"),
}


def name_cluster(terms):
    blob = " ".join(terms)
    scores = {k: sum(w in blob for w in kws) for k, kws in themes.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "mixed"


print("clusters")
for c in range(K):
    rows = X[labels == c]
    mean = rows.mean(axis=0)
    mean = mean.A1 if hasattr(mean, "A1") else mean.ravel()
    top = [names[i] for i in mean.argsort()[::-1][:6]]
    print(f"\ncluster {c} ({name_cluster(top)})")
    print("  terms:", ", ".join(top))
    for i, s in enumerate(sentences):
        if labels[i] == c:
            print(f"  S{i+1:02d}. {s}")

xy = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
plt.figure(figsize=(11, 8))
colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
for c in range(K):
    m = labels == c
    plt.scatter(xy[m, 0], xy[m, 1], c=colors[c], label=f"cluster {c}", s=90)
for i, (x, y) in enumerate(xy):
    plt.annotate(f"S{i+1}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
plt.legend()
plt.title("sentence clusters (tfidf + kmeans + pca)")
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "sentence_clusters.png")
plt.savefig(out, dpi=150)
print("\nsaved", out)
