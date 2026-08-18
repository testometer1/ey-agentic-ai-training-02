"""Tiny Word2Vec + KMeans demo on the same 4 sentences as the TF-IDF script."""

import os
import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "numpy", "scikit-learn", "gensim", "matplotlib"]
)

import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

docs = [
    "Agents plan act and observe in a loop",
    "The vector database stores embeddings for retrieval",
    "Agents use tools to observe and act on the environment",
    "Risk agents evaluate conservative moderate and aggressive profiles",
]
sentences = [doc.lower().split() for doc in docs]

# tiny corpus so min_count=1 and a lot of epochs
model = Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    seed=42,
    epochs=200,
)

words = list(model.wv.index_to_key)
vectors = np.array([model.wv[w] for w in words])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(vectors)

for i in range(3):
    print(f"Cluster {i}")
    print([w for w, lab in zip(words, labels) if lab == i])

xy = PCA(n_components=2, random_state=42).fit_transform(vectors)
plt.figure(figsize=(10, 8))
colors = ["tab:red", "tab:green", "tab:blue"]
for i in range(3):
    mask = labels == i
    plt.scatter(xy[mask, 0], xy[mask, 1], c=colors[i], label=f"Cluster {i}", s=90)
for x, y, word in zip(xy[:, 0], xy[:, 1], words):
    plt.annotate(word, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
plt.legend()
plt.title("Word2Vec K-means clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "cluster.png")
plt.savefig(out, dpi=150)
print("saved", out)
