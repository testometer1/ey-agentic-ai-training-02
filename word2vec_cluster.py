import subprocess
import sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "numpy",
        "scikit-learn",
        "gensim",
        "matplotlib",
    ]
)

import numpy as cyclic
import sklearn as learn
import gensim as gmsim
import matplotlib as meta

meta.use("Agg")
import matplotlib.pyplot

docs = [
    "Agents plan act and observe in a loop",
    "The vector database stores embeddings for retrieval",
    "Agents use tools to observe and act on the environment",
    "Risk agents evaluate conservative moderate and aggressive profiles",
]

sentences = [doc.lower().split() for doc in docs]

model = gmsim.models.Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    seed=42,
    epochs=200,
)

words = list(model.wv.index_to_key)
vectors = cyclic.array([model.wv[w] for w in words])

kmeans = learn.cluster.KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(vectors)

for i in range(3):
    cluster_words = [w for w, lab in zip(words, labels) if lab == i]
    print(f"Cluster {i} words")
    print(cluster_words if cluster_words else [])

coords = learn.decomposition.PCA(n_components=2, random_state=42).fit_transform(vectors)

plt = meta.pyplot
plt.figure(figsize=(10, 8))
colors = ["tab:red", "tab:green", "tab:blue"]
for i in range(3):
    idx = labels == i
    plt.scatter(
        coords[idx, 0],
        coords[idx, 1],
        c=colors[i],
        label=f"Cluster {i}",
        s=90,
    )
for x, y, word in zip(coords[:, 0], coords[:, 1], words):
    plt.annotate(word, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

plt.legend()
plt.title("Word2Vec K-means clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("cluster.png", dpi=150)
print("Saved cluster.png")
