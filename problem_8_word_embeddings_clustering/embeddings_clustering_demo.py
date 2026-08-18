#!/usr/bin/env python3
"""220 support tickets -> preprocess (prob 7) -> Word2Vec mean -> KMeans -> PCA.

K is fixed at 4 (four queues). Elbow plot is saved so that isn't just a guess.
Word2Vec is trained on this tiny set on purpose — pretrained vectors would be better.
Averaging words throws away order.
"""

import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "problem_7_preprocessing_pipeline"))

from preprocessing_pipeline_demo import install, nltk_data, preprocess_text

CSV = os.path.join(HERE, "support_tickets.csv")
K = 4


def make_tickets():
    loop = [
        "The live agent entered a tight tool call loop after a timeout on run {n}.",
        "The planner skipped the observe step and retried action {n} too quickly.",
        "Control loop {n} planned then acted then observed the new state.",
        "Observing state in loop {n} should happen before the next action.",
        "The observe plan act cycle failed on ticket {n} and kept retrying.",
        "Run {n} acted without observing, so the loop never reached a stable state.",
        "Planning for loop {n} must wait for the latest observed error.",
        "The agent retry loop on case {n} ignored the failed observe signal.",
    ]
    retrieval = [
        "Vector database retrieval for index {n} returned no policy chunks.",
        "Embeddings for document {n} are stale so similarity search ranks the wrong chunks.",
        "Pinecone retrieval quality dropped on European chunk set {n}.",
        "Cosine similarity search on embedding {n} failed to find related context.",
        "RAG pipeline {n} must retrieve chunks before the model writes an answer.",
        "The search index stored embeddings for retrieval request {n} incorrectly.",
        "Chunk retrieval {n} hallucinated numbers because the vector store was empty.",
        "Rebuild embedding index {n} so similar documents can be retrieved again.",
    ]
    risk = [
        "Conservative profile {n} should not receive an aggressive trade.",
        "Risk desk scoring for client {n} sits between moderate and aggressive.",
        "Leverage on account {n} is too high for a conservative investment profile.",
        "The committee rejected aggressive risk for new client {n}.",
        "Portfolio risk for profile {n} needs a written conservative explanation.",
        "Moderate risk client {n} asked to move toward a more aggressive profile.",
        "Failed recommendation {n} used the wrong conservative versus aggressive mapping.",
        "Score the investment profile for ticket {n} before approving leverage.",
    ]
    tools = [
        "Production guardrails blocked an unsafe tool call on job {n}.",
        "Log every tool call in production run {n} before retrying the API.",
        "Unsafe tool use on service {n} can harm the live billing API.",
        "Change {n} rejects dangerous tool calls in the production executor.",
        "CloudWatch fired because tool call {n} repeated more than three times.",
        "The executor selected an unsafe tool for production task {n}.",
        "Guardrail policy {n} must log and reject identical production tool calls.",
        "Live API tool retries on incident {n} were blocked by production safety rules.",
    ]
    rows = []
    n = 1
    for queue, tmpls in [
        ("agent_loop", loop),
        ("retrieval", retrieval),
        ("risk", risk),
        ("tools", tools),
    ]:
        for i in range(55):
            rows.append({"ticket_id": f"TCK-{n:03d}", "queue": queue, "text": tmpls[i % len(tmpls)].format(n=n)})
            n += 1
    return rows


def mean_vec(tokens, model, size):
    import numpy as np

    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(size)
    return np.mean(vecs, axis=0)


def main():
    install("nltk")
    install("sklearn", "scikit-learn")
    install("numpy")
    install("pandas")
    install("gensim")
    install("matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from gensim.models import Word2Vec
    from nltk.stem import WordNetLemmatizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    for name, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        nltk_data(name, path)

    stops = set(ENGLISH_STOP_WORDS)
    lemma = WordNetLemmatizer()

    if not os.path.exists(CSV):
        pd.DataFrame(make_tickets()).to_csv(CSV, index=False)
        print("wrote", CSV)

    df = pd.read_csv(CSV)
    print(len(df), "rows")

    token_lists = []
    cleaned = []
    for text in df["text"].astype(str):
        c = preprocess_text(text, True, True, stops, lemma)
        toks = c.split() or ["empty"]
        cleaned.append(c)
        token_lists.append(toks)
    df["cleaned"] = cleaned

    print("training word2vec on 220 tickets (weak on purpose)")
    w2v = Word2Vec(token_lists, vector_size=50, window=3, min_count=1, workers=1, seed=42, epochs=80)
    M = np.vstack([mean_vec(t, w2v, 50) for t in token_lists])

    ks, inertias = range(2, 9), []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(M)
        inertias.append(km.inertia_)
        print(f"k={k} inertia={km.inertia_:.1f}")

    plt.figure(figsize=(7, 4))
    plt.plot(list(ks), inertias, marker="o")
    plt.title("elbow")
    plt.xlabel("k")
    plt.ylabel("inertia")
    plt.tight_layout()
    elbow = os.path.join(HERE, "elbow.png")
    plt.savefig(elbow, dpi=150)
    print("saved", elbow)
    print(f"using k={K} because there are four ticket queues")

    df["cluster"] = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(M)
    xy = PCA(n_components=2, random_state=42).fit_transform(M)

    plt.figure(figsize=(10, 8))
    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
    for c in range(K):
        m = df["cluster"] == c
        plt.scatter(xy[m, 0], xy[m, 1], c=colors[c], label=f"cluster {c}", s=40, alpha=0.8)
    plt.legend()
    plt.title("tickets (word2vec mean + kmeans + pca)")
    plt.tight_layout()
    scatter = os.path.join(HERE, "embedding_clusters.png")
    plt.savefig(scatter, dpi=150)
    print("saved", scatter)

    names = {
        "agent_loop": "plan-act-observe loops",
        "retrieval": "retrieval / embeddings",
        "risk": "risk profiles",
        "tools": "production tool calls",
    }
    rng = random.Random(42)
    for c in range(K):
        part = df[df["cluster"] == c]
        cnt = Counter()
        for t in part["cleaned"]:
            cnt.update(t.split())
        top = [w for w, _ in cnt.most_common(5)]
        examples = part["text"].tolist()
        rng.shuffle(examples)
        q = part["queue"].mode().iloc[0]
        print(f"\ncluster {c}: {names.get(q, q)}  n={len(part)}")
        print("  terms:", ", ".join(top))
        for e in examples[:3]:
            print("  -", e)

    print("\nA person on the desk would split these the same way (loop / retrieval / risk / tools).")
    print("Mean-pooling still loses word order, so 'not conservative' can land next to conservative tickets.")


if __name__ == "__main__":
    main()
