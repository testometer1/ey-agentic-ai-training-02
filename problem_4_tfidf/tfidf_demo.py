from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

docs = [
    "Agents plan act and observe in a loop",
    "The vector database stores embeddings for retrieval",
    "Agents use tools to observe and act on the environment",
    "Risk agents evaluate conservative moderate and aggressive profiles",
]

vec = TfidfVectorizer()
tfidf = vec.fit_transform(docs)
scores = np.asarray(tfidf.max(axis=0)).ravel()
terms = vec.get_feature_names_out()
order = np.argsort(scores)[::-1]

print("all word scores:")
for term, score in zip(terms[order], scores[order]):
    print(term, round(score, 3))

print("\ntop 5:")
for term, score in zip(terms[order][:5], scores[order][:5]):
    print(term, round(score, 3))

print("\nagents is low because it shows up in a lot of docs; conservative is high because it's rare.")
