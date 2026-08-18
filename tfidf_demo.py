from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

docs=[
    "Agents plan act and observe in a loop",
    "The vector database stores embeddings for retrieval",
    "Agents use tools to observe and act on the environment",
    "Risk agents evaluate conservative moderate and aggressive profiles",
]

v=TfidfVectorizer()
tfidf=v.fit_transform(docs)
scores=np.asarray(tfidf.max(axis=0)).ravel()
terms=v.get_feature_names_out()
order=np.argsort(scores)[::-1]
print("all word scores:")
for term, score in zip(terms[order], scores[order]):
    print(term, round(score, 3))
print("\ntop 5 words:")
for term, score in zip(terms[order][:5], scores[order][:5]):
    print(term, round(score, 3))
print("agents is low because it appears in many docs; conservative is high because it appears rarely.")
