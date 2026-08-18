#!/usr/bin/env python3
"""POS, NER, classification, sentiment, n-grams on 3 domain notes."""

import importlib.util
import subprocess
import sys
from collections import Counter


def install(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg])


def nltk_data(name, path):
    import nltk

    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(name, quiet=True)
        except Exception as e:
            print("couldn't download", name, e)


def sentiment_label(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def entities_from_tree(tree):
    out = []
    for node in tree:
        if hasattr(node, "label"):
            out.append((node.label(), " ".join(tok for tok, _ in node.leaves())))
    return out


install("nltk")
install("sklearn", "scikit-learn")

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

for name, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ("maxent_ne_chunker", "chunkers/maxent_ne_chunker"),
    ("maxent_ne_chunker_tab", "chunkers/maxent_ne_chunker_tab"),
    ("words", "corpora/words"),
    ("stopwords", "corpora/stopwords"),
    ("vader_lexicon", "sentiment/vader_lexicon"),
]:
    nltk_data(name, path)

vader = SentimentIntensityAnalyzer()
stops = set(stopwords.words("english"))

docs = [
    (
        "ops incident",
        "On Monday, Sarah Chen from the London operations desk stopped a live agent that kept calling tools in a tight loop. "
        "The planner asked the python executor to retry a failed database query without observing the new error. "
        "Splunk logs in London showed forty repeated tool calls against the billing service at Amazon. "
        "The agent must observe the environment before the next action. "
        "Guardrails now block unsafe production retries.",
    ),
    (
        "retrieval note",
        "Engineers in San Francisco use OpenAI embeddings so similar questions retrieve the same policy chunks. "
        "The Pinecone vector database in California stores those embeddings and ranks documents by cosine similarity. "
        "Microsoft Azure hosts the search index for European clients. "
        "If retrieval fails, the model hallucinates numbers instead of citing a chunk. "
        "RAG pipelines retrieve context before the answer is written.",
    ),
    (
        "risk memo",
        "The risk agent in New York reviewed conservative, moderate, and aggressive profiles for Goldman Sachs clients. "
        "Conservative investors should not receive aggressive trades. "
        "The committee never approved high leverage for new accounts after the failed recommendation. "
        "EY partners in New York asked for a written explanation of each profile score. "
        "Sentiment on the trading floor was negative, but the moderate profile remains useful.",
    ),
]

train_x = [
    "The agent plans the next action and observes the environment.",
    "Production agents must log every tool call and retry safely.",
    "Guardrails block unsafe tool use in the live loop.",
    "The planner selects tools after it observes a new error.",
    "The vector database stores embeddings for retrieval search.",
    "RAG pipelines retrieve similar chunks before answering.",
    "Embeddings rank documents in the vector store by similarity.",
    "A search index returns related context from stored chunks.",
    "Risk agents evaluate conservative moderate and aggressive profiles.",
    "Conservative clients prefer low risk investment profiles.",
    "Aggressive profiles accept higher portfolio risk for return.",
    "The committee reviews leverage and client risk scores.",
]
train_y = ["agent_ops"] * 4 + ["retrieval"] * 4 + ["risk"] * 4

tfidf = TfidfVectorizer(stop_words="english")
clf = MultinomialNB()
clf.fit(tfidf.fit_transform(train_x), train_y)

for i, (title, text) in enumerate(docs, 1):
    print(f"\n===== {i}. {title} =====")
    print(text)

    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    ents = entities_from_tree(nltk.ne_chunk(tagged))
    X = tfidf.transform([text])
    pred = clf.predict(X)[0]
    probs = dict(zip(clf.classes_, clf.predict_proba(X)[0]))
    sent = vader.polarity_scores(text)
    content = [w.lower() for w, tag in tagged if w.isalpha() and w.lower() not in stops]

    print("\nPOS sample:", " ".join(f"{w}/{t}" for w, t in tagged[:15]), "...")
    print("nouns", sum(t.startswith("NN") for _, t in tagged), "verbs", sum(t.startswith("VB") for _, t in tagged))
    print("NER:", ents or "(none)")
    print(f"topic: {pred}  { {k: round(v, 2) for k, v in probs.items()} }")
    print(f"sentiment: {sentiment_label(sent['compound'])}  {sent['compound']:+.3f}")
    for n, name in [(1, "uni"), (2, "bi"), (3, "tri")]:
        top = Counter(" ".join(g) for g in ngrams(content, n)).most_common(6)
        print(name + ":", ", ".join(f"{p} ({c})" for p, c in top) if top else "(too short)")

print(
    """
Most useful here: classification. These notes need to be routed (ops / retrieval / risk).
NER is close. Sentiment is noisy on technical writing.
"""
)
