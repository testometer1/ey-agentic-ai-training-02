#!/usr/bin/env python3
"""Configurable preprocess: URLs/emails/case/spaces, then optional stopwords + lemma."""

import importlib.util
import re
import subprocess
import sys


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
            print("skip", name, e)


def wordnet_pos(tag):
    from nltk.corpus import wordnet

    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text, remove_stopwords=False, lemmatization=False, stop_words=None, lemmatizer=None):
    import nltk

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if lemmatization and lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(w, pos=wordnet_pos(tag)) for w, tag in nltk.pos_tag(tokens)]
    if remove_stopwords and stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


def main():
    install("nltk")
    install("sklearn", "scikit-learn")

    from nltk.stem import WordNetLemmatizer
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

    stop_words = set(ENGLISH_STOP_WORDS)
    lemmatizer = WordNetLemmatizer()

    samples = [
        (
            "ops incident",
            "  HELLO TEAM!!!   please CHECK https://status.internal.ey.com/agents/loop-42\n"
            "the LIVE agent at ops@ey-demo.com  KEPT   calling   tools...\n"
            "   extra    spaces    and---broken\n"
            "formatting.  The planner was NOT observing before it acted!!!",
        ),
        (
            "retrieval note",
            "FYI -- Vector DB is  VERY  slow??  See docs: https://docs.pinecone.io/guide\n"
            "contact  retrieval.help@example.com\n"
            "EMBEDDINGS   were   NOT   updating....   RAG pipelines  keep  RUNNING  retrieval\n"
            "on   the   same   CHUNKS   again   and   again.",
        ),
        (
            "risk memo",
            "RISK DESK:   conservative profile  BUT  client wants AGGRESSIVE trades!!!\n"
            "email:  risk.committee@ey.com\n"
            "visit http://risk.internal/profiles?id=99\n"
            "   never    approve   high   leverage...  The  scoring  was  FAILING  for  new  accounts.",
        ),
    ]

    configs = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]

    for i, (title, raw) in enumerate(samples, 1):
        print(f"\n=== sample {i}: {title} ===")
        print(raw)
        for stop, lemma in configs:
            cleaned = preprocess_text(
                raw,
                remove_stopwords=stop,
                lemmatization=lemma,
                stop_words=stop_words,
                lemmatizer=lemmatizer,
            )
            toks = cleaned.split()
            neg = any(w in toks for w in ("not", "never", "but"))
            print(f"\nstop={stop} lemma={lemma}  tokens={len(toks)} unique={len(set(toks))} negation={neg}")
            print(cleaned)

    print(
        """
Best for clustering / TF-IDF: stop=True, lemma=True.
If the next step is sentiment, leave stopwords in (see problem 2).
"""
    )


if __name__ == "__main__":
    main()
