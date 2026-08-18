#!/usr/bin/env python3
"""Stemming vs lemmatization on 10 words (NLTK)."""

import importlib.util
import subprocess
import sys


def install(pkg):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def nltk_data(name, path=None):
    import nltk

    path = path or {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    }.get(name, name)
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, quiet=True)


install("nltk")

from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk_data("wordnet")
nltk_data("omw-1.4")
nltk_data("averaged_perceptron_tagger")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# assumed POS so the lemmatizer isn't guessing in a vacuum
words = {
    "Running": "v",
    "Studies": "n",
    "better": "a",
    "tigers": "n",
    "slower": "a",
    "parents": "n",
    "complanies": "n",  # misspelled on purpose
    "leaves": "v",
    "Workflow": "n",
    "leafes": "n",  # misspelled on purpose
}

print(f"{'Word':<12} {'Stem':<12} {'Lemma':<12}")
print("-" * 36)
for word, pos in words.items():
    w = word.lower()
    print(f"{word:<12} {stemmer.stem(w):<12} {lemmatizer.lemmatize(w, pos=pos):<12}")

print(
    """
Stemming just chops endings (studies -> studi).
Lemmatization looks up a real base form (studies -> study).
'complanies' / 'leafes' are left as-is so the tools look confused, which is expected.
"""
)
