#!/usr/bin/env python3
"""Stop-word trap: sentiment with vs without not/never/very/but."""

import importlib.util
import re
import subprocess
import sys


def install(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg])


install("nltk")
install("sklearn", "scikit-learn")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import nltk

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

STOP = set(ENGLISH_STOP_WORDS)
vader = SentimentIntensityAnalyzer()

sentences = [
    "The retrieval agent is not good on live queries.",
    "Risk agents never succeed on live tool calls.",
    "The vector database is very useful for embedding search.",
    "The planner is fast but not safe for production tool calls.",
    "This agent is not bad at observing the environment.",
]


def tokens(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()
    return [t for t in text.split() if t]


def polarity(text):
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, score


print("in sklearn stop-words?", {w: w in STOP for w in ["not", "never", "very", "but"]})
print()

print(f"{'#':<3} {'with stops':<20} {'without':<20} {'flip':<5} sentence")
print("-" * 90)
flips = []
for i, sent in enumerate(sentences, 1):
    stripped = " ".join(t for t in tokens(sent) if t not in STOP)
    a_lab, a_sc = polarity(sent)
    b_lab, b_sc = polarity(stripped)
    flip = a_lab != b_lab
    if flip:
        flips.append((i, sent, a_lab, a_sc, b_lab, b_sc))
    print(f"{i:<3} {a_lab} ({a_sc:+.3f})".ljust(24) + f"{b_lab} ({b_sc:+.3f})".ljust(22) + f"{'YES' if flip else 'no':<5} {sent}")
    print(f"    filtered: {stripped}")

print("\nFlips:")
for i, sent, a_lab, a_sc, b_lab, b_sc in flips:
    print(f"  {i}. {sent}")
    print(f"     {a_lab} ({a_sc:+.3f}) -> {b_lab} ({b_sc:+.3f})")

print(
    """
'not' / 'never' / 'but' are in the stop list, so 'not good' becomes 'good'.
That's why 1, 2, 4, 5 flip. Sentence 3 only loses 'very', so it stays positive.
"""
)
