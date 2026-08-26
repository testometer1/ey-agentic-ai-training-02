#!/usr/bin/env python3
"""Simple next-word predictor from a phrase frequency table.

This is the greedy cousin of problem 9 (weighted n-gram generation):
same tokenisation idea, but we take the single most common following word
and report confidence = count / total for that phrase.

  python3 next_word_demo.py
  python3 next_word_demo.py "the quick brown"
  python3 next_word_demo.py --corpus phrases.txt "the agent must"
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(HERE, "phrases.txt")
P9_CORPUS = os.path.join(HERE, "..", "problem_9_ngram_generation", "domain_corpus.txt")


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def build_table(tokens, max_n=4):
    """Map phrase (1..max_n-1 tokens) -> Counter of next words."""
    table = defaultdict(Counter)
    for n in range(2, max_n + 1):
        hist = n - 1
        for i in range(len(tokens) - hist):
            phrase = tuple(tokens[i : i + hist])
            nxt = tokens[i + hist] if i + hist < len(tokens) else None
            if nxt:
                table[phrase][nxt] += 1
    return table


def predict(table, phrase_tokens, max_n=4):
    """Longest matching phrase first (up to max_n-1 tokens)."""
    words = phrase_tokens
    for hist in range(min(len(words), max_n - 1), 0, -1):
        key = tuple(words[-hist:])
        counter = table.get(key)
        if counter:
            word, count = counter.most_common(1)[0]
            total = sum(counter.values())
            return word, count, total, key
    return None, 0, 0, None


def load_text(path):
    return open(path, encoding="utf-8").read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phrase", nargs="*", help="seed phrase")
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    args = p.parse_args()

    path = args.corpus
    if not os.path.isfile(path) and os.path.isfile(P9_CORPUS):
        path = P9_CORPUS
        print("using problem 9 domain corpus")
    if not os.path.isfile(path):
        sys.exit("no corpus: " + path)

    tokens = tokenize(load_text(path))
    table = build_table(tokens)
    print("corpus:", os.path.abspath(path))
    print("tokens:", len(tokens), "phrase keys:", len(table))

    if args.phrase:
        phrase = " ".join(args.phrase)
    else:
        try:
            phrase = input("phrase: ").strip()
        except EOFError:
            phrase = "the quick brown"
            print(phrase)

    seed = tokenize(phrase)
    if not seed:
        sys.exit("empty phrase")

    word, count, total, key = predict(table, seed)
    if not word:
        print("no following-word counts for that phrase")
        sys.exit(1)
    conf = 100.0 * count / total
    print(f"matched phrase: {' '.join(key)}")
    print(f"predicted next word: {word}")
    print(f"confidence: {conf:.0f}% ({count}/{total})")


if __name__ == "__main__":
    main()
