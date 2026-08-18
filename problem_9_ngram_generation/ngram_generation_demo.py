#!/usr/bin/env python3
"""Bigram / trigram / 4-gram generator.

Weighted random sampling (not argmax). Back off to a shorter n if unseen.
4-grams copy the corpus — that's the lesson.
"""

import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "domain_corpus.txt")
SEED = "the agent must observe"
N_WORDS = 50
rng = random.Random(42)


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def build_corpus():
    sop = (
        "The agent must observe the environment before the next action. "
        "The planner then selects a tool, the executor calls the tool, and the loop records the result. "
        "If the tool call fails, the agent must observe the new error and must not retry blindly. "
        "Guardrails block more than three identical tool calls in one production loop. "
        "This observe plan act cycle keeps live retrieval agents grounded in London and New York. "
    )
    extra_bits = [
        "Live retrieval agents in London call the billing API through the production executor. "
        "When retrieval quality drops, the vector database returns no policy chunks and the model hallucinates numbers. "
        "Pinecone stores embeddings for European chunks and OpenAI embeddings must stay fresh. "
        "Sarah Chen stopped run AGNT-LOOP-42 after forty repeated tool calls appeared in Splunk.",
        "Conservative clients must not receive aggressive trades. "
        "The risk desk in New York scores conservative, moderate, and aggressive profiles after each failed recommendation. "
        "EY partners asked for a written explanation of profile RSK-PROF-C01 and the two point four million notional. "
        "Leverage for new accounts stays low until retrieval quality is confirmed.",
        "RAG pipelines retrieve context before the answer is written. "
        "Cosine similarity ranks documents in the vector store. "
        "Index EMB-IDX-EU-9 lives in California and fails over to Microsoft Azure in Dublin and Paris. "
        "Rebuild the embedding index whenever chunks stop matching similar questions.",
        "Production change TOOL-GRD-3 logs every tool call and rejects unsafe retries. "
        "Amazon CloudWatch alarms fire when the same API request repeats. "
        "Operators in Singapore and London confirmed the rule during a tabletop exercise. "
        "No customer funds were moved and no live trades were placed.",
        "Support tickets split into four queues: agent loop failures, retrieval index issues, "
        "risk profile reviews, and production tool guardrails. "
        "A person on the desk would sort those records the same way a clustering model should. "
        "Averaging word vectors still loses word order, so not ready can sit near ready.",
        "Jaccard overlap counts shared tokens and misses meaning. "
        "Document vectors from en_core_web_md rise on paraphrases and fall on approved versus rejected. "
        "The retrieval agent is ready for production is not the same as the retrieval agent is not ready for production. "
        "That stop-word trap is why sentiment must keep not and never.",
        "Internal product codes such as AGNT-LOOP-42, EMB-IDX-EU-9, RSK-PROF-C01, and TOOL-GRD-3 "
        "are missing from the default named entity model. Recording those misses is the NER exercise, "
        "not a reason to rewrite the tickets with only famous company names.",
        "Preprocessing lowercases text, strips URLs and email addresses, and collapses extra spaces. "
        "Lemmatization maps running and failing to run and fail. "
        "Removing stop-words helps TF-IDF and K-Means and hurts sentiment. "
        "The best clustering configuration used both flags together.",
    ]
    desks = ["London", "New York", "San Francisco", "Singapore", "Chicago", "Dublin", "Paris"]
    systems = ["planner", "executor", "vector database", "risk desk", "guardrail service", "embedding index"]
    notes = []
    for i in range(1, 91):
        notes.append(
            f"On run {i} the {systems[i % len(systems)]} in {desks[i % len(desks)]} recorded a timeout "
            f"and asked the agent to observe before acting. Ticket {i} stayed in the same queue until a "
            f"human confirmed the next tool call. The written summary for run {i} said the loop was unsafe "
            f"until logging was complete."
        )
    return " ".join([sop] * 10 + extra_bits + notes)


def ngram_table(tokens, n):
    table = defaultdict(Counter)
    for i in range(len(tokens) - n + 1):
        table[tuple(tokens[i : i + n - 1])][tokens[i + n - 1]] += 1
    return table


def pick(counter):
    words = list(counter)
    return rng.choices(words, [counter[w] for w in words], k=1)[0]


def generate(tables, unigrams, seed, n_words, order):
    out = list(seed)
    backoff = 0
    while len(out) < n_words:
        got = False
        for n in range(order, 1, -1):
            hist = n - 1
            if len(out) < hist:
                continue
            nxt = tables[n].get(tuple(out[-hist:]))
            if nxt:
                out.append(pick(nxt))
                if n != order:
                    backoff += 1
                got = True
                break
        if not got:
            if not unigrams:
                break
            out.append(pick(unigrams))
            backoff += 1
    return out[:n_words], backoff


def longest_copy(gen, corpus):
    best = 0
    for n in range(3, len(gen) + 1):
        seen = {tuple(corpus[i : i + n]) for i in range(len(corpus) - n + 1)}
        if any(tuple(gen[i : i + n]) in seen for i in range(len(gen) - n + 1)):
            best = n
        else:
            break
    return best


if os.path.exists(CORPUS):
    text = open(CORPUS, encoding="utf-8").read()
else:
    text = build_corpus()
    open(CORPUS, "w", encoding="utf-8").write(text)
    print("wrote", CORPUS)

tokens = tokenize(text)
print("tokens:", len(tokens))
if len(tokens) < 5000:
    sys.exit("need 5000+ words")

unigrams = Counter(tokens)
tables = {2: ngram_table(tokens, 2), 3: ngram_table(tokens, 3), 4: ngram_table(tokens, 4)}
seed = tokenize(SEED)
print("sampling: weighted random (not greedy)")
print("unseen n-gram: back off, then unigram")
print("seed:", seed)

stats = {}
for order, name in [(2, "bigram"), (3, "trigram"), (4, "four-gram")]:
    gen, bo = generate(tables, unigrams, seed, N_WORDS, order)
    copy = longest_copy(gen, tokens)
    rep = 1 - len(set(gen)) / len(gen)
    stats[name] = (copy, rep, bo)
    print(f"\n{name} ({len(gen)} words, backoff={bo}, verbatim={copy}, rep={rep:.2f})")
    print(" ".join(gen))

print("\ntrigram is usually the most readable. bigram jumps around.")
print(
    f"longest copy from source: bi {stats['bigram'][0]}, tri {stats['trigram'][0]}, "
    f"4g {stats['four-gram'][0]}. four-gram mostly memorises — that's the setup for transformers."
)
