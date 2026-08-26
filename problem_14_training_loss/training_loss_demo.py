#!/usr/bin/env python3
"""Simulate next-token training: embeddings + softmax, more data, lower loss.

Problem 13 is a frozen frequency table. Here the predictor starts random and
is updated with SGD. Three dataset sizes (100 / 1000 / 10000 sentences) and
two embedding widths (small vs wide) so you can see:
  1) loss falls with more training
  2) a wider model can overfit the 100-sentence set (train acc high, held-out worse)
  3) scaling data improves held-out accuracy

  python3 training_loss_demo.py
"""

import math
import os
import random
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P13 = os.path.join(HERE, "..", "problem_13_next_word", "phrases.txt")
P9 = os.path.join(HERE, "..", "problem_9_ngram_generation", "domain_corpus.txt")

TEMPLATES = [
    "the agent must observe the environment before the next action",
    "the planner then selects a tool and the executor calls the tool",
    "rag pipelines retrieve context before the answer is written",
    "cosine similarity ranks documents in the vector store",
    "conservative clients must not receive aggressive trades",
    "the risk desk scores conservative moderate and aggressive profiles",
    "guardrails block identical tool calls in the production loop",
    "rebuild the embedding index when chunks stop matching questions",
    "session expires randomly every few minutes on the mobile app",
    "new hires receive twenty vacation days in the first year",
    "the relationship manager writes a two paragraph client review",
    "the planner produces a plan and does not touch retrieval",
    "the executor runs each step and grades the retrieved context",
    "chroma stores embeddings for each ticket chunk with metadata",
    "sarah chen holds account rsk-prof-c01 on the london desk",
]


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def seed_sentences():
    lines = []
    for path in (P13, P9):
        if os.path.isfile(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip().rstrip(".")
                if len(tokenize(line)) >= 5:
                    lines.append(line.lower())
    return lines or [t for t in TEMPLATES]


def make_sentences(n, rng):
    base = seed_sentences()
    desks = ["london", "new york", "singapore", "dublin"]
    out = []
    while len(out) < n:
        t = rng.choice(base + TEMPLATES)
        words = tokenize(t)
        if rng.random() < 0.35:
            words = words + ["in", rng.choice(desks)]
        if len(words) >= 4:
            out.append(words)
    return out


def pairs(sentences):
    xs, ys = [], []
    for s in sentences:
        for i in range(len(s) - 1):
            xs.append(s[i])
            ys.append(s[i + 1])
    return xs, ys


def vocab_from(sentences):
    v = sorted({w for s in sentences for w in s})
    return {w: i for i, w in enumerate(v)}


class Predictor:
    def __init__(self, n_vocab, dim, rng):
        scale = 0.08
        self.emb = rng.normal(0, scale, size=(n_vocab, dim)).astype(np.float64)
        self.w = rng.normal(0, scale, size=(dim, n_vocab)).astype(np.float64)
        self.b = np.zeros(n_vocab, dtype=np.float64)

    def logits(self, idx):
        return self.emb[idx] @ self.w + self.b

    def step_batch(self, x_idx, y_idx, lr):
        # mean cross-entropy + SGD
        z = self.emb[x_idx] @ self.w + self.b
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z)
        p = exp / exp.sum(axis=1, keepdims=True)
        n = len(x_idx)
        loss = -np.log(p[np.arange(n), y_idx] + 1e-12).mean()
        acc = (p.argmax(axis=1) == y_idx).mean()
        dz = p
        dz[np.arange(n), y_idx] -= 1
        dz /= n
        g_w = self.emb[x_idx].T @ dz
        g_b = dz.sum(axis=0)
        g_emb = dz @ self.w.T
        self.w -= lr * g_w
        self.b -= lr * g_b
        np.add.at(self.emb, x_idx, -lr * g_emb)
        return float(loss), float(acc)


def eval_split(model, x_idx, y_idx):
    z = model.logits(x_idx)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(x_idx)), y_idx] + 1e-12).mean()
    acc = (p.argmax(axis=1) == y_idx).mean()
    return float(loss), float(acc)


def ascii_chart(rows, title):
    # rows: list of (label, loss)
    print(f"\n{title}")
    if not rows:
        return
    mx = max(v for _, v in rows) or 1
    width = 40
    for label, v in rows:
        n = int(round(width * v / mx))
        print(f"  {label:<22} {v:6.3f} {'#' * n}")


def run():
    rng = np.random.default_rng(7)
    py = random.Random(7)
    held_out = make_sentences(200, py)
    # shared vocab from a large pool so indices stay aligned
    pool = make_sentences(12000, py)
    stoi = vocab_from(pool + held_out)
    n_vocab = len(stoi)

    def encode(sentences):
        xs, ys = pairs(sentences)
        x = np.array([stoi[w] for w in xs if w in stoi and True], dtype=np.int64)
        # filter pairs in vocab
        xi, yi = [], []
        for a, b in zip(xs, ys):
            if a in stoi and b in stoi:
                xi.append(stoi[a])
                yi.append(stoi[b])
        return np.array(xi, dtype=np.int64), np.array(yi, dtype=np.int64)

    hx, hy = encode(held_out)
    configs = [
        ("small-d8", 8, [100, 1000, 10000]),
        ("wide-d48", 48, [100, 1000, 10000]),
    ]
    table = []
    chart_pts = []

    print("vocab:", n_vocab, "held-out next-token pairs:", len(hx))
    print("untrained = random embeddings; each epoch is one pass over that dataset size\n")

    for name, dim, sizes in configs:
        model = Predictor(n_vocab, dim, rng)
        # epoch 0: random
        loss0, acc0 = eval_split(model, hx, hy)
        table.append((0, name, 0, loss0, acc0, "held-out"))
        chart_pts.append((f"{name} e0", loss0))
        epoch = 0
        for size in sizes:
            data = make_sentences(size, py)
            x, y = encode(data)
            # more steps on tiny data so the wide net can overfit
            passes = 12 if size == 100 else (4 if size == 1000 else 2)
            tr_loss = tr_acc = 0
            for _ in range(passes):
                epoch += 1
                order = rng.permutation(len(x))
                bs = 64
                losses, accs = [], []
                for i in range(0, len(x), bs):
                    sl = order[i : i + bs]
                    lr = 0.35 if size == 100 else 0.25
                    L, A = model.step_batch(x[sl], y[sl], lr)
                    losses.append(L)
                    accs.append(A)
                tr_loss, tr_acc = float(np.mean(losses)), float(np.mean(accs))
            ho_loss, ho_acc = eval_split(model, hx, hy)
            table.append((epoch, name, size, tr_loss, tr_acc, "train"))
            table.append((epoch, name, size, ho_loss, ho_acc, "held-out"))
            chart_pts.append((f"{name} n={size}", ho_loss))
            gap = tr_acc - ho_acc
            print(
                f"{name:10} data={size:<6} train_loss={tr_loss:.3f} train_acc={tr_acc:.3f}  "
                f"heldout_loss={ho_loss:.3f} heldout_acc={ho_acc:.3f}  acc_gap={gap:+.3f}"
            )

    print("\n--- table: epoch | model | dataset size | split | avg loss | top-1 acc ---")
    print(f"{'epoch':>6} {'model':<10} {'n':>7} {'split':<9} {'loss':>8} {'acc':>8}")
    for epoch, name, size, loss, acc, split in table:
        print(f"{epoch:6d} {name:<10} {size:7d} {split:<9} {loss:8.3f} {acc:8.3f}")

    ascii_chart(chart_pts, "held-out loss (lower is better)")
    print(
        "\nread: loss drops as n grows. On n=100 the wide model often has a larger "
        "train-vs-held-out accuracy gap (overfit). Scaling to 10000 shrinks that gap."
    )


if __name__ == "__main__":
    run()
