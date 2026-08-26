#!/usr/bin/env python3
"""Semantic search over ~200 support tickets in ChromaDB.

Related to problem 8 (same queues) and feeds problem 17 (multi-hop).

  python3 ticket_search_demo.py
  python3 ticket_search_demo.py "app keeps logging me out"
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import chromadb  # noqa: F401
    import sklearn  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb", "numpy", "scikit-learn"])

from ticket_data import ensure_tickets
from ticket_store import ingest, search

QUERY = "app keeps logging me out"
TRUE_TICKET = "TCK-4021"


def main():
    query = " ".join(sys.argv[1:]) or QUERY
    ticket_dir = ensure_tickets()
    print("tickets:", ticket_dir)
    col, n = ingest(ticket_dir, rebuild=True)
    print("chunks indexed:", n)
    print("query:", query)
    print("(paraphrase test: no shared keywords with 'session expires randomly every few minutes')\n")
    rows = search(col, query, n=4)
    print("--- top matches ---")
    hit_rank = None
    for i, row in enumerate(rows, 1):
        tid = row["meta"].get("ticket_id", "")
        if tid == TRUE_TICKET and hit_rank is None:
            hit_rank = i
        print(
            f"{i}. {tid}  customer={row['meta'].get('customer')}  "
            f"category={row['meta'].get('category')}  status={row['meta'].get('status')}  "
            f"similarity={row['similarity']:.3f}  distance={row['distance']:.3f}"
        )
        print("   " + row["text"].replace("\n", " ")[:220])
        print()
    if TRUE_TICKET in {r["meta"].get("ticket_id") for r in rows[:3]}:
        print(f"PASS: {TRUE_TICKET} is in the top 3 (rank {hit_rank}).")
    else:
        print(f"FAIL: {TRUE_TICKET} was not in the top 3. Check embeddings / chunking.")
        sys.exit(1)


if __name__ == "__main__":
    main()
