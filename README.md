# livedemo — problems 1–17

Training scripts for the EY agentic-AI track. Later problems reuse earlier artefacts (same client, same tickets, same OpenRouter helper pattern).

| # | Folder | What it shows | Ties to |
|---|--------|----------------|---------|
| 12 | `problem_12_rm_review` | Two-paragraph RM review + fact check | Client Sarah Chen / `RSK-PROF-C01` from 8–9 |
| 13 | `problem_13_next_word` | Next-word frequency table + confidence | Greedy version of problem 9 n-grams |
| 14 | `problem_14_training_loss` | SGD next-token loss vs data size | Trains what 13 froze; sentences from 9/13 |
| 15 | `problem_15_policy_rag` | RAG over HR policies, refuse if missing | Grounding like 11; Chroma embedder from 16 |
| 16 | `problem_16_ticket_search` | Semantic search, paraphrase still hits | Queues from problem 8; ~200 ticket files |
| 17 | `problem_17_multihop` | Planner then Executor over tickets + churn | Search from 16; observe-plan-act from 9 |

## Run

```bash
# 12 (needs OPENROUTER_API_KEY)
python3 problem_12_rm_review/rm_review_demo.py

# 13
python3 problem_13_next_word/next_word_demo.py "the quick brown"

# 14
python3 problem_14_training_loss/training_loss_demo.py

# 15 (needs OPENROUTER_API_KEY)
python3 problem_15_policy_rag/policy_rag_demo.py

# 16
python3 problem_16_ticket_search/ticket_search_demo.py

# 17 (OpenRouter optional)
python3 problem_17_multihop/multihop_demo.py
```

Put `OPENROUTER_API_KEY` in a repo-root `.env` (already gitignored). Optional: `OPENROUTER_MODEL`.
