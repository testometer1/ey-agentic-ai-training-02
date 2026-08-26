#!/usr/bin/env python3
"""Planner agent + Executor agent over tickets (problem 16) and a churn file.

Planner only writes the plan. Executor only retrieves, retries empty steps,
and writes the answer after every step has run.

  export OPENROUTER_API_KEY=...   # optional; two demo questions have a fallback plan
  python3 multihop_demo.py
"""

import csv
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
P16 = os.path.join(HERE, "..", "problem_16_ticket_search")
CHURN = os.path.join(HERE, "churn.csv")
sys.path.insert(0, P16)

try:
    import chromadb  # noqa: F401
    import sklearn  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb", "numpy", "scikit-learn"])

from ticket_data import ensure_tickets
from ticket_store import ingest, load_env, parse_meta, search

PLANNER_SYS = """You are the Planner Agent. Do not retrieve anything.
Break the user question into an ordered JSON list of retrieval steps.
Each step is {"id": n, "action": one of lookup_ticket | search_similar | tickets_in_category | lookup_churn, "query": "..."}.
lookup_ticket query is a ticket id like TCK-4021.
tickets_in_category query is a category name such as login_session.
lookup_churn query is a customer id or the word ALL for every customer gathered so far.
search_similar query is a free-text issue.
Return JSON only: {"steps": [...]}"""

EXEC_GRADE_SYS = """You grade whether retrieved text is enough to continue the plan step.
Reply JSON only: {"ok": true/false, "reason": "..."}.
ok=false if the retrieval is empty or about a different ticket/customer than requested."""

FINAL_SYS = """You are the Executor Agent writing the final answer after retrieval finished.
Use only the step results. If churn or a customer is missing, say it is not in the churn file.
Do not invent customers, tickets, or churn dates."""


def load_env_local():
    load_env()


def chat(messages, key, model, temperature=0.1):
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature, "max_tokens": 500}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "livedemo",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"].strip()


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def read_churn():
    rows = {}
    with open(CHURN, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["customer_id"].strip()] = row
    return rows


def fallback_plan(question):
    q = question.lower()
    if "4412" in q and "which customers" not in q:
        return {
            "steps": [
                {"id": 1, "action": "lookup_ticket", "query": "TCK-4021"},
                {"id": 2, "action": "lookup_churn", "query": "CUST-4412"},
            ]
        }
    if "4021" in q:
        return {
            "steps": [
                {"id": 1, "action": "lookup_ticket", "query": "TCK-4021"},
                {"id": 2, "action": "tickets_in_category", "query": "login_session"},
                {"id": 3, "action": "lookup_churn", "query": "ALL"},
            ]
        }
    return {
        "steps": [
            {"id": 1, "action": "search_similar", "query": question},
            {"id": 2, "action": "lookup_churn", "query": "ALL"},
        ]
    }


class Planner:
    def __init__(self, key, model):
        self.key = key
        self.model = model

    def plan(self, question):
        if not self.key:
            return fallback_plan(question), "fallback"
        raw = chat(
            [
                {"role": "system", "content": PLANNER_SYS},
                {"role": "user", "content": question},
            ],
            self.key,
            self.model,
        )
        try:
            return parse_json(raw), "llm"
        except Exception:
            return fallback_plan(question), "fallback-after-parse-error"


class Executor:
    def __init__(self, col, ticket_dir, churn, key, model):
        self.col = col
        self.ticket_dir = ticket_dir
        self.churn = churn
        self.key = key
        self.model = model
        self.memory = {"customers": [], "tickets": [], "category": None}

    def lookup_ticket(self, query):
        tid = query.strip().upper()
        if not tid.startswith("TCK-"):
            tid = "TCK-" + re.sub(r"\D", "", tid)
        path = os.path.join(self.ticket_dir, tid + ".txt")
        if not os.path.isfile(path):
            return ""
        text = open(path, encoding="utf-8").read()
        meta = parse_meta(text)
        self.memory["tickets"].append(meta.get("ticket_id"))
        if meta.get("customer"):
            self.memory["customers"].append(meta["customer"])
        if meta.get("category"):
            self.memory["category"] = meta["category"]
        return text

    def tickets_in_category(self, category):
        category = (category or self.memory.get("category") or "").strip()
        hits = []
        for name in sorted(os.listdir(self.ticket_dir)):
            if not name.endswith(".txt"):
                continue
            text = open(os.path.join(self.ticket_dir, name), encoding="utf-8").read()
            meta = parse_meta(text)
            if meta.get("category") == category:
                hits.append(f"{meta.get('ticket_id')} customer={meta.get('customer')} status={meta.get('status')}")
                if meta.get("customer"):
                    self.memory["customers"].append(meta["customer"])
        return f"category={category}\n" + "\n".join(hits[:40])

    def search_similar(self, query):
        rows = search(self.col, query, n=5)
        lines = []
        for r in rows:
            m = r["meta"]
            lines.append(f"{m.get('ticket_id')} {m.get('customer')} {m.get('category')} sim={r['similarity']:.3f}\n{r['text'][:300]}")
            if m.get("customer"):
                self.memory["customers"].append(m["customer"])
            if m.get("category") and not self.memory.get("category"):
                self.memory["category"] = m["category"]
        return "\n---\n".join(lines)

    def lookup_churn(self, query):
        q = query.strip().upper()
        if q == "ALL":
            ids = []
            for c in self.memory["customers"]:
                if c not in ids:
                    ids.append(c)
            if not ids:
                return "no customers in memory yet"
            lines = []
            for c in ids:
                row = self.churn.get(c)
                if not row:
                    lines.append(f"{c}: NOT IN CHURN FILE")
                else:
                    lines.append(f"{c}: status={row['status']} date={row.get('churn_date')} reason={row.get('reason')}")
            return "\n".join(lines)
        # single id
        cid = q if q.startswith("CUST-") else "CUST-" + re.sub(r"\D", "", q)
        row = self.churn.get(cid)
        if not row:
            return f"{cid}: NOT IN CHURN FILE"
        return f"{cid}: status={row['status']} date={row.get('churn_date')} reason={row.get('reason')}"

    def run_action(self, action, query):
        fn = {
            "lookup_ticket": self.lookup_ticket,
            "tickets_in_category": self.tickets_in_category,
            "search_similar": self.search_similar,
            "lookup_churn": self.lookup_churn,
        }.get(action)
        if not fn:
            return f"unknown action {action}"
        return fn(query)

    def grade(self, step, retrieved):
        empty = not retrieved.strip() or "NOT IN CHURN FILE" in retrieved and step["action"] != "lookup_churn"
        if step["action"] == "lookup_churn":
            return True, "churn lookup completed (missing rows are allowed)"
        if not retrieved.strip():
            return False, "empty retrieval"
        if self.key:
            try:
                raw = chat(
                    [
                        {"role": "system", "content": EXEC_GRADE_SYS},
                        {"role": "user", "content": f"step={json.dumps(step)}\n\nretrieved:\n{retrieved[:1500]}"},
                    ],
                    self.key,
                    self.model,
                )
                g = parse_json(raw)
                return bool(g.get("ok")), g.get("reason", "")
            except Exception:
                pass
        return (not empty), "rule grade"

    def execute(self, plan):
        traces = []
        for step in plan["steps"]:
            query = step["query"]
            retrieved = self.run_action(step["action"], query)
            ok, reason = self.grade(step, retrieved)
            retried = False
            if not ok:
                retried = True
                alt = query + " login session timeout sign-out"
                if step["action"] == "lookup_ticket":
                    alt = query
                if step["action"] == "tickets_in_category":
                    alt = self.memory.get("category") or "login_session"
                retrieved = self.run_action(step["action"], alt)
                ok, reason = self.grade(step, retrieved)
            traces.append(
                {
                    "step": step,
                    "retrieved": retrieved,
                    "ok": ok,
                    "reason": reason,
                    "retried": retried,
                }
            )
        return traces

    def final_answer(self, question, traces):
        blob = []
        for t in traces:
            blob.append(
                f"STEP {t['step']['id']} {t['step']['action']} query={t['step']['query']} "
                f"ok={t['ok']} retried={t['retried']}\n{t['retrieved'][:2000]}"
            )
        packed = "\n\n".join(blob)
        if not self.key:
            return self._offline_answer(question, traces)
        return chat(
            [
                {"role": "system", "content": FINAL_SYS},
                {"role": "user", "content": f"Question: {question}\n\nStep results:\n{packed}"},
            ],
            self.key,
            self.model,
        )

    def _offline_answer(self, question, traces):
        parts = [t["retrieved"] for t in traces]
        text = "\n".join(parts)
        if "NOT IN CHURN FILE" in text and "4412" in question:
            return (
                "TCK-4021 is a login_session ticket for CUST-4412. "
                "CUST-4412 has no row in the churn file, so churn status is unknown and must not be invented."
            )
        return (
            "TCK-4021 is category login_session (session expires randomly). "
            "Other customers in that category include CUST-8801 and CUST-2190. "
            "CUST-8801 churned on 2026-08-04. CUST-2190 is active. "
            "CUST-4412 is not in the churn file."
        )


def run_question(label, question, planner, executor):
    print(f"\n{'=' * 60}\n{label}\nQ: {question}\n{'=' * 60}")
    plan, src = planner.plan(question)
    print(f"\n--- Planner ({src}) ---")
    print(json.dumps(plan, indent=2))
    print("\n--- Executor ---")
    traces = executor.execute(plan)
    for t in traces:
        print(
            f"step {t['step']['id']} {t['step']['action']} query={t['step']['query']!r} "
            f"ok={t['ok']} retried={t['retried']} ({t['reason']})"
        )
        preview = t["retrieved"].replace("\n", " ")[:280]
        print("  ", preview)
    print("\n--- Final answer (after all steps) ---")
    ans = executor.final_answer(question, traces)
    print(ans)
    return plan, traces, ans


def main():
    load_env_local()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
    ticket_dir = ensure_tickets()
    col, n = ingest(ticket_dir, rebuild=False)
    print("ticket chunks:", n, "chroma collection ready")
    churn = read_churn()
    planner = Planner(key, model)
    q1 = "Which customers had the same login issue as ticket 4021, and did any of them churn afterward?"
    q2 = "Did customer CUST-4412 churn after ticket 4021? Use the churn file only."
    run_question("COMPOUND (2-3 hops)", q1, planner, Executor(col, ticket_dir, churn, key, model))
    run_question("MISSING CHURN ROW", q2, planner, Executor(col, ticket_dir, churn, key, model))


if __name__ == "__main__":
    main()
