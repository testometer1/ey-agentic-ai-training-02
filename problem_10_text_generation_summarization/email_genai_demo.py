#!/usr/bin/env python3
"""Draft a customer reply + summarise the email via OpenRouter.

  export OPENROUTER_API_KEY=...
  python3 email_genai_demo.py
  python3 email_genai_demo.py < sample_customer_email.txt
"""

import json
import os
import sys
import urllib.error
import urllib.request


def load_env():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(os.getcwd(), ".env"), os.path.join(here, ".env"), os.path.join(here, "..", ".env")]:
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def read_email():
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Paste the email. Blank line or Ctrl-D when done.")
    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "" and lines:
                break
            if line.strip():
                lines.append(line)
    except (EOFError, KeyboardInterrupt):
        if not lines:
            sys.exit("cancelled")
    return "\n".join(lines).strip()


def chat(messages, key, model):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 500}).encode()
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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"network error: {e}")
    return data["choices"][0]["message"]["content"].strip()


load_env()
key = os.environ.get("OPENROUTER_API_KEY", "").strip()
model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
if not key:
    sys.exit("set OPENROUTER_API_KEY (or put it in a .env file)")

email = read_email()
if not email:
    sys.exit("no email text")

print("\n--- original ---\n" + email)
print(f"\ncalling {model} ...")

reply = chat(
    [
        {"role": "system", "content": "Short professional support reply for an AI-agent / retrieval team. Don't invent ticket numbers."},
        {"role": "user", "content": "Draft a reply to this customer email:\n\n" + email},
    ],
    key,
    model,
)
summary = chat(
    [
        {"role": "system", "content": "Summarise the email in 2-4 sentences. Don't add facts that aren't there."},
        {"role": "user", "content": "Summarise:\n\n" + email},
    ],
    key,
    model,
)

print("\n--- reply ---\n" + reply)
print("\n--- summary ---\n" + summary)
