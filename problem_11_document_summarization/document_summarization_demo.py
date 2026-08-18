#!/usr/bin/env python3
"""Summarise any document via OpenRouter.

  export OPENROUTER_API_KEY=...
  python3 document_summarization_demo.py sample_incident_report.txt
"""

import html.parser
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

MAX_CHARS = 12000


class TextFromHTML(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def install(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg])


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


def read_pdf(path):
    install("pypdf")
    from pypdf import PdfReader

    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def read_docx(path):
    install("docx", "python-docx")
    import docx

    return "\n".join(p.text for p in docx.Document(path).paragraphs)


def read_html(path):
    parser = TextFromHTML()
    parser.feed(open(path, encoding="utf-8", errors="replace").read())
    return " ".join(parser.parts)


def read_text(path):
    return open(path, encoding="utf-8", errors="replace").read()


def extract(path):
    ext = os.path.splitext(path)[1].lower()
    readers = {
        ".pdf": read_pdf,
        ".docx": read_docx,
        ".html": read_html,
        ".htm": read_html,
        ".txt": read_text,
        ".md": read_text,
        ".csv": read_text,
        ".json": read_text,
        ".log": read_text,
    }
    fn = readers.get(ext, read_text)
    print("reader:", fn.__name__)
    return fn(path)


def chat(messages, key, model):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 400}).encode()
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
        with urllib.request.urlopen(req, timeout=90) as resp:
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
    sys.exit("set OPENROUTER_API_KEY")

path = sys.argv[1] if len(sys.argv) > 1 else input("file path: ").strip().strip('"').strip("'")
if not path:
    sys.exit("no path")
path = os.path.abspath(os.path.expanduser(path))
if not os.path.isfile(path):
    sys.exit("not found: " + path)

print("reading", path)
text = " ".join(extract(path).split())
if not text:
    sys.exit("no text extracted")
print("chars:", len(text))
if len(text) > MAX_CHARS:
    text = text[:MAX_CHARS]
    print("truncated to", MAX_CHARS)

print("calling", model, "...")
summary = chat(
    [
        {"role": "system", "content": "Summarise in 5-8 sentences. Don't invent details."},
        {"role": "user", "content": "Summarise this document:\n\n" + text},
    ],
    key,
    model,
)
print("\n--- summary ---\n" + summary)
