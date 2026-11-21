"""
LLM gateway for the Generative Agents re-implementation.

Every model call made by the authors' code (gpt_structure.py) goes through this
module. It adds what the original code did not have:

  * one place to choose the chat / embedding model (gpt-3.5-turbo and the
    text-davinci models used in the paper are retired),
  * a SQLite cache of every answer, so a stopped run can resume without paying
    again,
  * token counting, a cost estimate and a hard budget stop,
  * a --mock mode that answers without any API, for free practice runs,
  * clean stops on quota / auth / budget errors (FatalLLMError), instead of the
    original behaviour of silently returning "ChatGPT ERROR" into the simulation.

The prompts themselves are NOT changed: they are the authors' templates.
"""
import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time

import numpy as np


class FatalLLMError(BaseException):
  """Stops the simulation. Inherits BaseException so the authors'
  `except Exception:` handlers do not swallow it."""


# USD per 1M tokens: (input, output). Approximate list prices; edit as needed.
PRICES = {
  "gpt-4o-mini": (0.15, 0.60),
  "gpt-4.1-mini": (0.40, 1.60),
  "gpt-4.1-nano": (0.10, 0.40),
  "gpt-4o": (2.50, 10.00),
  "text-embedding-3-small": (0.02, 0.0),
  "text-embedding-3-large": (0.13, 0.0),
  "text-embedding-ada-002": (0.10, 0.0),
}
DEFAULT_PRICE = (0.50, 1.50)

COMPLETION_SYSTEM = (
  "You are a text-completion engine. The user message is the beginning of a "
  "document. Write ONLY the text that comes directly after the last character "
  "of that document, exactly as the document's author would continue it. Do "
  "not repeat any of the given text, do not add explanations, quotes or "
  "markdown, and follow the format the document establishes.")


class Gateway:
  def __init__(self):
    self.mock = os.environ.get("GA_MOCK", "0") == "1"
    self.chat_model = os.environ.get("GA_CHAT_MODEL", "gpt-4o-mini")
    self.strong_model = os.environ.get("GA_STRONG_MODEL", self.chat_model)
    self.embed_model = os.environ.get("GA_EMBED_MODEL", "text-embedding-3-small")
    self.budget = float(os.environ.get("GA_BUDGET_USD", "25"))
    self.cache_path = os.environ.get("GA_CACHE", "llm_cache.sqlite")
    self.log_path = os.environ.get("GA_CALL_LOG", "")
    self.current_template = ""
    self._lock = threading.Lock()
    self._seen = {}
    self.stats = {"calls": 0, "cache_hits": 0, "in_tok": 0, "out_tok": 0,
                  "cost": 0.0, "by_template": {}}
    os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
    self.db = sqlite3.connect(self.cache_path, check_same_thread=False)
    self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
    self.db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    row = self.db.execute("SELECT v FROM meta WHERE k='spent'").fetchone()
    self.spent_total = float(row[0]) if row else 0.0
    self.client = None
    if not self.mock:
      if not os.environ.get("OPENAI_API_KEY"):
        raise FatalLLMError("OPENAI_API_KEY is not set (or use --mock).")
      from openai import OpenAI
      self.client = OpenAI(max_retries=0, timeout=120)

  # ------------------------------------------------------------------ cache
  def _key(self, kind, model, payload):
    base = json.dumps([kind, model, payload], sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(base.encode("utf-8")).hexdigest()
    # The n-th identical request in this process gets the n-th cached answer,
    # so retry loops (safe_generate_response) still see different samples,
    # and a resumed run replays the same answers.
    n = self._seen.get(h, 0)
    self._seen[h] = n + 1
    return f"{h}:{n}"

  def _get(self, k):
    row = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None

  def _put(self, k, v):
    self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, json.dumps(v)))
    self.db.execute("INSERT OR REPLACE INTO meta VALUES ('spent', ?)",
                    (str(self.spent_total),))
    self.db.commit()

  # ------------------------------------------------------------- accounting
  def _account(self, model, tin, tout, cached):
    price = PRICES.get(model, DEFAULT_PRICE)
    cost = 0.0 if (cached or self.mock) else (tin * price[0] + tout * price[1]) / 1e6
    t = self.stats["by_template"].setdefault(self.current_template or "?",
                                              {"calls": 0, "cost": 0.0})
    t["calls"] += 1
    t["cost"] += cost
    self.stats["calls"] += 1
    self.stats["cache_hits"] += int(cached)
    self.stats["in_tok"] += tin
    self.stats["out_tok"] += tout
    self.stats["cost"] += cost
    self.spent_total += cost
    if self.spent_total > self.budget:
      raise FatalLLMError(
        f"Budget reached: ${self.spent_total:.2f} > ${self.budget:.2f}. "
        "Raise it with --budget to continue; finished work is cached.")

  def _log(self, record):
    if self.log_path:
      with open(self.log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

  # ---------------------------------------------------------------- calling
  def _call_api(self, fn):
    delay = 2.0
    for attempt in range(8):
      try:
        return fn()
      except Exception as e:  # openai errors
        name = type(e).__name__
        msg = str(e)
        if name in ("AuthenticationError", "PermissionDeniedError",
                    "NotFoundError", "BadRequestError") and "context" not in msg:
          raise FatalLLMError(f"{name}: {msg}")
        if name == "RateLimitError" and ("insufficient_quota" in msg
                                         or "per day" in msg or "RPD" in msg):
          raise FatalLLMError(f"Quota or daily limit reached: {msg}")
        if attempt == 7:
          raise FatalLLMError(f"API kept failing: {name}: {msg}")
        time.sleep(delay + random.random())
        delay = min(delay * 2, 60)

  def chat(self, prompt, *, completion=False, temperature=None, max_tokens=None,
           stop=None, top_p=None, model=None):
    model = model or self.chat_model
    messages = []
    if completion:
      messages.append({"role": "system", "content": COMPLETION_SYSTEM})
    messages.append({"role": "user", "content": prompt})
    params = {"temperature": temperature, "max_tokens": max_tokens,
              "stop": stop, "top_p": top_p}
    params = {k: v for k, v in params.items() if v is not None}
    with self._lock:
      k = self._key("chat", model, [messages, params, self.mock])
      hit = self._get(k)
      if hit is not None:
        self._account(model, hit["in"], hit["out"], cached=True)
        return hit["text"]
      if self.mock:
        text = mock_answer(self.current_template, prompt, completion)
        tin, tout = len(prompt) // 4, len(text) // 4
      else:
        if isinstance(stop, list):
          stop = stop[:4]
        kwargs = dict(model=model, messages=messages)
        if "temperature" in params:
          kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
          kwargs["top_p"] = params["top_p"]
        if "max_tokens" in params:
          kwargs["max_tokens"] = params["max_tokens"]
        if stop:
          kwargs["stop"] = stop
        resp = self._call_api(lambda: self.client.chat.completions.create(**kwargs))
        text = resp.choices[0].message.content or ""
        tin = resp.usage.prompt_tokens if resp.usage else len(prompt) // 4
        tout = resp.usage.completion_tokens if resp.usage else len(text) // 4
      self._account(model, tin, tout, cached=False)
      self._put(k, {"text": text, "in": tin, "out": tout})
      self._log({"template": self.current_template, "model": model,
                 "completion": completion, "prompt": prompt, "output": text})
      return text

  def embed(self, text):
    model = self.embed_model
    with self._lock:
      h = hashlib.sha256(json.dumps([model, text, self.mock]).encode()).hexdigest()
      k = f"emb:{h}"
      hit = self._get(k)
      if hit is not None:
        return hit["v"]
      if self.mock:
        v = mock_embedding(text)
        tin = 0
      else:
        resp = self._call_api(lambda: self.client.embeddings.create(
          input=[text], model=model))
        v = resp.data[0].embedding
        tin = resp.usage.total_tokens if resp.usage else len(text) // 4
      self._account(model, tin, 0, cached=False)
      self._put(k, {"v": v})
      return v

  def summary(self):
    s = self.stats
    return (f"LLM calls: {s['calls']} (cache hits {s['cache_hits']}), "
            f"tokens in/out: {s['in_tok']}/{s['out_tok']}, "
            f"cost this session: ${s['cost']:.3f}, total spent: ${self.spent_total:.3f}")


_GATEWAY = None


def gateway():
  global _GATEWAY
  if _GATEWAY is None:
    _GATEWAY = Gateway()
  return _GATEWAY


# =========================================================================
#                                MOCK MODE
# Free, offline answers in the format each prompt template expects. They are
# only meant to exercise the whole pipeline end to end, not to be realistic.
# =========================================================================
def mock_embedding(text, dim=256):
  v = np.zeros(dim)
  for w in re.findall(r"[a-z]+", text.lower()):
    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
  n = np.linalg.norm(v)
  return list(v / n) if n else list(np.ones(dim) / np.sqrt(dim))


def _rng(prompt):
  return random.Random(int(hashlib.md5(prompt.encode()).hexdigest(), 16))


def _braces(prompt, after):
  """Options inside {...} on the last line containing `after`."""
  idx = prompt.rfind(after)
  if idx < 0:
    return []
  m = re.search(r"\{([^{}]*)\}", prompt[idx:])
  if not m:
    return []
  return [o.strip() for o in m.group(1).split(",") if o.strip()]


def mock_answer(template, prompt, completion):
  t = os.path.basename(template or "")
  r = _rng(prompt)

  # ChatGPT-style prompts carry their own example output: echo it.
  if not completion:
    m = re.search(r'Example output json:\n(\{.*\})\s*$', prompt, re.S)
    if m:
      return m.group(1)
    return '{"output": "ok"}'

  if t.startswith("wake_up_hour"):
    return f"{r.choice([6, 7, 8])}am"
  if t.startswith("daily_planning"):
    return (" eat breakfast at 8:00 am, 3) work from 9:00 am to 12:00 pm, "
            "4) have lunch at 12:00 pm, 5) work from 1:00 pm to 5:00 pm, "
            "6) have dinner at 6:00 pm, 7) relax from 7:00 pm to 10:00 pm, "
            "8) go to bed at 11:00 pm, 9)")
  if t.startswith("generate_hourly_schedule"):
    hour = re.findall(r"-- (\d\d:\d\d [AP]M)\] Activity: \S+ is\s*$", prompt)
    h = hour[-1] if hour else "12:00 PM"
    hh = int(h[:2]) % 12 + (12 if h.endswith("PM") else 0)
    if hh < 7 or hh >= 23:
      return " sleeping"
    return " " + {7: "having breakfast", 12: "having lunch",
                  18: "having dinner"}.get(hh, r.choice(
                    ["working", "reading a book", "taking a walk",
                     "chatting with friends", "working on a project"]))
  if t.startswith("task_decomp"):
    total = int(re.findall(r"\(total duration in minutes (\d+)\)", prompt)[-1])
    name = re.findall(r"\n1\) (.+?) is\s*$", prompt)
    name = name[0] if name else "They"
    n = max(1, min(3, total // 5))
    parts = [total // n // 5 * 5] * n
    parts[-1] += total - sum(parts)
    left = total
    lines = []
    for i, d in enumerate(parts):
      left -= d
      step = ["getting started", "doing the main part", "wrapping up"][i % 3]
      prefix = "" if i == 0 else f"{i + 1}) {name} is "
      lines.append(f"{prefix}{step}. (duration in minutes: {d}, minutes left: {left})")
    return " " + "\n".join(lines)
  if t.startswith("action_location_sector"):
    opts = _braces(prompt, "Area options:")
    cur = re.findall(r"is currently in \{([^}]*)\}", prompt)
    choice = cur[-1] if cur and cur[-1] in opts else (r.choice(opts) if opts else "kitchen")
    return choice + "}"
  if t.startswith("action_location_object"):
    opts = _braces(prompt, "MUST pick one of")
    return (r.choice(opts) if opts else "kitchen") + "}"
  if t.startswith("action_object"):
    opts = _braces(prompt, "Objects available:")
    return " " + (r.choice(opts) if opts else "bed")
  if t.startswith("generate_event_triple"):
    return " is, busy)"
  if t.startswith("new_decomp_schedule"):
    return ""  # validator fails -> authors' fail-safe schedule is used
  if t.startswith("decide_to_talk"):
    return " They are both free.\nAnswer in yes or no: " + r.choice(["yes", "no", "no"])
  if t.startswith("decide_to_react"):
    return " They should continue.\nAnswer: Option 2"
  if t.startswith("generate_focal_pt"):
    return " What is important to them?\n2) Who do they spend time with?\n3) What are they working on?"
  if t.startswith("insight_and_evidence"):
    return " They value their routine (because of 1, 2)\n2. They care about the people around them (because of 1, 3)"
  if t.startswith("whisper_inner_thought") or t.startswith("planning_thought") \
     or t.startswith("memo_on_convo") or t.startswith("convo_to_thoughts") \
     or t.startswith("keyword_to_thoughts"):
    return 'They had a pleasant chat."'
  if t.startswith("get_keywords"):
    return " daily life, routine"
  if t.startswith("generate_next_convo_line"):
    return 'Hello! How is your day going?"'
  return " ok"
