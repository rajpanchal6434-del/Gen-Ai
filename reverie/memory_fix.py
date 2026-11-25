"""
Our memory fixes (master's contribution, not part of the paper's code).

All switches are OFF by default, so with no switches the simulation is exactly
the authors' code. Switch them on with `main.py ... --memory-fix A,B,C,D`.

  A  Recency by the clock. recency = decay ** (game hours since the memory was
     last used), as the paper describes (Sec. 4.1). The authors' code uses
     decay ** (place in a list sorted oldest-first), which ignores time and
     gives the most recently used memory the LOWEST score.
     -> persona/cognitive_modules/retrieve.py: extract_recency_hours

  B  Keyword score. Names, dates, times and numbers in the question are
     matched word for word against each memory and added as a fourth score
     next to recency + importance + relevance, so "What do you know about
     Klaus?" reliably finds the memories that name Klaus.
     -> question_keywords, keyword_scores (used by retrieve.new_retrieve)

  C  Fact-check. Before a line is spoken (in conversations between agents and
     in interviews), the AI model checks every claim in it against the
     memories the line was based on plus the agent's own background. Claims
     with no support are removed.
     -> fact_check (used by converse.generate_one_utterance and main.interview)

  D  Facts from conversations. When a conversation ends, the facts in it are
     saved as separate memories ("the party is on Feb 14 at 5 pm"), each with
     its own embedding and a link to the conversation. The authors' code keeps
     only a short summary searchable, so details such as times are lost.
     -> convo_facts, add_convo_facts (used by reflect.reflect),
        backfill_convo_facts (Test 1: facts for a run made without D)
"""
import datetime
import json
import os
import re

IMPLEMENTED = {"A", "B", "C", "D"}

# Fix B: weight of the keyword score. Relevance has weight 3, importance 2 and
# recency 0.5 in the authors' code; a full keyword match counts half as much as
# a perfect meaning match.
KEYWORD_WEIGHT = 1.5

# Fix D: marks memories made by fix D, so they are never added twice.
FACT_TAG = "convo fact"


def parse(value):
  """'a, B,' -> {'A', 'B'}; exits on unknown letters."""
  fixes = {x.strip().upper() for x in (value or "").split(",") if x.strip()}
  bad = fixes - IMPLEMENTED
  if bad:
    raise SystemExit(f"Unknown memory fix: {', '.join(sorted(bad))}. "
                     f"Available: {', '.join(sorted(IMPLEMENTED))}")
  return fixes


def enabled(name):
  return name in parse(os.environ.get("GA_MEMORY_FIX", ""))


def _llm_json(template, prompt):
  """One AI call that must answer with a JSON object; None if it does not."""
  from llm_gateway import gateway
  gw = gateway()
  previous = gw.current_template
  gw.current_template = template
  try:
    out = gw.chat(prompt, temperature=0)
  finally:
    gw.current_template = previous
  m = re.search(r"\{.*\}", out or "", re.S)
  if not m:
    return None
  try:
    return json.loads(m.group(0))
  except ValueError:
    return None


def _is_mock():
  from llm_gateway import gateway
  return gateway().mock


# ------------------------------------------------------------------ fix B
_NOT_KEYWORDS = {
  "i", "i'm", "i've", "i'd", "you", "your", "we", "they", "he", "she", "it",
  "do", "does", "did", "what", "who", "whom", "when", "where", "why", "how",
  "which", "is", "are", "was", "were", "am", "be", "have", "has", "had",
  "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
  "with", "about", "know", "tell", "can", "could", "would", "will", "there",
  "this", "that", "these", "those", "any", "anything", "hello", "hi", "yes",
  "no", "oh", "well", "so", "if", "my", "me", "our", "their", "day", "today",
}
_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}(?::\d{2})\b", re.I)
_DATE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b", re.I)


def question_keywords(text):
  """Exact things worth matching word for word: capitalised names and places,
  dates, times and numbers. 'Is the party at Hobbs Cafe at 5 pm?' ->
  ['hobbs', 'cafe', '5 pm']."""
  text = text or ""
  found = [m.group(0).lower() for m in _DATE.finditer(text)]
  found += [re.sub(r"\s+", " ", m.group(0).lower()) for m in _TIME.finditer(text)]
  rest = _TIME.sub(" ", _DATE.sub(" ", text))
  for word in re.findall(r"[A-Za-z][A-Za-z'\-]*|\d+", rest):
    w = word.lower()
    if w.endswith("'s"):
      w = w[:-2]
    if w in _NOT_KEYWORDS or len(w) < 2:
      continue
    if word[0].isupper() or word.isdigit():
      found.append(w)
  seen, out = set(), []
  for w in found:
    if w not in seen:
      seen.add(w)
      out.append(w)
  return out


def keyword_scores(nodes, focal_pt):
  """node_id -> share of the question's keywords found in that memory (0-1)."""
  keys = question_keywords(focal_pt)
  out = {}
  for node in nodes:
    if not keys:
      out[node.node_id] = 0.0
      continue
    text = " ".join(str(x) for x in (node.description, node.embedding_key,
                                     node.subject, node.object) if x).lower()
    text = re.sub(r"\s+", " ", text)
    hits = sum(1 for k in keys if re.search(r"\b" + re.escape(k) + r"\b", text))
    out[node.node_id] = hits / len(keys)
  return out


# ------------------------------------------------------------------ fix C
NOT_SURE = "I'm not sure I know about that."


def _memory_lines(persona, nodes, limit=40):
  """[0] = the agent's own background, [1..] = the retrieved memories."""
  lines = [f"[0] Background of {persona.scratch.name}: "
           + " ".join(persona.scratch.get_str_iss().split())]
  seen = set()
  for n in nodes:
    key = n.embedding_key or n.description
    if key and key not in seen:
      seen.add(key)
      lines.append(f"[{len(lines)}] {key}")
    if len(lines) > limit:
      break
  return lines


def _sentences(line):
  """Split into sentences; a very short one ("Yes!", "Oh, really?") is joined
  to the next, so "Yes! I'm excited about the party." is checked as one claim
  instead of a bare "Yes!" that says nothing on its own."""
  parts = [p for p in (x.strip() for x in re.split(r"(?<=[.!?])\s+", line.strip())) if p]
  out, carry = [], ""
  for p in parts:
    p = f"{carry} {p}".strip() if carry else p
    carry = ""
    if len(p.split()) <= 3:
      carry = p
    else:
      out.append(p)
  if carry:
    if out:
      out[-1] = f"{out[-1]} {carry}"
    else:
      out.append(carry)
  return out


_GENERIC = _NOT_KEYWORDS | {
  "heard", "hear", "known", "knows", "met", "meet", "remember", "think",
  "sounds", "sound", "really", "yes", "yeah", "sure", "great", "good", "fun",
  "people", "something", "someone", "thing", "things", "just", "very", "also",
  "been", "being", "some", "like", "about", "aware", "familiar", "know",
}


def _content_words(text):
  return {w for w in re.findall(r"[a-z][a-z'\-]{2,}", (text or "").lower())
          if w.rstrip("'s") not in _GENERIC and w not in _GENERIC}


def _really_supports(memory, sentence, context):
  """A cited memory must share a meaningful word with the claim or with what
  it answers; otherwise the citation is a guess and does not count."""
  mem = _content_words(memory)
  return bool(mem & (_content_words(sentence) | _content_words(context)))


def fact_check(persona, line, nodes, context=""):
  """Return `line` without the sentences that make a claim no memory supports.

  The AI model only LABELS each sentence (claim or not, and which memory
  supports it); the code then removes every claim without a memory number.
  So the rule "every claim must cite a memory" is enforced by code, not left
  to the model's rewrite. Greetings, questions, feelings, opinions and saying
  you do not know something are not claims and are always kept. If nothing is
  left, the agent says it is not sure. `context` is what the line answers, so
  "Yes, I heard about it!" can be checked. On any problem the original line
  is returned unchanged, so this can never break a run."""
  sents = _sentences(line or "")
  if not sents:
    return line
  memories = _memory_lines(persona, nodes)
  numbered = "\n".join(f"S{i + 1}: {s}" for i, s in enumerate(sents))
  replying_to = (f"What {persona.scratch.name} is replying to:\n{context.strip()}\n\n"
                 if context and context.strip() else "")
  prompt = (
    f"Memories of {persona.scratch.name}:\n" + "\n".join(memories) + "\n\n"
    f"{replying_to}"
    f"What {persona.scratch.name} is about to say, sentence by sentence:\n{numbered}\n\n"
    "For each sentence decide:\n"
    "- \"claim\": it states a fact about people, events, plans, places, dates "
    "or times, OR says the speaker knows, has met, has heard of or remembers "
    "a person or event (\"Yes, I know Kane\", \"I heard about the party\").\n"
    "- \"other\": greetings, questions, feelings, opinions, suggestions, or "
    "saying the speaker does NOT know or has not heard of something.\n"
    "For each claim, give the number of the memory that states it (0 = the "
    "background), or null if no memory states it. Being told about something "
    "in the question does not count as a memory. Do not guess: a memory only "
    "supports a claim if it says the same thing.\n"
    'Answer with JSON only: {"sentences": [{"s": 1, "kind": "claim", "memory": 3}, '
    '{"s": 2, "kind": "other", "memory": null}]}')
  out = _llm_json("memory_fix/fact_check", prompt)
  if _is_mock() or not isinstance(out, dict) or not isinstance(out.get("sentences"), list):
    return line
  drop = set()
  for item in out["sentences"]:
    if not isinstance(item, dict):
      continue
    try:
      i = int(item.get("s")) - 1
    except (TypeError, ValueError):
      continue
    if not (0 <= i < len(sents)) or item.get("kind") != "claim":
      continue
    mem = item.get("memory")
    cited = (isinstance(mem, int) and not isinstance(mem, bool)
             and 0 <= mem < len(memories)
             and _really_supports(memories[mem], sents[i], context))
    if not cited:
      drop.add(i)
  if not drop:
    return line
  kept = [s for i, s in enumerate(sents) if i not in drop]
  # The first sentence is the direct answer ("Yes, I know him!"). If it had
  # no support, what follows would still imply "yes", so say so honestly.
  if 0 in drop or not kept or all(s.endswith("?") for s in kept):
    return " ".join([NOT_SURE] + kept)
  return " ".join(kept)


# ------------------------------------------------------------------ fix D
def convo_facts(persona, all_utt):
  """Facts the persona learned in a conversation, one short sentence each."""
  if not all_utt or not all_utt.strip():
    return []
  prompt = (
    f"Conversation that {persona.scratch.name} took part in:\n{all_utt}\n"
    f"List the facts {persona.scratch.name} learned or agreed to in this "
    "conversation: plans, events, invitations, times, dates, places, and "
    "facts about people. Write each as one short, self-contained sentence "
    "with full names, the place and the date or time if they were said "
    "(e.g. \"Isabella Rodriguez is hosting a Valentine's Day party at Hobbs "
    "Cafe on February 14 at 5 pm\"). Only include what was actually said. "
    "Leave out greetings and small talk. At most 6 facts.\n"
    "For each fact give who it is mainly about and how important it is to "
    f"{persona.scratch.name} from 1 (mundane) to 10 (life-changing).\n"
    'Answer with JSON only: {"facts": [{"fact": "...", "about": "<full name>", "importance": 5}]}')
  out = _llm_json("memory_fix/convo_facts", prompt)
  if _is_mock():
    return [{"fact": f"{persona.scratch.name} had a conversation", "about":
             persona.scratch.name, "importance": 2}]
  facts = out.get("facts") if isinstance(out, dict) else None
  clean = []
  for f in facts or []:
    if not isinstance(f, dict) or not isinstance(f.get("fact"), str):
      continue
    text = f["fact"].strip()
    if not text:
      continue
    try:
      imp = int(f.get("importance", 4))
    except (TypeError, ValueError):
      imp = 4
    about = f.get("about") if isinstance(f.get("about"), str) else ""
    clean.append({"fact": text, "about": about.strip(), "importance": max(1, min(10, imp))})
  return clean[:6]


def add_convo_facts(persona, all_utt, chat_node_id, created):
  """Save each fact as its own memory (a thought), linked to the chat."""
  from persona.prompt_template.gpt_structure import get_embedding
  expiration = created + datetime.timedelta(days=30)
  added = []
  for f in convo_facts(persona, all_utt):
    s = f["about"] or persona.scratch.name
    keywords = {s, persona.scratch.name, FACT_TAG}
    node = persona.a_mem.add_thought(
      created, expiration, s, "is", f["fact"], f["fact"], keywords,
      f["importance"], (f["fact"], get_embedding(f["fact"])),
      [chat_node_id] if chat_node_id else [])
    added.append(node)
  return added


def backfill_convo_facts(persona):
  """Test 1: add fix-D facts for conversations saved by a run made without D.
  Changes memory in this process only; nothing is written to disk."""
  done = set()
  for n in persona.a_mem.seq_thought:
    if FACT_TAG in (n.keywords or ()):
      done.update(n.filling or [])
  added = 0
  for chat in reversed(persona.a_mem.seq_chat):  # oldest first
    if chat.node_id in done or not chat.filling:
      continue
    all_utt = "".join(f"{row[0]}: {row[1]}\n" for row in chat.filling
                      if isinstance(row, (list, tuple)) and len(row) == 2)
    added += len(add_convo_facts(persona, all_utt, chat.node_id, chat.created))
  return added
