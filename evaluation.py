"""
End-to-end evaluation, following the paper's Section 7 (emergent behaviour):

  * Information diffusion -- at the end, every agent is interviewed about the
    Valentine's Day party (seeded only in Isabella) and the mayor election
    (seeded only in Sam Moore). Paper: party 4% -> 52%, election 4% -> 32%.
    As in the paper, a "yes" only counts if the agent's memory stream really
    contains the information (otherwise it is a hallucination).
  * Relationship formation -- every agent is asked "Do you know of <name>?"
    about every other agent; network density = 2|E| / (|V|(|V|-1)).
    Paper: 0.167 -> 0.74.
  * Coordination -- how many agents who knew about the party were at Hobbs
    Cafe on Feb 14 between 5 and 7 pm. Paper: 5 of 12 invited came.
  * Hallucination probe -- "Do you know Kane Martinez?" (a person who does not
    exist).
  * Recall test (ours) -- for every agent whose memory really holds the party /
    election, is a memory about it among the top 50 that retrieval returns for
    the interview question? Measures the paper's "failed to retrieve relevant
    memories" directly, without the noise of the AI's wording.

Interviews use the authors' own interview pipeline (retrieve 50 memories ->
summarise -> answer), the same one as `call -- analysis` in their code.
"""
import datetime
import json
import os
import re


PARTY_Q = "Did you know there is a Valentine's Day party?"
ELECTION_Q = "Do you know who is running for mayor?"
FAKE_PERSON = "Kane Martinez"
PARTY_TIME_Q = "When and where is the Valentine's Day party?"
FACT_TAG = "convo fact"  # memories added by memory fix D (reverie/memory_fix.py)


def judge(question, answer, criterion, speaker=None):
  """LLM judge: does the answer show the agent knows <criterion>? -> bool.
  `speaker` is who answered, so "Yes, I'm running for mayor" from Sam Moore
  counts as knowing that Sam Moore is running."""
  from persona.prompt_template.gpt_structure import ChatGPT_request
  from llm_gateway import gateway
  gateway().current_template = "eval/judge"
  prompt = (f"Question asked to {speaker or 'a person'}: {question}\n"
            f"Their answer: {answer}\n\n"
            f"Does the answer show that the person knows {criterion}? "
            "Reply with exactly one word: yes or no.")
  out = ChatGPT_request(prompt).strip().lower()
  if gateway().mock:
    out = "yes" if re.search(r"\byes\b|party|mayor|know", answer.lower()) else "no"
  return out.startswith("yes")


def memory_text(persona, include_seed=True):
  """Everything the agent can know: its seed description, its memory stream
  and the full text of its conversations. Memories added by memory fix D are
  left out: they are copies of conversation text, so with or without the fix
  the agent "knows" exactly the same things and the comparison stays fair.
  The recall test leaves out the seed description: it is always in the
  prompt, so finding it is not retrieval's job."""
  s = persona.scratch
  seed = (" ".join(str(x) for x in (s.learned, s.currently, s.innate) if x)
          if include_seed else "")
  parts = []
  for n in persona.a_mem.id_to_node.values():
    if FACT_TAG in (n.keywords or ()):
      continue
    parts.append(n.description)
    if n.type == "chat" and n.filling:
      parts += [str(row[1]) for row in n.filling
                if isinstance(row, (list, tuple)) and len(row) == 2]
  return (seed + " " + " ".join(parts)).lower()


def info_diffusion(personas, interview):
  rows = []
  for name, p in personas.items():
    mem = memory_text(p)
    a1 = interview(p, PARTY_Q)
    a2 = interview(p, ELECTION_Q)
    k1 = judge(PARTY_Q, a1, "about a Valentine's Day party at Hobbs Cafe", name)
    k2 = judge(ELECTION_Q, a2, "that Sam Moore is running for mayor", name)
    rows.append({
      "agent": name,
      "party_answer": a1, "party_says_knows": k1,
      "party_in_memory": "party" in mem and "valentine" in mem,
      "election_answer": a2, "election_says_knows": k2,
      "election_in_memory": "mayor" in mem or "election" in mem,
    })
    r = rows[-1]
    if r["party_says_knows"] and r["party_in_memory"]:
      a3 = interview(p, PARTY_TIME_Q)
      r["party_time_answer"] = a3
      r["party_time_right"] = judge(PARTY_TIME_Q, a3,
                                    "that the party is on February 14 at 5 pm at Hobbs Cafe", name)
  return rows


def network(personas, interview):
  names = list(personas)
  knows = {}
  for a in names:
    for b in names:
      if a == b:
        continue
      q = f"Do you know of {b}?"
      ans = interview(personas[a], q)
      knows[(a, b)] = judge(q, ans, f"who {b} is", a)
  n = len(names)
  either = {frozenset((a, b)) for (a, b), k in knows.items() if k}
  both = {frozenset((a, b)) for (a, b), k in knows.items()
          if k and knows.get((b, a))}
  pairs = n * (n - 1) / 2
  return {"density_either": len(either) / pairs, "density_both": len(both) / pairs,
          "edges": [[a, b] for (a, b), k in knows.items() if k]}


def hallucination(personas, interview):
  q = f"Do you know {FAKE_PERSON}?"
  out = []
  for name, p in personas.items():
    ans = interview(p, q)
    out.append({"agent": name, "answer": ans,
                "claims_to_know": judge(q, ans, f"a person called {FAKE_PERSON}", name)})
  return out


def party_attendance(sim_folder):
  """Agents at Hobbs Cafe on Feb 14 2023, 5-7 pm, from the movement log."""
  path = os.path.join(sim_folder, "movement", "movements.jsonl")
  start = datetime.datetime(2023, 2, 14, 17, 0)
  end = datetime.datetime(2023, 2, 14, 19, 0)
  seen = set()
  if not os.path.exists(path):
    return []
  with open(path, encoding="utf-8") as f:
    for line in f:
      m = json.loads(line)
      t = datetime.datetime.strptime(m["meta"]["curr_time"], "%B %d, %Y, %H:%M:%S")
      if start <= t <= end:
        for name, mv in m["persona"].items():
          if "Hobbs Cafe" in (mv.get("description") or ""):
            seen.add(name)
  return sorted(seen)


RECALL_K = 50  # the interview pipeline reads the top 50 memories
# The party and "5 pm" in the same sentence: most agents' daily plan says
# "dinner at 5:00 pm", which says nothing about the party time.
_SENT = r"[^.!?\n]*"
_FIVE = r"\b5\s?(?::00\s?)?pm\b"
_PARTY = r"(?:valentine|\bparty\b)"
PARTY_AT_5 = re.compile(f"{_PARTY}{_SENT}{_FIVE}|{_FIVE}{_SENT}{_PARTY}", re.I)
RECALL_TOPICS = [
  # name, question, "really knows" check on memory_text, "this memory is about it"
  ("party", PARTY_Q, lambda m: "party" in m and "valentine" in m,
   re.compile(r"valentine|\bparty\b", re.I)),
  ("election", ELECTION_Q, lambda m: "mayor" in m or "election" in m,
   re.compile(r"mayor|election", re.I)),  # same test as "knows"
  ("party_time", PARTY_TIME_Q,
   lambda m: PARTY_AT_5.search(m) is not None, PARTY_AT_5),
]


def recall_test(personas, k=RECALL_K):
  """For each agent who really knows a topic, does retrieval put a memory about
  it in the top k for the interview question, and at what rank? No chat calls
  (one embedding per question). Every memory's last-used time is put back
  afterwards, so this does not change the interviews that follow."""
  from persona.cognitive_modules.retrieve import new_retrieve
  rows = []
  for name, p in personas.items():
    mem = memory_text(p, include_seed=False)
    saved = {n.node_id: n.last_accessed for n in p.a_mem.id_to_node.values()}
    try:
      for topic, question, knows, about in RECALL_TOPICS:
        if not knows(mem):
          continue
        got = new_retrieve(p, [question], k)[question]
        rank = next((i + 1 for i, n in enumerate(got)
                     if about.search(f"{n.embedding_key} {n.description}")), None)
        rows.append({"agent": name, "topic": topic, "found": rank is not None,
                     "rank": rank})
    finally:
      for n in p.a_mem.id_to_node.values():
        n.last_accessed = saved[n.node_id]
  return rows


def recall_summary(rows):
  out = {}
  for topic, *_ in RECALL_TOPICS:
    r = [x for x in rows if x["topic"] == topic]
    ranks = [x["rank"] for x in r if x["found"]]
    out[topic] = {"knowers": len(r), "found": len(ranks),
                  "mean_rank": round(sum(ranks) / len(ranks), 1) if ranks else None}
  return out


def recall_lines(rs):
  names = {"party": "Party memory in top 50 (of agents who know)",
           "election": "Election memory in top 50 (of agents who know)",
           "party_time": "Party-time (5 pm) memory in top 50 (of agents who know it)"}
  lines = []
  for topic, s in rs.items():
    rank = f", mean rank {s['mean_rank']}" if s["mean_rank"] is not None else ""
    lines.append(f"| {names[topic]} | - | {s['found']} of {s['knowers']}{rank} |")
  return lines


def evaluate(personas, sim_folder, interview, out_dir, do_network=True,
             recall_only=False):
  os.makedirs(out_dir, exist_ok=True)
  recall = recall_test(personas)
  rsum = recall_summary(recall)
  if recall_only:
    res = {"recall": recall, "summary": {"recall": rsum}}
    with open(os.path.join(out_dir, "recall.json"), "w", encoding="utf-8") as f:
      json.dump(res, f, indent=2, ensure_ascii=False)
    lines = ["| Measure | Paper | Ours |", "| --- | --- | --- |"] + recall_lines(rsum)
    with open(os.path.join(out_dir, "recall.md"), "w", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")
    return res["summary"], "\n".join(lines)
  res = {"recall": recall,
         "diffusion": info_diffusion(personas, interview),
         "hallucination": hallucination(personas, interview),
         "attendance": party_attendance(sim_folder)}
  if do_network:
    res["network"] = network(personas, interview)
  n = len(personas)
  d = res["diffusion"]
  party_true = [r for r in d if r["party_says_knows"] and r["party_in_memory"]]
  elect_true = [r for r in d if r["election_says_knows"] and r["election_in_memory"]]
  party_false = [r for r in d if r["party_says_knows"] and not r["party_in_memory"]]
  elect_false = [r for r in d if r["election_says_knows"] and not r["election_in_memory"]]
  party_missed = [r for r in d if not r["party_says_knows"] and r["party_in_memory"]]
  elect_missed = [r for r in d if not r["election_says_knows"] and r["election_in_memory"]]
  party_right_time = [r for r in party_true if r.get("party_time_right")]
  knew = {r["agent"] for r in party_true}
  came = [a for a in res["attendance"] if a in knew]
  fake = [r for r in res["hallucination"] if r["claims_to_know"]]
  summary = {
    "agents": n,
    "party_knows_pct": 100 * len(party_true) / n,
    "election_knows_pct": 100 * len(elect_true) / n,
    "party_hallucinated": len(party_false),
    "election_hallucinated": len(elect_false),
    "party_missed": len(party_missed),
    "election_missed": len(elect_missed),
    "party_in_memory": sum(r["party_in_memory"] for r in d),
    "election_in_memory": sum(r["election_in_memory"] for r in d),
    "party_true": len(party_true), "election_true": len(elect_true),
    "party_right_time": len(party_right_time),
    "party_knew": len(knew), "party_attended_of_knew": len(came),
    "attended_any": len(res["attendance"]),
    "fake_person_claims": len(fake),
    "recall": rsum,
  }
  if do_network:
    summary["density_either"] = res["network"]["density_either"]
    summary["density_both"] = res["network"]["density_both"]
  res["summary"] = summary
  with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

  lines = ["| Measure | Paper | Ours |", "| --- | --- | --- |",
           f"| Know about the party | 4% -> 52% | {summary['party_knows_pct']:.0f}% |",
           f"| Know about the election | 4% -> 32% | {summary['election_knows_pct']:.0f}% |",
           f"| Party attendance (of those who knew) | 5 of 12 | "
           f"{summary['party_attended_of_knew']} of {summary['party_knew']} |",
           f"| Claimed knowledge not in memory (party / election) | rare | "
           f"{summary['party_hallucinated']} / {summary['election_hallucinated']} |",
           f"| Claim to know '{FAKE_PERSON}' (does not exist) | rare | "
           f"{summary['fake_person_claims']} of {n} |",
           f"| Missed: in memory but says no (party / election) | - | "
           f"{summary['party_missed']} of {summary['party_in_memory']} / "
           f"{summary['election_missed']} of {summary['election_in_memory']} |",
           f"| Right party time and place (of those who truly know) | - | "
           f"{summary['party_right_time']} of {summary['party_true']} |"]
  lines += recall_lines(rsum)
  if do_network:
    lines.append(f"| Network density | 0.167 -> 0.74 | {summary['density_either']:.3f} "
                 f"(both-ways {summary['density_both']:.3f}) |")
  with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  return summary, "\n".join(lines)
