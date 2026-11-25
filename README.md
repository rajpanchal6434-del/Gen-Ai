# Generative Agents: re-implementation (SENG 5080, Group 5)

A faithful re-implementation of **Generative Agents: Interactive Simulacra of
Human Behavior** (Park, O'Brien, Cai, Morris, Liang, Bernstein; UIST 2023;
[arXiv 2304.03442](https://arxiv.org/abs/2304.03442)): 25 agents living in the
authors' Smallville for two game days.

The simulation is the **authors' own code and prompts**
([joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents),
Apache-2.0), ported to Python 3.15 and the current OpenAI API. Only what no
longer runs was changed; every change is listed in [NOTICE](NOTICE).

## What is the same as the paper

| Part | Paper | Here |
| --- | --- | --- |
| Town | Smallville, 140 x 100 tiles | the authors' map files |
| Agents | 25, with seed descriptions and relationships | the authors' 25 personas + `agent_history_init_n25.csv` |
| Clock | Feb 13 2023 00:00, 10 s per step, 2 days | same (17,280 steps) |
| Memory stream, retrieval | recency (decay 0.995) + importance + relevance, all weights 1 | authors' code |
| Reflection | when summed importance > **150** | 150 (the authors' files say 250; we follow the paper) |
| Planning, reacting, dialogue | recursive plans, react / talk decisions | authors' code and prompts, word for word |
| Evaluation | diffusion, network density, party attendance | `python main.py evaluate` |

## What is different, and why

| Paper | Here | Why |
| --- | --- | --- |
| text-davinci-002/003, gpt-3.5-turbo | one current chat model (default `gpt-4o-mini`) | the old models are retired |
| text-embedding-ada-002 | `text-embedding-3-small` (configurable) | current embedding model |
| Completion prompts | same prompts, sent to a chat model told to continue the text, same stop tokens and temperatures | the completion endpoint models are gone |
| Phaser browser frontend | headless; the backend moves agents one tile per step exactly as the frontend did | graphics do not affect results |
| Human raters (100) | LLM judge for the end-to-end measures | budget |

## Quick start (Windows PowerShell)

```powershell
uv venv .venv --python 3.15         # once
uv pip install --python .venv\Scripts\python.exe -r pyproject.toml
.venv\Scripts\python.exe main.py setup                        # authors' map + agents
.venv\Scripts\python.exe main.py run --sim practice --mock --hours 2   # free, no API

$env:OPENAI_API_KEY = "sk-..."      # never commit it
.venv\Scripts\python.exe main.py run --sim full --hours 1     # real test, check cost
.venv\Scripts\python.exe main.py run --sim full               # continue to 48 game hours
.venv\Scripts\python.exe main.py run --sim noref --ablation reflection
.venv\Scripts\python.exe main.py evaluate --sim full --start  # "before" numbers
.venv\Scripts\python.exe main.py evaluate --sim full          # "after" numbers
.venv\Scripts\python.exe main.py interview --sim full --persona "Isabella Rodriguez" --question "What are you planning for Valentine's Day?"
```

## Watching the agents (the paper's game view)

The authors' Phaser replay viewer, ported to current Django:

```powershell
.venv\Scripts\python.exe main.py setup --viewer          # once: tiles and character sprites (~38 MB)
.venv\Scripts\python.exe main.py export-replay --sim full
.venv\Scripts\python.exe main.py viewer                  # http://127.0.0.1:8000/
```

Open `http://127.0.0.1:8000/demo/<sim>/<start step>/<speed>/`, for example
`/demo/full/0/3/` (speed 1 slowest to 6 fastest; start step = game hours x 360).
Click an agent's name for its memory, plans and reflections ("State Details").
If rooms appear black, the graphics card cannot load the tall interior
tilesets: add `?renderer=canvas` to the address.

* **Resume:** rerun the same command. State is saved every game hour and every
  model answer is cached in `runs/llm_cache.sqlite`, so finished work is never
  paid for twice.
* **Budget:** each run stops at `--budget` dollars (default 25) of total spend.
* **Logs:** `runs/<sim>.console.log` (the authors' debug prints),
  `runs/<sim>.llm_calls.jsonl` (every prompt and answer),
  `runs/<sim>.costs.json` (calls and cost per prompt template),
  `runs/<sim>/movement/movements.jsonl` (every agent, every step).

## Memory fixes (our contribution)

The paper's authors report that agents "failed to retrieve relevant memories"
and "fabricated embellishments" (Sec. 1). Reading their code shows four causes;
each has a fix in `reverie/memory_fix.py`, switched on with `--memory-fix`.
All are off by default, so a run without the switch is the authors' code.

| Fix | Problem in the authors' code | What the fix does |
| --- | --- | --- |
| A | Recency is `0.995 ^ place in a list sorted oldest-first`: it ignores the clock and gives the newest memory the lowest score | Recency is `0.995 ^ game hours since last used`, as the paper says (Sec. 4.1) |
| B | Relevance is meaning only, so exact names, dates and times are missed | Adds a keyword score: names, dates and times in the question matched word for word |
| C | Nothing checks what an agent says against its memories | Each line (in conversations and interviews) is fact-checked; unsupported claims are removed |
| D | A whole conversation is saved as one short summary; its details are never searched | Each fact in a conversation is saved as its own memory, linked to the conversation |

```powershell
# Recall test: does retrieval bring up the memories that hold the answer? (only question embeddings, near $0)
.venv\Scripts\python.exe main.py evaluate --sim before --recall-only
.venv\Scripts\python.exe main.py evaluate --sim before --recall-only --memory-fix A,D,B
# Test 1: same saved run, recalled with and without the fixes (cheap, no new simulation)
.venv\Scripts\python.exe main.py evaluate --sim before                      # -> results\before
.venv\Scripts\python.exe main.py evaluate --sim before --memory-fix A,B,C,D # -> results\before__fix-ABCD
# Test 2: full simulations, before vs. after
.venv\Scripts\python.exe main.py run --sim before
.venv\Scripts\python.exe main.py run --sim after --memory-fix A,B,C,D
```

On a run made without fix D, `evaluate --memory-fix ...D` first turns its saved
conversations into fact memories (in memory only, nothing is written). The
evaluation's "really knew" check reads the agent's full conversations and
ignores fix-D memories, so it gives the same answer with and without the fixes.

The recall test asks each agent who really knows a fact (the Valentine's
party, the mayor election, the party time) a question about it, retrieves the
top 50 memories as an agent would, and checks whether a memory holding the
fact is among them and at which rank. Starting background text is left out,
because retrieval cannot return it. Results go to `recall.json` / `recall.md`.

## Layout

```text
main.py            CLI: setup | run | status | interview | evaluate | export-replay | viewer
evaluation.py      paper Sec. 7 measures
frontend/          the authors' replay viewer (Django + Phaser), ported to Django 6
reverie/           the authors' backend (modified, see NOTICE)
  reverie.py         simulation loop (ReverieServer)
  maze.py            Smallville map, tiles, events
  path_finder.py     walking on the map
  llm_gateway.py     NEW: model calls, cache, cost, budget, mock
  memory_fix.py      NEW: our memory fixes A-D (off by default, --memory-fix)
  persona/
    persona.py         one agent: perceive -> retrieve -> plan -> reflect -> execute
    cognitive_modules/ perceive, retrieve, plan, reflect, execute, converse
    memory_structures/ associative memory (memory stream), spatial memory, scratch
    prompt_template/   the authors' prompts (v1, v2, v3_ChatGPT) and runners
external/          the authors' data (downloaded by `main.py setup`, not committed)
runs/              simulations, cache and logs (not committed)
results/           evaluation outputs
```
