"""
Generative Agents (Park et al., UIST 2023) -- re-implementation runner.

  python main.py setup                         download the authors' Smallville data
  python main.py run --sim run1 --hours 2      run (or resume) a simulation
  python main.py run --sim run1 --mock         free practice run, no API calls
  python main.py run --sim noref --ablation reflection
  python main.py status --sim run1
  python main.py interview --sim run1 --persona "Isabella Rodriguez" \
                 --question "Are you planning anything for Valentine's Day?"

The simulation itself is the authors' code (reverie/), ported to Python 3.15
and the current OpenAI API; see NOTICE and README for every change.
"""
import argparse
import contextlib
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "reverie")
AUTHORS = os.path.join(ROOT, "external", "generative_agents")
AUTHORS_REPO = "https://github.com/joonspk-research/generative_agents.git"
SPARSE_PATHS = [
  "environment/frontend_server/static_dirs/assets/the_ville/matrix",
  "environment/frontend_server/static_dirs/assets/the_ville/agent_history_init_n25.csv",
  "environment/frontend_server/static_dirs/assets/the_ville/agent_history_init_n3.csv",
  "environment/frontend_server/storage/base_the_ville_n25",
  "environment/frontend_server/storage/base_the_ville_isabella_maria_klaus",
]
VIEWER_ASSETS = "environment/frontend_server/static_dirs"  # tiles, sprites (~38 MB)
BASE_SIM = "base_the_ville_n25"
STEPS_PER_HOUR = 360  # 10 game-seconds per step
PAPER_REFLECTION_THRESHOLD = 150  # paper Sec. 4.2 (authors' files use 250)


def log(msg):
  print(msg, file=sys.__stderr__, flush=True)


# --------------------------------------------------------------------- setup
def cmd_setup(args):
  paths = list(SPARSE_PATHS)
  if args.viewer:
    paths.append(VIEWER_ASSETS)
  if not os.path.exists(os.path.join(AUTHORS, ".git")):
    os.makedirs(os.path.dirname(AUTHORS), exist_ok=True)
    log("Downloading the authors' Smallville data (sparse clone)...")
    subprocess.check_call(["git", "clone", "--depth", "1", "--filter=blob:none",
                           "--sparse", AUTHORS_REPO, AUTHORS])
  subprocess.check_call(["git", "-C", AUTHORS, "sparse-checkout", "set",
                         "--no-cone"] + ["/" + p for p in paths])
  log(f"Authors' data ready in {AUTHORS}" + (" (with viewer art)" if args.viewer else ""))


def configure_env(args):
  runs = os.path.join(ROOT, "runs")
  os.environ["GA_STORAGE"] = runs
  os.environ["GA_AUTHORS_DIR"] = AUTHORS
  os.environ["GA_CACHE"] = os.path.join(runs, "llm_cache.sqlite")
  if getattr(args, "mock", False):
    os.environ["GA_MOCK"] = "1"
  if getattr(args, "model", None):
    os.environ["GA_CHAT_MODEL"] = args.model
  if getattr(args, "embed_model", None):
    os.environ["GA_EMBED_MODEL"] = args.embed_model
  if getattr(args, "budget", None) is not None:
    os.environ["GA_BUDGET_USD"] = str(args.budget)
  set_memory_fix(getattr(args, "memory_fix", ""))
  os.makedirs(runs, exist_ok=True)
  # The authors' modules use cwd-relative paths and flat imports.
  os.chdir(BACKEND)
  sys.path.insert(0, BACKEND)
  base_dst = os.path.join(runs, BASE_SIM)
  if not os.path.exists(base_dst):
    src = os.path.join(AUTHORS, "environment", "frontend_server", "storage", BASE_SIM)
    if not os.path.exists(src):
      sys.exit("Authors' data missing: run `python main.py setup` first.")
    shutil.copytree(src, base_dst)
  return runs


def set_memory_fix(value):
  """Our memory fixes (see reverie/memory_fix.py); '' = the authors' code."""
  sys.path.insert(0, BACKEND)
  import memory_fix
  fixes = sorted(memory_fix.parse(value))
  os.environ["GA_MEMORY_FIX"] = ",".join(fixes)
  return ",".join(fixes)


def load_server(sim):
  from reverie import ReverieServer
  return ReverieServer(BASE_SIM, sim)


def sim_folder(runs, sim):
  return os.path.join(runs, sim)


def read_extra(folder):
  p = os.path.join(folder, "reverie", "run_config.json")
  return json.load(open(p)) if os.path.exists(p) else {}


def write_extra(folder, cfg):
  with open(os.path.join(folder, "reverie", "run_config.json"), "w") as f:
    json.dump(cfg, f, indent=2)


# ----------------------------------------------------------------------- run
def cmd_run(args):
  runs = configure_env(args)
  folder = sim_folder(runs, args.sim)
  console = open(os.path.join(runs, f"{args.sim}.console.log"), "a", encoding="utf-8")
  os.environ["GA_CALL_LOG"] = os.path.join(runs, f"{args.sim}.llm_calls.jsonl")

  with contextlib.redirect_stdout(console):
    from llm_gateway import gateway, FatalLLMError
    gw = gateway()
    rs = load_server(args.sim)
    cfg = read_extra(folder)

    if not rs.resumed:
      cfg = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
             "chat_model": gw.chat_model, "embed_model": gw.embed_model,
             "mock": gw.mock, "ablation": args.ablation,
             "memory_fix": os.environ["GA_MEMORY_FIX"],
             "reflection_threshold": args.reflect_threshold}
      write_extra(folder, cfg)
    else:
      if cfg.get("ablation") != args.ablation:
        log(f"Note: this simulation was created with ablation={cfg.get('ablation')}; keeping it.")
      args.ablation = cfg.get("ablation")
      saved_fix = cfg.get("memory_fix", "")
      if saved_fix != os.environ["GA_MEMORY_FIX"]:
        log(f"Note: this simulation was created with memory fix '{saved_fix or 'none'}'; keeping it.")
        set_memory_fix(saved_fix)
      # Drop movement lines written after the last save (they will be replayed).
      mv = os.path.join(folder, "movement", "movements.jsonl")
      if os.path.exists(mv):
        keep = [l for l in open(mv, encoding="utf-8")
                if json.loads(l)["meta"]["step"] < rs.step]
        open(mv, "w", encoding="utf-8").writelines(keep)
    rs.last_saved_step = rs.step

    if args.ablation == "reflection":
      import persona.persona as pp
      pp.reflect = lambda persona: None
      log("Ablation: reflection is switched OFF for every agent.")

    if not cfg.get("history_loaded"):
      for p in rs.personas.values():
        p.scratch.importance_trigger_max = cfg["reflection_threshold"]
        p.scratch.importance_trigger_curr = cfg["reflection_threshold"]
        p.scratch.curr_time = rs.curr_time
      log("Loading the authors' starting memories (agent_history_init_n25.csv)...")
      from global_methods import read_file_to_list
      from persona.cognitive_modules.converse import load_history_via_whisper
      from utils import maze_assets_loc
      rows = read_file_to_list(f"{maze_assets_loc}/the_ville/agent_history_init_n25.csv",
                               header=True, strip_trail=True)[1]
      whispers = []
      for row in rows:
        for w in row[1].split(";"):
          whispers.append([row[0].strip(), w.strip()])
      load_history_via_whisper(rs.personas, whispers)
      # The memories needed a timestamp; reset the clock so that each agent's
      # first move is still treated as "First day" (daily planning).
      for p in rs.personas.values():
        p.scratch.curr_time = None
      cfg["history_loaded"] = True
      write_extra(folder, cfg)
      rs.save()
      # Snapshot of the starting state, for the "before" side of the evaluation.
      snap = os.path.join(runs, f"{args.sim}__step0")
      if not os.path.exists(snap):
        shutil.copytree(folder, snap)
      log(f"Loaded {len(whispers)} starting memories for {len(rs.personas)} agents.")

    total = args.steps if args.steps else int(args.hours * STEPS_PER_HOUR)
    target = rs.step + total
    started = time.time()

    def save_now(server):
      server.save()
      old = getattr(server, "last_saved_step", None)
      server.last_saved_step = server.step
      if old not in (None, 0, server.step):
        p = os.path.join(folder, "environment", f"{old}.json")
        if os.path.exists(p):
          os.remove(p)

    def on_step(server):
      if server.step % args.save_every == 0:
        save_now(server)
      if server.step % 60 == 0 or server.step == target:
        done = total - (target - server.step)
        log(f"[{server.curr_time:%b %d %H:%M}] step {server.step} "
            f"({done}/{total}) | {gw.summary()} | {time.time() - started:.0f}s")

    rs.on_step = on_step
    log(f"Running {args.sim}: steps {rs.step} -> {target} "
        f"({total / STEPS_PER_HOUR:.2f} game hours), model={gw.chat_model}, "
        f"mock={gw.mock}, budget=${gw.budget:.2f}, "
        f"memory fix={os.environ['GA_MEMORY_FIX'] or 'none'}")
    try:
      rs.start_server(total)
      save_now(rs)
      log("Finished. " + gw.summary())
    except FatalLLMError as e:
      log(f"STOPPED: {e}\nResume with the same command; work up to step "
          f"{rs.last_saved_step} is saved and every answer is cached.")
      sys.exit(2)
    except KeyboardInterrupt:
      log(f"Interrupted. Saved up to step {rs.last_saved_step}; rerun to resume.")
      sys.exit(130)
    finally:
      report_costs(runs, args.sim, gw)


def report_costs(runs, sim, gw):
  p = os.path.join(runs, f"{sim}.costs.json")
  with open(p, "w") as f:
    json.dump({"chat_model": gw.chat_model, "embed_model": gw.embed_model,
               "spent_total_usd": gw.spent_total, **gw.stats}, f, indent=2)


# -------------------------------------------------------------------- status
def cmd_status(args):
  runs = configure_env(args)
  folder = sim_folder(runs, args.sim)
  meta = json.load(open(os.path.join(folder, "reverie", "meta.json")))
  cfg = read_extra(folder)
  log(f"{args.sim}: step {meta['step']}, game time {meta['curr_time']}, "
      f"config {cfg}")
  for name in meta["persona_names"]:
    s = json.load(open(os.path.join(folder, "personas", name, "bootstrap_memory",
                                    "scratch.json")))
    nodes = json.load(open(os.path.join(folder, "personas", name, "bootstrap_memory",
                                        "associative_memory", "nodes.json")))
    log(f"  {name:20s} memories={len(nodes):5d}  now: {s.get('act_description')}"
        f" @ {s.get('act_address')}")


# ----------------------------------------------------------------- interview
def interview(persona, question):
  """Non-interactive version of the authors' open_convo_session("analysis")."""
  from persona.cognitive_modules.retrieve import new_retrieve
  from persona.cognitive_modules.converse import (generate_summarize_ideas,
                                                  generate_next_line)
  import memory_fix
  retrieved = new_retrieve(persona, [question], 50)[question]
  summarized_idea = generate_summarize_ideas(persona, retrieved, question)
  convo = [["Interviewer", question]]
  answer = generate_next_line(persona, "Interviewer", convo, summarized_idea)
  if memory_fix.enabled("C"):  # same fact-check as in agent conversations
    answer = memory_fix.fact_check(persona, answer, retrieved,
                                   f"Interviewer: {question}")
  return answer


def cmd_interview(args):
  runs = configure_env(args)
  with contextlib.redirect_stdout(open(os.devnull, "w")):
    from llm_gateway import gateway
    rs = load_server(args.sim)
    import memory_fix
    if memory_fix.enabled("D"):
      memory_fix.backfill_convo_facts(rs.personas[args.persona])
    answer = interview(rs.personas[args.persona], args.question)
  log(f"{args.persona}: {answer}")
  log(gateway().summary())


def cmd_export_replay(args):
  """Same output as the authors' reverie/compress_sim_storage.py: one
  master_movement.json holding, for every step, only the agents whose
  position, emoji, description or chat changed."""
  runs = os.path.join(ROOT, "runs")
  folder = os.path.join(runs, args.sim)
  move_log = os.path.join(folder, "movement", "movements.jsonl")
  if not os.path.exists(move_log):
    sys.exit(f"No movement log in {folder}; run the simulation first.")
  out = os.path.join(runs, "replays", args.sim)
  if os.path.exists(out):
    shutil.rmtree(out)
  os.makedirs(out)

  meta = json.load(open(os.path.join(folder, "reverie", "meta.json")))
  master_move, last = {}, {}
  fields = ("movement", "pronunciatio", "description", "chat")
  with open(move_log, encoding="utf-8") as f:
    for line in f:
      m = json.loads(line)
      step = m["meta"]["step"]
      if step >= meta["step"] and not args.include_unsaved:
        break
      frame = {}
      for name, mv in m["persona"].items():
        cur = {k: mv.get(k) for k in fields}
        if step == 0 or cur != last.get(name):
          frame[name] = cur
          last[name] = cur
      master_move[step] = frame
  if not master_move:
    sys.exit("Nothing to export yet.")
  steps = max(master_move) + 1
  for i in range(steps):  # the viewer expects every step key
    master_move.setdefault(i, {})
  with open(os.path.join(out, "master_movement.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): master_move[k] for k in range(steps)}, f)
  meta["step"] = steps
  with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
  shutil.copytree(os.path.join(folder, "personas"), os.path.join(out, "personas"))
  log(f"Replay written to {out} ({steps} steps). Start the viewer with:\n"
      f"  python main.py viewer   then open http://127.0.0.1:8000/demo/{args.sim}/0/3/")


def cmd_viewer(args):
  manage = os.path.join(ROOT, "frontend", "manage.py")
  if not os.path.exists(os.path.join(AUTHORS, "environment", "frontend_server",
                                     "static_dirs", "assets", "characters")):
    sys.exit("The authors' art assets are missing: run `python main.py setup --viewer`.")
  log(f"Viewer on http://127.0.0.1:{args.port}/  (Ctrl+C to stop)")
  subprocess.call([sys.executable, manage, "runserver", f"127.0.0.1:{args.port}",
                   "--noreload"], cwd=os.path.join(ROOT, "frontend"))


def cmd_evaluate(args):
  runs = configure_env(args)
  sim = args.sim + ("__step0" if args.start else "")
  if not os.path.exists(os.path.join(runs, sim)):
    sys.exit(f"No simulation {sim} in {runs}")
  # Same saved run, different memory fix -> separate results (Test 1).
  tag = sim + (f"__fix-{os.environ['GA_MEMORY_FIX'].replace(',', '')}"
               if os.environ["GA_MEMORY_FIX"] else "")
  out_dir = os.path.join(ROOT, "results", tag)
  os.environ["GA_CALL_LOG"] = os.path.join(runs, f"{tag}.eval_calls.jsonl")
  with contextlib.redirect_stdout(open(os.path.join(runs, f"{tag}.eval.log"), "a",
                                       encoding="utf-8")):
    from llm_gateway import gateway, FatalLLMError
    import evaluation
    rs = load_server(sim)
    for p in rs.personas.values():  # interviews happen "now" in game time
      p.scratch.curr_time = rs.curr_time
    import memory_fix
    if memory_fix.enabled("D"):  # Test 1 on a run made without fix D
      added = sum(memory_fix.backfill_convo_facts(p) for p in rs.personas.values())
      log(f"Memory fix D: added {added} conversation facts (in memory only).")
    try:
      summary, table = evaluation.evaluate(rs.personas, os.path.join(runs, sim),
                                           interview, out_dir,
                                           do_network=not args.no_network)
    except FatalLLMError as e:
      log(f"STOPPED: {e}. Rerun to continue; answers so far are cached.")
      sys.exit(2)
  log(table)
  log(f"Saved to {out_dir}. " + gateway().summary())


# ----------------------------------------------------------------------- cli
def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  st = sub.add_parser("setup")
  st.add_argument("--viewer", action="store_true",
                  help="also download the art assets for the game viewer")

  def common(p):
    p.add_argument("--sim", required=True, help="simulation name (folder in runs/)")
    p.add_argument("--mock", action="store_true", help="offline fake LLM, costs $0")
    p.add_argument("--model", help="chat model (default gpt-4o-mini or $GA_CHAT_MODEL)")
    p.add_argument("--embed-model", help="embedding model (default text-embedding-3-small)")
    p.add_argument("--budget", type=float, default=25.0, help="stop when total spend passes this (USD)")
    p.add_argument("--memory-fix", default="",
                   help="our memory fixes to switch on, e.g. A (default: none = authors' code)")

  r = sub.add_parser("run")
  common(r)
  r.add_argument("--hours", type=float, default=48.0, help="game hours to simulate (paper: 48)")
  r.add_argument("--steps", type=int, help="steps instead of hours (1 step = 10 s)")
  r.add_argument("--ablation", choices=["reflection"], help="switch a module off")
  r.add_argument("--reflect-threshold", type=int, default=PAPER_REFLECTION_THRESHOLD)
  r.add_argument("--save-every", type=int, default=STEPS_PER_HOUR,
                 help="autosave interval in steps")
  s = sub.add_parser("status")
  common(s)
  i = sub.add_parser("interview")
  common(i)
  i.add_argument("--persona", required=True)
  i.add_argument("--question", required=True)

  e = sub.add_parser("evaluate")
  common(e)
  e.add_argument("--start", action="store_true",
                 help="evaluate the step-0 snapshot (the paper's 'before' numbers)")
  e.add_argument("--no-network", action="store_true",
                 help="skip the 600 'Do you know of X?' interviews")

  x = sub.add_parser("export-replay", help="make a replay for the viewer")
  x.add_argument("--sim", required=True)
  x.add_argument("--include-unsaved", action="store_true",
                 help="also export steps after the last autosave")
  v = sub.add_parser("viewer", help="start the authors' game viewer (Django + Phaser)")
  v.add_argument("--port", type=int, default=8000)

  args = ap.parse_args()
  {"setup": cmd_setup, "run": cmd_run, "status": cmd_status,
   "interview": cmd_interview, "evaluate": cmd_evaluate,
   "export-replay": cmd_export_replay, "viewer": cmd_viewer}[args.cmd](args)


if __name__ == "__main__":
  main()
