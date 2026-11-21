"""
Author: Joon Sung Park (joonspk@stanford.edu)
File: views.py

MODIFIED for this re-implementation: only the replay views are kept (demo and
replay_persona_state, logic unchanged); file paths come from settings
(REPLAY_DIR / STORAGE_DIR) instead of the current working directory; the
landing page lists the available replays. The removed live-simulation views
(home, replay, process_environment, update_environment, path_tester) talked
to the backend through temp files every step, which the headless backend here
does not need.
"""
import datetime
import json
import os

from django.conf import settings
from django.http import Http404
from django.shortcuts import render


def _replay_folder(sim_code):
  folder = os.path.join(settings.REPLAY_DIR, sim_code)
  if not os.path.exists(os.path.join(folder, "master_movement.json")):
    raise Http404(f"No replay '{sim_code}'. Run: python main.py export-replay --sim <name>")
  return folder


def landing(request):
  replays = []
  if os.path.isdir(settings.REPLAY_DIR):
    for name in sorted(os.listdir(settings.REPLAY_DIR)):
      meta_file = os.path.join(settings.REPLAY_DIR, name, "meta.json")
      if os.path.exists(meta_file):
        meta = json.load(open(meta_file))
        replays.append({"sim_code": name, "steps": meta.get("step", 0),
                        "curr_time": meta.get("curr_time", "")})
  return render(request, "landing/landing.html", {"replays": replays})


def demo(request, sim_code, step, play_speed="2"):
  folder = _replay_folder(sim_code)
  move_file = os.path.join(folder, "master_movement.json")
  meta_file = os.path.join(folder, "meta.json")
  step = int(step)
  play_speed_opt = {"1": 1, "2": 2, "3": 4,
                    "4": 8, "5": 16, "6": 32}
  if play_speed not in play_speed_opt: play_speed = 2
  else: play_speed = play_speed_opt[play_speed]

  # Loading the basic meta information about the simulation.
  meta = dict()
  with open(meta_file) as json_file:
    meta = json.load(json_file)

  sec_per_step = meta["sec_per_step"]
  start_datetime = datetime.datetime.strptime(meta["start_date"] + " 00:00:00",
                                              '%B %d, %Y %H:%M:%S')
  for i in range(step):
    start_datetime += datetime.timedelta(seconds=sec_per_step)
  start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")

  # Loading the movement file
  raw_all_movement = dict()
  with open(move_file) as json_file:
    raw_all_movement = json.load(json_file)
  if str(step) not in raw_all_movement:
    raise Http404(f"Step {step} is past the end of this replay "
                  f"({len(raw_all_movement) - 1} steps).")

  # Loading all names of the personas
  persona_names = []
  persona_names_set = set()
  for p in list(raw_all_movement["0"].keys()):
    persona_names += [{"original": p,
                       "underscore": p.replace(" ", "_"),
                       "initial": p[0] + p.split(" ")[-1][0]}]
    persona_names_set.add(p)

  # <all_movement> is the main movement variable that we are passing to the
  # frontend. For this demo, we send all movement information in one step.
  all_movement = dict()

  # Preparing the initial step.
  # <init_prep> sets the locations and descriptions of all agents at the
  # beginning of the demo determined by <step>.
  init_prep = dict()
  for int_key in range(step + 1):
    key = str(int_key)
    val = raw_all_movement[key]
    for p in persona_names_set:
      if p in val:
        init_prep[p] = val[p]
  persona_init_pos = dict()
  for p in persona_names_set:
    persona_init_pos[p.replace(" ", "_")] = init_prep[p]["movement"]
  all_movement[step] = init_prep

  # Finish loading <all_movement>
  for int_key in range(step + 1, len(raw_all_movement.keys())):
    all_movement[int_key] = raw_all_movement[str(int_key)]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": json.dumps(persona_init_pos),
             "all_movement": json.dumps(all_movement),
             "start_datetime": start_datetime,
             "sec_per_step": sec_per_step,
             "play_speed": play_speed,
             "mode": "demo"}
  return render(request, "demo/demo.html", context)


def replay_persona_state(request, sim_code, step, persona_name):
  step = int(step)

  persona_name_underscore = persona_name
  persona_name = " ".join(persona_name.split("_"))
  memory = os.path.join(settings.REPLAY_DIR, sim_code, "personas", persona_name,
                        "bootstrap_memory")
  if not os.path.exists(memory):
    memory = os.path.join(settings.STORAGE_DIR, sim_code, "personas", persona_name,
                          "bootstrap_memory")
  if not os.path.exists(memory):
    raise Http404(f"No saved memory for {persona_name} in {sim_code}")

  with open(os.path.join(memory, "scratch.json")) as json_file:
    scratch = json.load(json_file)

  with open(os.path.join(memory, "spatial_memory.json")) as json_file:
    spatial = json.load(json_file)

  with open(os.path.join(memory, "associative_memory", "nodes.json")) as json_file:
    associative = json.load(json_file)

  a_mem_event = []
  a_mem_chat = []
  a_mem_thought = []

  for count in range(len(associative.keys()), 0, -1):
    node_id = f"node_{str(count)}"
    node_details = associative[node_id]

    if node_details["type"] == "event":
      a_mem_event += [node_details]

    elif node_details["type"] == "chat":
      a_mem_chat += [node_details]

    elif node_details["type"] == "thought":
      a_mem_thought += [node_details]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_name": persona_name,
             "persona_name_underscore": persona_name_underscore,
             "scratch": scratch,
             "spatial": spatial,
             "a_mem_event": a_mem_event,
             "a_mem_chat": a_mem_chat,
             "a_mem_thought": a_mem_thought}
  return render(request, "persona_state/persona_state.html", context)
