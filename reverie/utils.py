"""
utils.py -- the authors' code expects this file (it is git-ignored in their
repo because it held the API key). Paths are set by main.py through
environment variables; the API key is read by the OpenAI SDK from
OPENAI_API_KEY and is never stored here.
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AUTHORS = os.environ.get(
  "GA_AUTHORS_DIR", os.path.join(_ROOT, "external", "generative_agents"))

openai_api_key = None  # unused; the SDK reads OPENAI_API_KEY
key_owner = ""

maze_assets_loc = os.path.join(
  _AUTHORS, "environment", "frontend_server", "static_dirs", "assets")
env_matrix = os.path.join(maze_assets_loc, "the_ville", "matrix")
env_visuals = os.path.join(maze_assets_loc, "the_ville", "visuals")

fs_storage = os.environ.get("GA_STORAGE", os.path.join(_ROOT, "runs"))
fs_temp_storage = os.path.join(fs_storage, "_temp")
base_storage = os.path.join(
  _AUTHORS, "environment", "frontend_server", "storage")

collision_block_id = "32125"

# Verbose logging of every cognitive function (the authors default to True).
debug = os.environ.get("GA_DEBUG", "0") == "1"
