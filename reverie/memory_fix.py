"""
Switches for our memory fixes (master's contribution, not part of the paper's
code). All are OFF by default, so with no switches the simulation is exactly
the authors' code. Set with `main.py ... --memory-fix A` (comma-separated).

  A  recency = decay ** (game hours since the memory was last used), as the
     paper describes (Sec. 4.1), instead of decay ** (place in a list sorted
     oldest-first), which also ranked the most recently used memory lowest.
"""
import os

IMPLEMENTED = {"A"}


def parse(value):
  """'a, A,' -> {'A'}; exits on unknown or not-yet-built letters."""
  fixes = {x.strip().upper() for x in (value or "").split(",") if x.strip()}
  bad = fixes - IMPLEMENTED
  if bad:
    raise SystemExit(f"Unknown or not yet implemented memory fix: {', '.join(sorted(bad))}. "
                     f"Available: {', '.join(sorted(IMPLEMENTED))}")
  return fixes


def enabled(name):
  return name in parse(os.environ.get("GA_MEMORY_FIX", ""))
