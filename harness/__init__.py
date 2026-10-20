"""The frozen experiment harness. See CLAUDE.md before touching anything in here.

Importing the package points MTAH_DATA_DIR at the main checkout's data/ (unless already set), so
games.py finds the match data from any git worktree, not just the main one.
"""

from harness import paths  # noqa: F401  (side effect: sets MTAH_DATA_DIR, extends sys.path)
