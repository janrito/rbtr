"""Path constants for rbtr.

Shared base directory for user-level storage.  Module-specific paths
(``CONFIG_PATH``, ``CREDS_PATH``, ``HISTORY_PATH``) live in the modules
that own them (``conf``, ``creds``, ``input``).

Provider-specific constants (client IDs, URLs, token paths, model lists)
live in their respective modules under ``rbtr.providers.*`` and
``rbtr.github.*``.
"""

from pathlib import Path

# User-level storage root — ~/.config/rbtr
RBTR_DIR = Path.home() / ".config" / "rbtr"
