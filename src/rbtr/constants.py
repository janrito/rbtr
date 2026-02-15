"""All constants for rbtr live here."""

from pathlib import Path

# GitHub Oauth App (device flow authentication)
GITHUB_CLIENT_ID = "Ov23li4OTCYyo2YNwAuk"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_OAUTH_URL = "https://github.com/login/oauth/access_token"

# User-level storage (not repo-level)
TOKEN_PATH = Path.home() / ".config" / "rbtr" / "token"
HISTORY_PATH = Path.home() / ".config" / "rbtr" / "history"

# API limits
MAX_BRANCHES = 30  # Cap branch listing to avoid excessive API calls
GITHUB_TIMEOUT = 10  # Seconds — prevents API calls from blocking shutdown

# TUI
SHELL_MAX_LINES = 25  # Truncate shell output in the panel after this many lines
SHELL_MAX_COMPLETIONS = 15  # Cap shell autocomplete suggestions
SHELL_COMPLETION_TIMEOUT = 2.0  # Hard cap (seconds) for bash completion subprocess
DOUBLE_CTRL_C_WINDOW = 0.5  # Seconds — two Ctrl-C within this window exits rbtr
POLL_INTERVAL = 1 / 30  # Seconds — main-loop poll rate (keystroke display latency)
REFRESH_PER_SECOND = 30  # Rich Live repaint rate
