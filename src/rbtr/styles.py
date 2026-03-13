"""Rich theme for rbtr.

All visual styling lives in a ``Theme`` object built by
``build_theme()``.  Code that renders must reference the semantic
keys defined here -- never use inline hex colours or ad-hoc style
strings.

Only TUI modules import this file.  Engine and LLM code communicate
styling intent through the ``OutputLevel`` enum in ``events.py``;
the TUI maps levels to theme keys at render time.
"""

from rich.theme import Theme

from rbtr.config import PaletteConfig, ThemeConfig

# ── String constants (theme key names) ───────────────────────────────

PROMPT = "rbtr.prompt"
INPUT_TEXT = "rbtr.input"
CURSOR = "rbtr.cursor"
DIM = "rbtr.dim"
MUTED = "rbtr.muted"
WARNING = "rbtr.warning"
ERROR = "rbtr.error"

RULE = "rbtr.rule"
FOOTER = "rbtr.footer"

COMPLETION_SELECTED = "rbtr.completion.selected"
COMPLETION_NAME = "rbtr.completion.name"
COMPLETION_DESC = "rbtr.completion.desc"

COLUMN_BRANCH = "rbtr.column.branch"

LINK_STYLE = "rbtr.link"
PASTE_MARKER = "rbtr.paste_marker"
CONTEXT_MARKER = "rbtr.context_marker"

BG_INPUT = "rbtr.bg.input"
BG_ACTIVE = "rbtr.bg.active"
BG_SUCCEEDED = "rbtr.bg.succeeded"
BG_FAILED = "rbtr.bg.failed"
BG_QUEUED = "rbtr.bg.queued"
BG_TOOLCALL = "rbtr.bg.toolcall"

USAGE_OK = "rbtr.usage.ok"
USAGE_WARNING = "rbtr.usage.warning"
USAGE_CRITICAL = "rbtr.usage.critical"
USAGE_UNCERTAIN = "rbtr.usage.uncertain"
USAGE_MESSAGES = "rbtr.usage.messages"

STYLE_DIM = "rbtr.out.dim"
STYLE_DIM_ITALIC = "rbtr.out.dim_italic"
STYLE_WARNING = "rbtr.out.warning"
STYLE_ERROR = "rbtr.out.error"
STYLE_SHELL_STDERR = "rbtr.out.shell_stderr"


# ── Palette → Theme mapping ───────────────────────────────────────────
#
# Each ``PaletteConfig`` field maps to a ``rbtr.*`` theme key.
# Convention: ``rbtr.<prefix>.<suffix>`` where the first underscore
# in the field name is the split point.  Fields that don't follow
# the convention are listed in ``_KEY_OVERRIDES``.

_KEY_OVERRIDES: dict[str, str] = {
    "input_text": "rbtr.input",
    "paste_marker": "rbtr.paste_marker",
    "context_marker": "rbtr.context_marker",
}


def _field_to_key(name: str) -> str:
    """Convert a ``PaletteConfig`` field name to a ``rbtr.*`` theme key."""
    if name in _KEY_OVERRIDES:
        return _KEY_OVERRIDES[name]
    parts = name.split("_", 1)
    return f"rbtr.{'.'.join(parts)}"


def palette_to_styles(palette: PaletteConfig) -> dict[str, str]:
    """Convert a palette to a ``{theme_key: style_string}`` dict."""
    return {_field_to_key(name): value for name, value in palette.model_dump().items()}


def build_theme(cfg: ThemeConfig) -> Theme:
    """Build a Rich ``Theme`` from theme config.

    Selects the palette for the configured mode and converts it
    to a ``Theme`` via ``palette_to_styles()``.
    """
    palette = cfg.dark if cfg.mode == "dark" else cfg.light
    return Theme(palette_to_styles(palette))


# ── Module-level default (used by tests) ─────────────────────────────

THEME = build_theme(ThemeConfig())
