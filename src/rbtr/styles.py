"""Ayu Mirage-based Rich theme for rbtr.

All visual styling lives in the ``THEME`` object.  Code that renders
must reference the semantic keys defined here — never use inline hex
colours or ad-hoc style strings.

Engine code (which must not import Rich) uses the plain-string
constants exported below so that style names travel through events.

Colour palette derived from Ayu (https://github.com/ayu-theme/ayu-colors):

    MIT License

    Copyright (c) Konstantin Pschera <me@kons.ch> (kons.ch)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
"""

from rich.theme import Theme

# ── Ayu Mirage palette reference ─────────────────────────────────────
#
# surface.sunk   #181C26    surface.base  #1F2430
# surface.lift   #242936    ui.panel.bg   #282E3B
# ui.fg          #707A8C    editor.fg     #CCCAC2
# ui.line        #171B24    editor.line   #1A1F29
#
# red   #F28779    orange  #FFA659    yellow #FFCD66
# green #D5FF80    teal    #95E6CB    indigo #5CCFE6
# blue  #73D0FF    purple  #DFBFFF    pink   #F29E74

THEME = Theme(
    {
        # ── Prompt / input ───────────────────────────────────────────
        "rbtr.prompt": "bold #5CCFE6",
        "rbtr.input": "bold #CCCAC2",
        "rbtr.cursor": "reverse",
        # ── Panel backgrounds ────────────────────────────────────────
        "rbtr.bg.input": "on #282E3B",
        "rbtr.bg.active": "on #1C212C",
        "rbtr.bg.succeeded": "on #1A2620",
        "rbtr.bg.failed": "on #2A1D20",
        "rbtr.bg.queued": "on #1A1F29",
        "rbtr.bg.toolcall": "on #231D2F",
        # ── General text styles ──────────────────────────────────────
        "rbtr.dim": "#707A8C",
        "rbtr.muted": "#565E6B",
        "rbtr.warning": "#FFCD66",
        "rbtr.error": "bold #F28779",
        "rbtr.success": "#D5FF80",
        # ── Chrome ───────────────────────────────────────────────────
        "rbtr.rule": "#282E3B",
        "rbtr.footer": "#565E6B",
        # ── Completion menu ──────────────────────────────────────────
        "rbtr.completion.selected": "bold #5CCFE6",
        "rbtr.completion.name": "bold #73D0FF",
        "rbtr.completion.desc": "#707A8C",
        # ── Table columns ────────────────────────────────────────────
        "rbtr.column.branch": "#5CCFE6",
        # ── Inline markup ────────────────────────────────────────────
        "rbtr.link": "bold #5CCFE6",
        "rbtr.code": "bold #FFCD66",
        # ── Usage / context ───────────────────────────────────────────
        "rbtr.usage.ok": "dim #D5FF80",
        "rbtr.usage.warning": "dim #FFCD66",
        "rbtr.usage.critical": "dim #F28779",
        "rbtr.usage.uncertain": "#282E3B",
        "rbtr.usage.messages": "dim #707A8C",
        # ── Output styles (travel through events) ────────────────────
        "rbtr.out.dim": "#707A8C",
        "rbtr.out.dim_italic": "italic #707A8C",
        "rbtr.out.warning": "#FFCD66",
        "rbtr.out.error": "bold #F28779",
        "rbtr.out.shell_stderr": "#F28779",
    }
)

# ── String constants for Engine (no Rich imports needed) ─────────────
# These are the theme key names that travel through Output events.

PROMPT = "rbtr.prompt"
INPUT_TEXT = "rbtr.input"
CURSOR = "rbtr.cursor"
DIM = "rbtr.dim"
MUTED = "rbtr.muted"
WARNING = "rbtr.warning"
ERROR = "rbtr.error"
SUCCESS = "rbtr.success"

RULE = "rbtr.rule"
FOOTER = "rbtr.footer"

COMPLETION_SELECTED = "rbtr.completion.selected"
COMPLETION_NAME = "rbtr.completion.name"
COMPLETION_DESC = "rbtr.completion.desc"

COLUMN_BRANCH = "rbtr.column.branch"

LINK_STYLE = "rbtr.link"
CODE_HIGHLIGHT = "rbtr.code"

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
