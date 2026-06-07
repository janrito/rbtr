"""Vega-Lite chart rendering to PNG."""

from __future__ import annotations

import json
from pathlib import Path

import vl_convert as vlc


def render_vl_to_png(spec: dict, output: Path, *, scale: int = 2) -> None:
    """Render a Vega-Lite spec dict to a PNG file."""
    png = vlc.vegalite_to_png(json.dumps(spec), scale=scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png)
