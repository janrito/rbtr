"""Tests for theme config, palette mapping, and build_theme."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rich.errors import StyleSyntaxError

import rbtr.styles as s
from rbtr.config import DarkPalette, LightPalette, PaletteConfig, ThemeConfig
from rbtr.styles import _field_to_key, build_theme, palette_to_styles

# ── Field-to-key mapping ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("prompt", "rbtr.prompt"),
        ("dim", "rbtr.dim"),
        ("bg_input", "rbtr.bg.input"),
        ("completion_selected", "rbtr.completion.selected"),
        ("out_dim_italic", "rbtr.out.dim_italic"),
        ("out_shell_stderr", "rbtr.out.shell_stderr"),
        # Overrides — don't follow the convention.
        ("input_text", "rbtr.input"),
        ("paste_marker", "rbtr.paste_marker"),
    ],
    ids=lambda v: v if "/" not in v else None,
)
def test_field_to_key(field: str, expected: str) -> None:
    assert _field_to_key(field) == expected


def test_palette_to_styles_has_all_fields() -> None:
    d = palette_to_styles(DarkPalette())
    assert len(d) == len(PaletteConfig.model_fields)
    assert all(k.startswith("rbtr.") for k in d)


def test_style_keys_match_constants() -> None:
    """Every theme key from `palette_to_styles` has a constant in `styles`."""

    constants = {
        v
        for k, v in vars(s).items()
        if isinstance(v, str) and v.startswith("rbtr.") and k.isupper()
    }
    style_keys = set(palette_to_styles(DarkPalette()).keys())
    assert constants == style_keys


# ── Palette defaults ─────────────────────────────────────────────────


def test_base_palette_requires_backgrounds() -> None:
    with pytest.raises(ValidationError):
        PaletteConfig()  # type: ignore[call-arg]  # testing validation of missing required fields


def test_dark_palette_has_dark_backgrounds() -> None:
    p = DarkPalette()
    assert p.bg_succeeded == "on #1A2620"
    assert p.bg_failed == "on #2A1D20"


def test_light_palette_has_light_backgrounds() -> None:
    p = LightPalette()
    assert p.bg_succeeded == "on #E8F5E9"
    assert p.bg_failed == "on #FFEBEE"


def test_light_palette_adjusts_warning_for_readability() -> None:
    dark = DarkPalette()
    light = LightPalette()
    assert dark.warning == "yellow"
    assert light.warning == "dark_orange"


# ── ThemeConfig partial overrides ────────────────────────────────────


def test_partial_light_override_keeps_light_defaults() -> None:
    """Overriding one field in `[theme.light]` doesn't pull dark defaults."""
    cfg = ThemeConfig.model_validate({"light": {"bg_succeeded": "on #E0FFE0"}})
    assert cfg.light.bg_succeeded == "on #E0FFE0"
    assert cfg.light.bg_failed == LightPalette().bg_failed


def test_partial_dark_override_keeps_dark_defaults() -> None:
    cfg = ThemeConfig.model_validate({"dark": {"prompt": "bold magenta"}})
    assert cfg.dark.prompt == "bold magenta"
    assert cfg.dark.bg_succeeded == DarkPalette().bg_succeeded


# ── build_theme ──────────────────────────────────────────────────────


def test_build_theme_dark_default() -> None:
    theme = build_theme(ThemeConfig())
    assert theme.styles["rbtr.bg.succeeded"].bgcolor is not None


def test_build_theme_light() -> None:
    dark = build_theme(ThemeConfig())
    light = build_theme(ThemeConfig(mode="light"))
    assert str(dark.styles["rbtr.bg.succeeded"]) != str(light.styles["rbtr.bg.succeeded"])


def test_build_theme_override_applies() -> None:
    cfg = ThemeConfig(dark=DarkPalette(prompt="bold magenta"))
    theme = build_theme(cfg)
    assert "magenta" in str(theme.styles["rbtr.prompt"])


def test_build_theme_invalid_style_raises() -> None:
    cfg = ThemeConfig(dark=DarkPalette(prompt="not a valid style!!!"))
    with pytest.raises(StyleSyntaxError):
        build_theme(cfg)
