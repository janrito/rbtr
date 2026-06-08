"""Cross-repo CLI rendering — behaviour through the `emit` surface.

Drives `emit()` and asserts on the rendered text captured by the
`rendered` fixture.  The render internals are never called
directly.
"""

from __future__ import annotations

from io import StringIO

from pytest_cases import parametrize_with_cases

from rbtr.cli.output import emit

from .cases_output import RenderScenario


@parametrize_with_cases("scenario", cases=".cases_output")
def test_emit_renders_repo_attribution(scenario: RenderScenario, rendered: StringIO) -> None:
    """Rendered output shows (and omits) the expected repo cues."""
    emit(scenario.model)

    out = rendered.getvalue()
    for text in scenario.expected:
        assert text in out
    for text in scenario.forbidden:
        assert text not in out
