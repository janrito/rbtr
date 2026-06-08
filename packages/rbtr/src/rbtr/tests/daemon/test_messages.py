"""Tests for daemon protocol messages — discriminated unions.

Scenarios live in `case_messages.py`. Two behaviours:
- **Routing:** raw JSON bytes → TypeAdapter → correct model type
  with expected field values.
- **Roundtrip:** model → JSON → TypeAdapter → same type preserved.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from .cases_messages import MessageScenario


@parametrize_with_cases("scenario")
def test_routing(scenario: MessageScenario) -> None:
    """Raw JSON is routed to the correct model type."""
    result = scenario.adapter.validate_json(scenario.raw)
    assert isinstance(result, scenario.expected_type)
    for field_name, expected in scenario.checks.items():
        assert getattr(result, field_name) == expected


@parametrize_with_cases("scenario")
def test_roundtrip(scenario: MessageScenario) -> None:
    """Model survives serialise → deserialise via its adapter."""
    model = scenario.adapter.validate_json(scenario.raw)
    roundtripped = scenario.adapter.validate_json(model.model_dump_json())
    assert type(roundtripped) is type(model)
