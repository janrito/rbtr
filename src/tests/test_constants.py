"""Tests for config — defaults, invariants, and layered loading."""

import tomllib
from pathlib import Path

import tomli_w

from rbtr.config import Config, EndpointConfig, _deep_merge, config

# ── Keystroke latency target ─────────────────────────────────────────
#
# Worst-case keystroke-to-display latency is bounded by:
#   poll_interval (main loop notices the change)
#   + 1/refresh_per_second (Rich Live paints it)
#
# Target: under 100ms.  Perceptible typing lag starts around 100ms.

_MAX_KEYSTROKE_LATENCY = 0.100  # seconds


def test_poll_interval_is_positive():
    assert config.tui.poll_interval > 0


def test_refresh_rate_is_positive():
    assert config.tui.refresh_per_second > 0


def test_worst_case_keystroke_latency():
    worst_case = config.tui.poll_interval + 1 / config.tui.refresh_per_second
    assert worst_case < _MAX_KEYSTROKE_LATENCY, (
        f"Worst-case latency {worst_case * 1000:.0f}ms exceeds "
        f"{_MAX_KEYSTROKE_LATENCY * 1000:.0f}ms target"
    )


def test_poll_interval_not_faster_than_refresh():
    """No point polling faster than Rich can repaint."""
    assert config.tui.poll_interval >= 1 / config.tui.refresh_per_second - 0.001


# ── Other timing invariants ──────────────────────────────────────────


def test_completion_timeout_is_bounded():
    """Completion subprocess must not block longer than a few seconds."""
    assert 0 < config.tui.shell_completion_timeout <= 5.0


def test_double_ctrl_c_window_is_reasonable():
    """Window must be short enough to feel responsive, long enough to be intentional."""
    assert 0.2 <= config.tui.double_ctrl_c_window <= 1.0


# ── _deep_merge unit tests ───────────────────────────────────────────


def test_deep_merge_flat():
    assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_deep_merge_overwrite_scalar():
    assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_deep_merge_nested():
    base = {"s": {"x": 1, "y": 2}}
    assert _deep_merge(base, {"s": {"x": 99}}) == {"s": {"x": 99, "y": 2}}


def test_deep_merge_empty_dict_replaces():
    """An empty dict clears the section (doesn't silently keep old keys)."""
    base = {"ep": {"a": {"url": "http://a"}}}
    assert _deep_merge(base, {"ep": {}}) == {"ep": {}}


def test_deep_merge_does_not_mutate_base():
    base = {"s": {"x": 1}}
    _deep_merge(base, {"s": {"x": 2}})
    assert base == {"s": {"x": 1}}


# ── Layered loading ──────────────────────────────────────────────────


def _setup(monkeypatch, tmp_path, *, user=None, ws=None):
    """Write user/workspace TOML files and point config at them."""
    user_path = tmp_path / "user.toml"
    ws_path = tmp_path / "ws.toml"
    if user is not None:
        user_path.write_text(tomli_w.dumps(user))
    if ws is not None:
        ws_path.write_text(tomli_w.dumps(ws))
    monkeypatch.setattr("rbtr.config.CONFIG_PATH", user_path)
    monkeypatch.setattr("rbtr.config.WORKSPACE_PATH", ws_path)
    config.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init
    return user_path, ws_path


def test_defaults_when_no_files(tmp_path: Path, monkeypatch):
    """All values fall back to class defaults when no files exist."""
    _setup(monkeypatch, tmp_path)

    assert config.model is None
    assert config.endpoints == {}
    assert config.github.timeout == 10
    assert config.github.max_branches == 30


def test_user_file_overrides_defaults(tmp_path: Path, monkeypatch):
    _setup(monkeypatch, tmp_path, user={"github": {"timeout": 42}})

    assert config.github.timeout == 42
    assert config.github.max_branches == 30


def test_workspace_overrides_user(tmp_path: Path, monkeypatch):
    """Workspace wins on conflict; user's non-conflicting keys survive."""
    _setup(
        monkeypatch,
        tmp_path,
        user={"github": {"timeout": 20, "max_branches": 50}},
        ws={"github": {"timeout": 99}},
    )

    assert config.github.timeout == 99
    assert config.github.max_branches == 50


def test_workspace_overrides_user_top_level(tmp_path: Path, monkeypatch):
    """Workspace model overrides user model."""
    _setup(
        monkeypatch,
        tmp_path,
        user={"model": "user/model"},
        ws={"model": "ws/model"},
    )

    assert config.model == "ws/model"


def test_loading_complex_workspace(tmp_path: Path, monkeypatch):
    """Complex nested workspace settings load and merge correctly."""
    _setup(
        monkeypatch,
        tmp_path,
        user={
            "model": "user/model",
            "endpoints": {"alpha": {"base_url": "https://alpha.example.com"}},
            "github": {"timeout": 20},
        },
        ws={
            "model": "ws/model",
            "endpoints": {"beta": {"base_url": "https://beta.example.com"}},
            "github": {"max_branches": 100},
            "tui": {"shell_max_lines": 50},
        },
    )

    assert config.model == "ws/model"
    # Endpoints deep-merge: both survive
    assert "alpha" in config.endpoints
    assert "beta" in config.endpoints
    # github deep-merge: timeout from user, max_branches from workspace
    assert config.github.timeout == 20
    assert config.github.max_branches == 100
    # tui from workspace
    assert config.tui.shell_max_lines == 50
    # tui defaults survive
    assert config.tui.max_completions == 20


# ── model_dump excludes defaults ─────────────────────────────────────


def test_dump_excludes_defaults():
    c = Config(model="saved/model")
    data = c.model_dump(exclude_none=True, exclude_defaults=True, mode="json")
    assert data == {"model": "saved/model"}


def test_dump_nested_excludes_defaults():
    c = Config(github={"timeout": 99})
    data = c.model_dump(exclude_none=True, exclude_defaults=True, mode="json")
    assert data == {"github": {"timeout": 99}}


# ── update() routing ─────────────────────────────────────────────────


def test_update_saves_to_user_when_no_workspace(tmp_path: Path, monkeypatch):
    user_path, ws_path = _setup(monkeypatch, tmp_path)

    config.update(model="to-user")

    assert user_path.exists()
    assert not ws_path.exists()
    assert tomllib.loads(user_path.read_text())["model"] == "to-user"


def test_update_saves_to_workspace_when_it_exists(tmp_path: Path, monkeypatch):
    _, ws_path = _setup(monkeypatch, tmp_path, ws={"model": "old-ws"})

    config.update(model="new-ws")

    assert tomllib.loads(ws_path.read_text())["model"] == "new-ws"


def test_update_does_not_write_user_when_workspace_exists(tmp_path: Path, monkeypatch):
    """When workspace exists, update only touches the workspace file."""
    user_path, _ws_path = _setup(
        monkeypatch,
        tmp_path,
        user={"model": "user-model"},
        ws={"github": {"timeout": 5}},
    )

    user_mtime = user_path.stat().st_mtime
    config.update(model="changed")

    assert user_path.stat().st_mtime == user_mtime
    assert tomllib.loads(user_path.read_text())["model"] == "user-model"


# ── update() reload ──────────────────────────────────────────────────


def test_update_reloads_merged_state(tmp_path: Path, monkeypatch):
    _setup(monkeypatch, tmp_path)

    assert config.model is None
    config.update(model="after-update")
    assert config.model == "after-update"


def test_update_reload_merges_both_files(tmp_path: Path, monkeypatch):
    """After updating workspace, config still sees user settings."""
    _setup(
        monkeypatch,
        tmp_path,
        user={"model": "user/model", "github": {"timeout": 77}},
        ws={"tui": {"shell_max_lines": 50}},
    )

    config.update(tui={"max_completions": 5})

    # Updated workspace value
    assert config.tui.max_completions == 5
    # Existing workspace value preserved
    assert config.tui.shell_max_lines == 50
    # User values still merged in
    assert config.model == "user/model"
    assert config.github.timeout == 77


# ── update() preserves / clears ──────────────────────────────────────


def test_update_preserves_unrelated_keys_in_target_file(tmp_path: Path, monkeypatch):
    """Updating one key doesn't clobber other overrides in the same file."""
    user_path, _ = _setup(
        monkeypatch,
        tmp_path,
        user={
            "model": "keep-me",
            "endpoints": {"custom": {"base_url": "https://example.com"}},
        },
    )

    config.update(
        endpoints={
            **config.endpoints,
            "new": EndpointConfig(base_url="https://new.example.com"),
        }
    )

    data = tomllib.loads(user_path.read_text())
    assert data["model"] == "keep-me"
    assert "custom" in data["endpoints"]
    assert "new" in data["endpoints"]


def test_update_preserves_ws_keys_when_updating_ws(tmp_path: Path, monkeypatch):
    """Updating workspace preserves other workspace overrides."""
    _, ws_path = _setup(
        monkeypatch,
        tmp_path,
        ws={
            "model": "ws-model",
            "endpoints": {"ep": {"base_url": "https://ep.example.com"}},
            "github": {"timeout": 42, "max_branches": 99},
        },
    )

    config.update(model="new-ws-model")

    ws_data = tomllib.loads(ws_path.read_text())
    assert ws_data["model"] == "new-ws-model"
    assert ws_data["endpoints"]["ep"]["base_url"] == "https://ep.example.com"
    assert ws_data["github"]["timeout"] == 42
    assert ws_data["github"]["max_branches"] == 99


def test_update_can_clear_to_default(tmp_path: Path, monkeypatch):
    user_path, _ = _setup(monkeypatch, tmp_path, user={"model": "will-be-cleared"})

    config.update(model=None)

    data = tomllib.loads(user_path.read_text())
    assert "model" not in data
    assert config.model is None


def test_update_remove_endpoint(tmp_path: Path, monkeypatch):
    user_path, _ = _setup(
        monkeypatch,
        tmp_path,
        user={"endpoints": {"ep1": {"base_url": "https://ep1.example.com"}}},
    )

    assert "ep1" in config.endpoints
    config.update(endpoints={})
    assert config.endpoints == {}

    data = tomllib.loads(user_path.read_text())
    assert "endpoints" not in data


def test_update_only_saves_non_defaults_to_file(tmp_path: Path, monkeypatch):
    """update() never writes default values to disk."""
    user_path, _ = _setup(monkeypatch, tmp_path)

    config.update(model="my/model")

    data = tomllib.loads(user_path.read_text())
    # Only the override — no github, tui, oauth, providers sections
    assert data == {"model": "my/model"}


def test_update_only_saves_non_defaults_to_workspace(tmp_path: Path, monkeypatch):
    """Same as above but for workspace file."""
    _, ws_path = _setup(monkeypatch, tmp_path, ws={})

    config.update(model="ws/model")

    data = tomllib.loads(ws_path.read_text())
    assert data == {"model": "ws/model"}


# ── update() deep merge in target file ───────────────────────────────


def test_update_deep_merges_nested_section(tmp_path: Path, monkeypatch):
    """Updating one nested key preserves sibling keys in the file."""
    user_path, _ = _setup(
        monkeypatch,
        tmp_path,
        user={"github": {"timeout": 77, "max_branches": 99}},
    )

    config.update(github={"timeout": 5})

    data = tomllib.loads(user_path.read_text())
    assert data["github"]["timeout"] == 5
    assert data["github"]["max_branches"] == 99


def test_update_deep_merges_endpoints(tmp_path: Path, monkeypatch):
    """Adding an endpoint preserves existing ones in the file via deep merge."""
    user_path, _ = _setup(
        monkeypatch,
        tmp_path,
        user={"endpoints": {"old": {"base_url": "https://old.example.com"}}},
    )

    config.update(endpoints={"new": {"base_url": "https://new.example.com"}})

    data = tomllib.loads(user_path.read_text())
    assert data["endpoints"]["old"]["base_url"] == "https://old.example.com"
    assert data["endpoints"]["new"]["base_url"] == "https://new.example.com"


def test_update_ws_deep_merges_nested_section(tmp_path: Path, monkeypatch):
    """Deep merge works on workspace file too."""
    _, ws_path = _setup(
        monkeypatch,
        tmp_path,
        ws={"github": {"timeout": 77, "max_branches": 99}, "model": "ws/m"},
    )

    config.update(github={"timeout": 5})

    data = tomllib.loads(ws_path.read_text())
    assert data["github"]["timeout"] == 5
    assert data["github"]["max_branches"] == 99
    assert data["model"] == "ws/m"


# ── End-to-end scenario ──────────────────────────────────────────────


def test_full_lifecycle(tmp_path: Path, monkeypatch):
    """add endpoint → change model → remove endpoint → verify files."""
    user_path, _ = _setup(monkeypatch, tmp_path)

    # 1. Add endpoint
    config.update(endpoints={"api": {"base_url": "https://api.example.com"}})
    assert config.endpoints["api"].base_url == "https://api.example.com"

    # 2. Change model — endpoint survives
    config.update(model="test/model")
    assert config.model == "test/model"
    assert "api" in config.endpoints
    data = tomllib.loads(user_path.read_text())
    assert data["model"] == "test/model"
    assert "api" in data["endpoints"]

    # 3. Remove endpoint — model survives
    config.update(endpoints={})
    assert config.endpoints == {}
    assert config.model == "test/model"
    data = tomllib.loads(user_path.read_text())
    assert "endpoints" not in data
    assert data["model"] == "test/model"


def test_full_lifecycle_with_workspace(tmp_path: Path, monkeypatch):
    """User has settings, workspace appears, updates go to workspace."""
    user_path, ws_path = _setup(
        monkeypatch,
        tmp_path,
        user={"model": "user/model", "github": {"timeout": 30}},
    )

    # No workspace yet — update goes to user
    config.update(model="updated/model")
    assert tomllib.loads(user_path.read_text())["model"] == "updated/model"

    # Workspace appears
    ws_path.write_text(tomli_w.dumps({"tui": {"shell_max_lines": 100}}))
    config.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init

    # Now updates go to workspace, user file untouched
    user_before = user_path.read_text()
    config.update(model="ws/model")
    assert user_path.read_text() == user_before

    ws_data = tomllib.loads(ws_path.read_text())
    assert ws_data["model"] == "ws/model"
    assert ws_data["tui"]["shell_max_lines"] == 100

    # Merged view: workspace model wins, user github survives
    assert config.model == "ws/model"
    assert config.github.timeout == 30
    assert config.tui.shell_max_lines == 100
