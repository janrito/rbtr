"""Credential storage — `~/.config/rbtr/creds.toml`.

The `creds` instance reloads in place via `reload()`, so a direct
import is safe — identity never changes::

    from rbtr_legacy.creds import creds

    creds.openai_api_key            # read
    creds.update(openai_api_key=…)  # write, persist, reload in place
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from rbtr_legacy.config import config


class OAuthCreds(BaseModel):
    """OAuth token set — pure data, persisted in `creds.toml`."""

    access_token: str = ""
    refresh_token: str = ""
    expires_at: float | None = None
    account_id: str = ""
    project_id: str = ""


class Creds(BaseSettings):
    model_config = SettingsConfigDict()

    github_token: str = ""
    claude: OAuthCreds = OAuthCreds()
    chatgpt: OAuthCreds = OAuthCreds()
    openai_api_key: str = ""
    fireworks_api_key: str = ""
    openrouter_api_key: str = ""
    google: OAuthCreds = OAuthCreds()
    endpoint_keys: dict[str, str] = {}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=str(Path(config.user_dir) / "creds.toml"),
            ),
            env_settings,
        )

    def update(self, **kwargs: Any) -> None:
        """Set fields, persist to disk (0600), and reload in place."""
        path = Path(config.user_dir) / "creds.toml"
        merged = self.model_copy(update=kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(merged.model_dump(exclude_defaults=True)))
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self.reload()

    def reload(self) -> None:
        """Re-read credentials from disk in place."""
        self.__init__()  # type: ignore[misc]  # pydantic re-init


creds = Creds()
