"""Credential storage — ``~/.config/rbtr/creds.toml``.

The ``creds`` instance reloads in place via ``__init__()``, so a direct
import is safe — identity never changes::

    from rbtr.creds import creds

    creds.openai_api_key            # read
    creds.update(openai_api_key=…)  # write, persist, reload in place
"""

from __future__ import annotations

import stat
from typing import Any

import tomli_w
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from rbtr.constants import RBTR_DIR

CREDS_PATH = RBTR_DIR / "creds.toml"


class OAuthCreds(BaseModel):
    """OAuth token set — pure data, persisted in ``creds.toml``."""

    access_token: str = ""
    refresh_token: str = ""
    expires_at: float | None = None
    account_id: str = ""


class Creds(BaseSettings):
    model_config = SettingsConfigDict(toml_file=str(CREDS_PATH))

    github_token: str = ""
    claude: OAuthCreds = OAuthCreds()
    chatgpt: OAuthCreds = OAuthCreds()
    openai_api_key: str = ""
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
        return (TomlConfigSettingsSource(settings_cls),)

    def update(self, **kwargs: Any) -> None:
        """Set fields, persist to disk (0600), and reload in place."""
        merged = self.model_copy(update=kwargs)
        CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CREDS_PATH.write_text(tomli_w.dumps(merged.model_dump(exclude_defaults=True)))
        CREDS_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init


creds = Creds()
