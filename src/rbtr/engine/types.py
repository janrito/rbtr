"""Engine enums — commands, services, task types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum


@dataclass
class _CommandMixin:
    slash: str
    description: str = field(default="")


class Command(_CommandMixin, Enum):
    """Slash commands recognised by the engine."""

    HELP = "/help", "Show available commands"
    REVIEW = "/review", "List open PRs, or select a target"
    CONNECT = "/connect", "Connect a service"
    MODEL = "/model", "List or set active model"
    NEW = "/new", "Start a new conversation"
    QUIT = "/quit", "Exit rbtr (also /q)"

    @classmethod
    def _missing_(cls, value: object) -> Command | None:
        for member in cls:
            if member.slash == value:
                return member
        if value == "/q":
            return cls.QUIT
        return None


@dataclass
class _ServiceMixin:
    key: str
    description: str = field(default="")


class Service(_ServiceMixin, Enum):
    """Services accepted by /connect."""

    GITHUB = "github", "Authenticate with GitHub"
    CLAUDE = "claude", "Sign in with Claude Pro/Max"
    CHATGPT = "chatgpt", "Sign in with ChatGPT Plus/Pro"
    OPENAI = "openai", "Connect with an OpenAI API key"
    ENDPOINT = "endpoint", "OpenAI-compatible endpoint"

    @classmethod
    def _missing_(cls, value: object) -> Service | None:
        for member in cls:
            if member.key == value:
                return member
        return None


class TaskType(StrEnum):
    """Types of work the engine can run in a daemon thread."""

    SETUP = "setup"
    COMMAND = "command"
    SHELL = "shell"
    LLM = "llm"


class TaskCancelled(Exception):
    """Raised inside a task thread when the user requests cancellation."""
