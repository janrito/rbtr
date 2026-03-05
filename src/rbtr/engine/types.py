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
    DRAFT = "/draft", "View, sync, or post the review draft"
    CONNECT = "/connect", "Connect a service"
    MODEL = "/model", "List or set active model"
    INDEX = "/index", "Index status, clear, rebuild, prune, model"
    COMPACT = "/compact", "Summarise older context (reset = undo last)"
    SESSION = "/session", "List, inspect, or delete sessions"
    STATS = "/stats", "Show session token and cost statistics"
    RELOAD = "/reload", "Show active prompt sources"
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
    """Non-provider services accepted by /connect.

    LLM providers (builtins and endpoints) are handled separately
    by the connect command.
    """

    GITHUB = "github", "Authenticate with GitHub"

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
