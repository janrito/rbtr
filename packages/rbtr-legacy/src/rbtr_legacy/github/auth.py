"""GitHub device flow authentication for rbtr."""

from __future__ import annotations

import threading
import time
from typing import TypedDict

import httpx

from rbtr_legacy.exceptions import RbtrError
from rbtr_legacy.oauth import deobfuscate

# ── Constants ────────────────────────────────────────────────────────

_CLIENT_ID = deobfuscate("T3YyM2xpNE9UQ1l5bzJZTndBdWs=")
_DEVICE_CODE_URL = "https://github.com/login/device/code"
_OAUTH_URL = "https://github.com/login/oauth/access_token"


class DeviceCodeResponse(TypedDict):
    """Shape of GitHub's device code endpoint response."""

    device_code: str
    user_code: str
    verification_uri: str
    interval: str  # GitHub returns this as a string


def request_device_code() -> DeviceCodeResponse:
    """Start the device flow."""
    resp = httpx.post(
        _DEVICE_CODE_URL,
        data={"client_id": _CLIENT_ID, "scope": "repo read:org"},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]  # httpx returns Any, we know shape matches TypedDict


def poll_for_token(device_code: str, interval: int, cancel: threading.Event | None = None) -> str:
    """Poll GitHub until the user authorizes or the code expires.

    If *cancel* is set, raises `RbtrError` promptly.
    """
    with httpx.Client() as client:
        while True:
            # Interruptible sleep — wake early on cancel
            if cancel is not None:
                if cancel.wait(timeout=interval):
                    raise RbtrError("Cancelled.")
            else:
                time.sleep(interval)

            resp = client.post(
                _OAUTH_URL,
                data={
                    "client_id": _CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            body: dict[str, str] = resp.json()

            match body.get("error"):
                case None:
                    return body["access_token"]
                case "authorization_pending":
                    continue
                case "slow_down":
                    interval += 5
                case "expired_token":
                    raise RbtrError("Device code expired. Please try logging in again.")
                case "access_denied":
                    raise RbtrError("Login cancelled by user.")
                case error:
                    raise RbtrError(f"Unexpected error during login: {error}")
