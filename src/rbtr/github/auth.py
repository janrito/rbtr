"""GitHub device flow authentication for rbtr."""

import stat
import time

import httpx

from rbtr import RbtrError
from rbtr.constants import (
    GITHUB_CLIENT_ID,
    GITHUB_DEVICE_CODE_URL,
    GITHUB_OAUTH_URL,
    TOKEN_PATH,
)


def request_device_code() -> dict[str, str]:
    """Start the device flow. Returns device_code, user_code, verification_uri, interval."""
    with httpx.Client() as client:
        resp = client.post(
            GITHUB_DEVICE_CODE_URL,
            data={"client_id": GITHUB_CLIENT_ID, "scope": "repo read:org"},
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]


def poll_for_token(device_code: str, interval: int) -> str:
    """Poll GitHub until the user authorizes or the code expires. Returns access token."""
    with httpx.Client() as client:
        while True:
            time.sleep(interval)
            resp = client.post(
                GITHUB_OAUTH_URL,
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            body: dict[str, str] = resp.json()

            error = body.get("error")
            if not error:
                return body["access_token"]

            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error == "expired_token":
                raise RbtrError("Device code expired. Please try logging in again.")
            if error == "access_denied":
                raise RbtrError("Login cancelled by user.")
            raise RbtrError(f"Unexpected error during login: {error}")


def save_token(token: str) -> None:
    """Write token to disk with restricted permissions (0600)."""
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token)
    TOKEN_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)


def load_token() -> str | None:
    """Load a stored token from disk. Returns None if not found or empty."""
    if TOKEN_PATH.exists():
        token = TOKEN_PATH.read_text().strip()
        if token:
            return token
    return None


def clear_token() -> None:
    """Remove the stored token file."""
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
