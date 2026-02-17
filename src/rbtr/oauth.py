"""Shared OAuth / PKCE utilities for provider modules."""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import TYPE_CHECKING

from rbtr.config import config

if TYPE_CHECKING:
    from rbtr.creds import OAuthCreds


def oauth_is_set(oauth: OAuthCreds) -> bool:
    """Return whether the token set contains an access token."""
    return bool(oauth.access_token)


def oauth_expired(oauth: OAuthCreds) -> bool:
    """Return whether the access token has expired (or is about to)."""
    if oauth.expires_at is None:
        return False
    return time.time() >= oauth.expires_at - config.oauth.refresh_buffer_seconds


def make_verifier() -> str:
    """Generate a cryptographically random PKCE code verifier (43-128 chars)."""
    return secrets.token_urlsafe(64)[:128]


def make_challenge(verifier: str) -> str:
    """Derive a S256 code challenge from the verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
