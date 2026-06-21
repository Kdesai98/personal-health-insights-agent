"""Privacy, consent, and lightweight auth primitives."""

from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from .models import DataZone, PurposeOfUse, jsonable, now_iso


@dataclass
class ConsentGrant:
    user_id_hash: str
    purpose: PurposeOfUse
    zones: list[DataZone]
    granted: bool
    granted_at: str = field(default_factory=now_iso)
    revoked_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class ConsentRegistry:
    def __init__(self) -> None:
        self._grants: dict[tuple[str, str], ConsentGrant] = {}

    def grant(self, user_id_hash: str, purpose: PurposeOfUse, zones: list[DataZone]) -> ConsentGrant:
        grant = ConsentGrant(
            user_id_hash=user_id_hash,
            purpose=purpose,
            zones=zones,
            granted=True,
        )
        self._grants[(user_id_hash, purpose.value)] = grant
        return grant

    def revoke(self, user_id_hash: str, purpose: PurposeOfUse) -> None:
        grant = self._grants.get((user_id_hash, purpose.value))
        if grant:
            grant.granted = False
            grant.revoked_at = now_iso()

    def assert_allowed(self, user_id_hash: str, purpose: PurposeOfUse, zone: DataZone) -> None:
        grant = self._grants.get((user_id_hash, purpose.value))
        if not grant or not grant.granted or zone not in grant.zones:
            raise PermissionError(f"missing consent for {purpose.value} on {zone.value}")


class PHIRedactor:
    """Redacts common identifiers before lower-sensitivity logs or prompts."""

    EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")

    def redact_text(self, text: str) -> str:
        text = self.EMAIL_RE.sub("[redacted_email]", text)
        text = self.PHONE_RE.sub("[redacted_phone]", text)
        return text


class BearerTokenAuth:
    """Simple bearer-token check for local and first hosted deployments."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ.get("KRUNAL_HEALTH_API_TOKEN")

    def enabled(self) -> bool:
        return bool(self.token)

    def assert_authorized(self, authorization_header: str | None) -> None:
        if not self.enabled():
            return
        expected = f"Bearer {self.token}"
        if not authorization_header or not hmac.compare_digest(authorization_header, expected):
            raise PermissionError("unauthorized")


def hash_user_identifier(identifier: str, salt: str | None = None) -> str:
    salt_value = salt or os.environ.get("KRUNAL_HEALTH_HASH_SALT", "local_dev_salt")
    return sha256(f"{salt_value}:{identifier}".encode("utf-8")).hexdigest()[:24]
