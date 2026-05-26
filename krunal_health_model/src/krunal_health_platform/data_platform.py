"""Four-zone data platform and memory scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import DataZone, PurposeOfUse, generate_id, jsonable, now_iso


@dataclass
class ZoneRecord:
    record_id: str
    user_id_hash: str
    zone: DataZone
    payload: Any
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    audit_id: str
    user_id_hash: str
    from_zone: DataZone | None
    to_zone: DataZone
    operation: str
    purpose_of_use: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PurposeGuard:
    """Simple purpose-of-use guardrail for the scaffold."""

    ALLOWED_READS = {
        PurposeOfUse.CONSUMER_COACHING: {
            DataZone.ZONE2_CANONICAL,
            DataZone.ZONE3_FEATURE_EXPERIMENT,
            DataZone.ZONE4_INSIGHT,
        },
        PurposeOfUse.RESEARCH: {DataZone.ZONE3_FEATURE_EXPERIMENT},
        PurposeOfUse.CLINICAL_EXPORT: {
            DataZone.ZONE2_CANONICAL,
            DataZone.ZONE3_FEATURE_EXPERIMENT,
            DataZone.ZONE4_INSIGHT,
        },
    }

    def assert_read_allowed(self, zone: DataZone, purpose: PurposeOfUse) -> None:
        if zone not in self.ALLOWED_READS[purpose]:
            raise PermissionError(f"{purpose.value} cannot read from {zone.value}")


class InMemoryZoneStore:
    def __init__(self) -> None:
        self._records: dict[DataZone, dict[str, ZoneRecord]] = {
            zone: {} for zone in DataZone
        }

    def write(
        self,
        zone: DataZone,
        user_id_hash: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ZoneRecord:
        record = ZoneRecord(
            record_id=generate_id(zone.value),
            user_id_hash=user_id_hash,
            zone=zone,
            payload=jsonable(payload),
            created_at=now_iso(),
            metadata=metadata or {},
        )
        self._records[zone][record.record_id] = record
        return record

    def read(self, zone: DataZone, record_id: str) -> ZoneRecord | None:
        return self._records[zone].get(record_id)

    def list_for_user(self, zone: DataZone, user_id_hash: str) -> list[ZoneRecord]:
        return [
            record
            for record in self._records[zone].values()
            if record.user_id_hash == user_id_hash
        ]


class FourZoneDataPlatform:
    """Reference in-memory implementation of the four-zone design."""

    def __init__(self) -> None:
        self.store = InMemoryZoneStore()
        self.guard = PurposeGuard()
        self.audit_log: list[AuditEvent] = []

    def _audit(
        self,
        user_id_hash: str,
        to_zone: DataZone,
        operation: str,
        purpose: PurposeOfUse,
        from_zone: DataZone | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            audit_id=generate_id("audit"),
            user_id_hash=user_id_hash,
            from_zone=from_zone,
            to_zone=to_zone,
            operation=operation,
            purpose_of_use=purpose.value,
            created_at=now_iso(),
            metadata=metadata or {},
        )
        self.audit_log.append(event)
        return event

    def ingest_raw_vendor_payload(
        self,
        user_id_hash: str,
        vendor: str,
        payload: Any,
        purpose: PurposeOfUse = PurposeOfUse.CONSUMER_COACHING,
    ) -> ZoneRecord:
        record = self.store.write(
            DataZone.ZONE1_INGESTION,
            user_id_hash,
            payload,
            metadata={"vendor": vendor},
        )
        self._audit(
            user_id_hash,
            DataZone.ZONE1_INGESTION,
            operation="ingest_raw_vendor_payload",
            purpose=purpose,
            metadata={"vendor": vendor, "record_id": record.record_id},
        )
        return record

    def write_canonical_timeline(
        self,
        user_id_hash: str,
        canonical_timeline: Any,
        source_record_ids: list[str],
        purpose: PurposeOfUse = PurposeOfUse.CONSUMER_COACHING,
    ) -> ZoneRecord:
        record = self.store.write(
            DataZone.ZONE2_CANONICAL,
            user_id_hash,
            canonical_timeline,
            metadata={"source_record_ids": source_record_ids},
        )
        self._audit(
            user_id_hash,
            DataZone.ZONE2_CANONICAL,
            operation="normalize_to_canonical_timeline",
            purpose=purpose,
            from_zone=DataZone.ZONE1_INGESTION,
            metadata={"record_id": record.record_id, "source_record_ids": source_record_ids},
        )
        return record

    def write_feature_bundle(
        self,
        user_id_hash: str,
        feature_bundle: Any,
        canonical_record_id: str,
        purpose: PurposeOfUse = PurposeOfUse.CONSUMER_COACHING,
    ) -> ZoneRecord:
        record = self.store.write(
            DataZone.ZONE3_FEATURE_EXPERIMENT,
            user_id_hash,
            feature_bundle,
            metadata={"canonical_record_id": canonical_record_id},
        )
        self._audit(
            user_id_hash,
            DataZone.ZONE3_FEATURE_EXPERIMENT,
            operation="materialize_feature_bundle",
            purpose=purpose,
            from_zone=DataZone.ZONE2_CANONICAL,
            metadata={"record_id": record.record_id, "canonical_record_id": canonical_record_id},
        )
        return record

    def publish_insight(
        self,
        user_id_hash: str,
        insight: Any,
        feature_record_id: str,
        purpose: PurposeOfUse = PurposeOfUse.CONSUMER_COACHING,
    ) -> ZoneRecord:
        record = self.store.write(
            DataZone.ZONE4_INSIGHT,
            user_id_hash,
            insight,
            metadata={"feature_record_id": feature_record_id},
        )
        self._audit(
            user_id_hash,
            DataZone.ZONE4_INSIGHT,
            operation="publish_app_insight",
            purpose=purpose,
            from_zone=DataZone.ZONE3_FEATURE_EXPERIMENT,
            metadata={"record_id": record.record_id, "feature_record_id": feature_record_id},
        )
        return record


class MemoryArchitecture:
    """Short-term vs long-term memory separation from the document."""

    def __init__(self) -> None:
        self.raw_interaction_log: dict[str, list[dict[str, Any]]] = {}
        self.short_term_working_memory: dict[str, dict[str, Any]] = {}
        self.long_term_knowledge_store: dict[str, list[dict[str, Any]]] = {}
        self.insight_cache: dict[str, dict[str, Any]] = {}

    def append_interaction(self, user_id_hash: str, event: dict[str, Any]) -> None:
        self.raw_interaction_log.setdefault(user_id_hash, []).append(event)

    def set_working_memory(self, user_id_hash: str, goal: str, plan: dict[str, Any]) -> None:
        self.short_term_working_memory[user_id_hash] = {
            "goal": goal,
            "plan": plan,
            "updated_at": now_iso(),
        }

    def remember_fact(self, user_id_hash: str, fact_type: str, value: Any) -> None:
        self.long_term_knowledge_store.setdefault(user_id_hash, []).append(
            {
                "fact_type": fact_type,
                "value": jsonable(value),
                "recorded_at": now_iso(),
            }
        )

    def cache_insight(self, user_id_hash: str, insight: dict[str, Any]) -> None:
        self.insight_cache[user_id_hash] = {
            "insight": jsonable(insight),
            "updated_at": now_iso(),
        }

    def get_context(self, user_id_hash: str) -> dict[str, Any]:
        return {
            "working_memory": self.short_term_working_memory.get(user_id_hash, {}),
            "long_term_facts": self.long_term_knowledge_store.get(user_id_hash, []),
            "insight_cache": self.insight_cache.get(user_id_hash, {}),
        }

