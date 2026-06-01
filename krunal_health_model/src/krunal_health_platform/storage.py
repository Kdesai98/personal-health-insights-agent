"""SQLite persistence for local app development."""

from __future__ import annotations

import json
import sqlite3
from threading import Lock
from pathlib import Path
from typing import Any

from .models import generate_id, jsonable, now_iso


class SQLiteHealthStore:
    """Durable local store for states, feature bundles, traces, and feedback."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = Lock()
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    def _init_schema(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_states (
                  user_id_hash TEXT NOT NULL,
                  date TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (user_id_hash, date)
                );

                CREATE TABLE IF NOT EXISTS feature_bundles (
                  user_id_hash TEXT NOT NULL,
                  as_of_date TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY (user_id_hash, as_of_date)
                );

                CREATE TABLE IF NOT EXISTS intervention_traces (
                  trace_id TEXT PRIMARY KEY,
                  user_id_hash TEXT NOT NULL,
                  date TEXT NOT NULL,
                  selected_action_id TEXT,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                  event_id TEXT PRIMARY KEY,
                  user_id_hash TEXT NOT NULL,
                  date TEXT NOT NULL,
                  action_id TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_user_date
                  ON feedback_events (user_id_hash, date);

                CREATE INDEX IF NOT EXISTS idx_trace_user_date
                  ON intervention_traces (user_id_hash, date);
                """
            )
            self.connection.commit()

    def upsert_daily_state(self, user_id_hash: str, daily_state: dict[str, Any]) -> None:
        date = str(daily_state.get("date"))
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO daily_states (user_id_hash, date, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id_hash, date) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (
                    user_id_hash,
                    date,
                    json.dumps(jsonable(daily_state), sort_keys=True),
                    now_iso(),
                ),
            )
            self.connection.commit()

    def list_daily_states(self, user_id_hash: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT payload_json FROM daily_states WHERE user_id_hash = ? "
            "ORDER BY date ASC"
        )
        with self._lock:
            rows = self.connection.execute(sql, (user_id_hash,)).fetchall()
        if limit is not None:
            rows = rows[-limit:]
        return [json.loads(row["payload_json"]) for row in rows]

    def save_feature_bundle(self, user_id_hash: str, feature_bundle: dict[str, Any]) -> None:
        as_of_date = str(feature_bundle.get("as_of_date"))
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO feature_bundles (user_id_hash, as_of_date, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id_hash, as_of_date) DO UPDATE SET
                  payload_json=excluded.payload_json,
                  created_at=excluded.created_at
                """,
                (
                    user_id_hash,
                    as_of_date,
                    json.dumps(jsonable(feature_bundle), sort_keys=True),
                    now_iso(),
                ),
            )
            self.connection.commit()

    def list_feature_bundles(self, user_id_hash: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT payload_json FROM feature_bundles
                WHERE user_id_hash = ?
                ORDER BY as_of_date ASC, created_at ASC
                """,
                (user_id_hash,),
            ).fetchall()
        if limit is not None:
            rows = rows[-limit:]
        return [json.loads(row["payload_json"]) for row in rows]

    def save_intervention_trace(
        self,
        user_id_hash: str,
        date: str,
        trace: dict[str, Any],
    ) -> str:
        trace_id = str(trace.get("trace_id") or generate_id("trace"))
        selected_action_id = trace.get("selected_action_id")
        with self._lock:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO intervention_traces
                  (trace_id, user_id_hash, date, selected_action_id, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    user_id_hash,
                    date,
                    selected_action_id,
                    json.dumps(jsonable(trace), sort_keys=True),
                    now_iso(),
                ),
            )
            self.connection.commit()
        return trace_id

    def list_intervention_traces(self, user_id_hash: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT date, payload_json FROM intervention_traces
                WHERE user_id_hash = ?
                ORDER BY date ASC, created_at ASC
                """,
                (user_id_hash,),
            ).fetchall()
        if limit is not None:
            rows = rows[-limit:]
        traces = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload.setdefault("date", row["date"])
            traces.append(payload)
        return traces

    def append_feedback_event(self, user_id_hash: str, feedback: dict[str, Any]) -> str:
        event_id = str(feedback.get("event_id") or generate_id("feedback"))
        date = str(feedback.get("date"))
        action_id = str(feedback.get("action_id"))
        payload = {**feedback, "event_id": event_id}
        with self._lock:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO feedback_events
                  (event_id, user_id_hash, date, action_id, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    user_id_hash,
                    date,
                    action_id,
                    json.dumps(jsonable(payload), sort_keys=True),
                    now_iso(),
                ),
            )
            self.connection.commit()
        return event_id

    def list_feedback_events(self, user_id_hash: str, action_id: str | None = None) -> list[dict[str, Any]]:
        if action_id is None:
            with self._lock:
                rows = self.connection.execute(
                    """
                    SELECT payload_json FROM feedback_events
                    WHERE user_id_hash = ?
                    ORDER BY date ASC, created_at ASC
                    """,
                    (user_id_hash,),
                ).fetchall()
        else:
            with self._lock:
                rows = self.connection.execute(
                    """
                    SELECT payload_json FROM feedback_events
                    WHERE user_id_hash = ? AND action_id = ?
                    ORDER BY date ASC, created_at ASC
                    """,
                    (user_id_hash, action_id),
                ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]
