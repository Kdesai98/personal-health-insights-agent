"""Minimal local HTTP API around the health intelligence service.

This is intentionally standard-library only. It gives the app a concrete API
contract now; it can be swapped for FastAPI when hosting decisions are locked.
"""

from __future__ import annotations

import argparse
import json
import os
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from krunal_health_engine.daily_health_engine import build_daily_state

from .privacy import BearerTokenAuth
from .service import HealthIntelligenceService


class HealthAPIHandler(BaseHTTPRequestHandler):
    service: HealthIntelligenceService
    auth: BearerTokenAuth
    web_root = Path(__file__).resolve().parents[2] / "web"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/app.js", "/styles.css"}:
            self._serve_static(parsed.path)
            return
        if not self._authorized():
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/learning-summary":
            query = parse_qs(parsed.query)
            user_id_hash = query.get("user_id_hash", ["krunal_demo_user"])[0]
            self._send_json(self.service.build_learning_summary(user_id_hash))
            return
        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def _serve_static(self, path: str) -> None:
        file_path = self.web_root / ("index.html" if path == "/" else path.lstrip("/"))
        if not file_path.exists():
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if not self._authorized():
            return
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/daily-cycle":
                user_id_hash = payload.get("user_id_hash", "krunal_demo_user")
                daily_state = payload.get("daily_state")
                input_dir = payload.get("input_dir")
                if daily_state is None and input_dir:
                    daily_state = build_daily_state(Path(input_dir))
                if daily_state is None:
                    self._send_json(
                        {"error": "daily_state or input_dir is required"},
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                result = self.service.run_daily_cycle(
                    user_id_hash=user_id_hash,
                    daily_state=daily_state,
                )
                self._send_json(result.to_dict())
                return
            if parsed.path == "/feedback":
                event_id = self.service.record_feedback(
                    user_id_hash=payload.get("user_id_hash", "krunal_demo_user"),
                    date=payload["date"],
                    action_id=payload["action_id"],
                    adhered=payload.get("adhered"),
                    outcomes=payload.get("outcomes"),
                    reward=payload.get("reward"),
                    notes=payload.get("notes"),
                )
                self._send_json({"event_id": event_id})
                return
            if parsed.path == "/voice/food/parse":
                self._send_json(self.service.parse_food_voice_log(str(payload["raw_text"])))
                return
            if parsed.path == "/labs/parse":
                self._send_json(self.service.parse_lab_text(str(payload["text"])))
                return
            if parsed.path == "/escalation/evaluate":
                self._send_json(
                    self.service.evaluate_escalation(
                        daily_state=payload.get("daily_state"),
                        free_text=payload.get("free_text"),
                    )
                )
                return
            if parsed.path == "/network/experiment-plan":
                self._send_json(
                    self.service.design_network_experiment(
                        experiment_id=str(payload["experiment_id"]),
                        user_ids=list(payload["user_ids"]),
                        edges=payload.get("edges") or [],
                        treatment_arms=payload.get("treatment_arms") or ["control", "treatment"],
                    )
                )
                return
        except KeyError as error:
            self._send_json({"error": f"missing required field: {error.args[0]}"}, HTTPStatus.BAD_REQUEST)
            return
        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _authorized(self) -> bool:
        try:
            self.auth.assert_authorized(self.headers.get("Authorization"))
            return True
        except PermissionError:
            self._send_json({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return False

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local health intelligence API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--db-path",
        default=os.environ.get("KRUNAL_HEALTH_DB", ":memory:"),
        help="SQLite path. Use KRUNAL_HEALTH_DB to set this without a flag.",
    )
    args = parser.parse_args()

    service = HealthIntelligenceService(args.db_path)
    HealthAPIHandler.service = service
    HealthAPIHandler.auth = BearerTokenAuth()
    server = ThreadingHTTPServer((args.host, args.port), HealthAPIHandler)
    try:
        print(f"health API listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
