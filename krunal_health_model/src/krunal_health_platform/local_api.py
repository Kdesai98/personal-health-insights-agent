"""Minimal local HTTP API around the health intelligence service.

This is intentionally standard-library only. It gives the app a concrete API
contract now; it can be swapped for FastAPI when hosting decisions are locked.
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from krunal_health_engine.daily_health_engine import build_daily_state

from .service import HealthIntelligenceService


class HealthAPIHandler(BaseHTTPRequestHandler):
    service: HealthIntelligenceService

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/learning-summary":
            query = parse_qs(parsed.query)
            user_id_hash = query.get("user_id_hash", ["krunal_demo_user"])[0]
            self._send_json(self.service.build_learning_summary(user_id_hash))
            return
        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
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
        except KeyError as error:
            self._send_json({"error": f"missing required field: {error.args[0]}"}, HTTPStatus.BAD_REQUEST)
            return
        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

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
