"""
Gyan HTTP Server — wraps GyanEngine with REST API.

Endpoints:
  POST /query          — ask a question
  POST /learn          — teach a Q&A pair
  POST /feedback       — boost/penalize
  GET  /stats          — collection stats
  GET  /health         — health check
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from gyan_engine import GyanEngine

DB_PATH = os.environ.get("GYAN_DB", os.path.expanduser(
    "~/webmind-research/trained_model/saqt.db"
))
QDRANT_PATH = os.environ.get("GYAN_QDRANT_PATH", None)
PORT = int(os.environ.get("GYAN_PORT", "3003"))
AUTO_LOAD = os.environ.get("GYAN_AUTO_LOAD", "1") == "1"

engine = None


def get_engine():
    global engine
    if engine is None:
        engine = GyanEngine(qdrant_path=QDRANT_PATH)
        if AUTO_LOAD and engine.stats()["points"] == 0:
            engine.load_from_sqlite(DB_PATH)
    return engine


class Handler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        try:
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        eng = get_engine()

        if path == "/health":
            self._json({"status": "ok", "points": eng.stats()["points"]})
        elif path == "/stats":
            self._json(eng.stats())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        eng = get_engine()

        try:
            body = self._read_body()
        except Exception as e:
            self._json({"error": str(e)}, 400)
            return

        if path == "/query":
            question = body.get("question", "")
            context = body.get("context", "")
            if not question:
                self._json({"error": "question required"}, 400)
                return
            result = eng.query(question, context)
            self._json(result)

        elif path == "/learn":
            question = body.get("question", "")
            answer = body.get("answer", "")
            source = body.get("source", "learned")
            if not question or not answer:
                self._json({"error": "question and answer required"}, 400)
                return
            result = eng.learn(question, answer, source)
            self._json(result)

        elif path == "/feedback":
            point_id = body.get("id")
            positive = body.get("positive", True)
            if point_id is None:
                self._json({"error": "id required"}, 400)
                return
            eng.feedback(int(point_id), positive)
            self._json({"status": "ok"})

        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        # Quieter logging
        pass


def main():
    eng = get_engine()
    stats = eng.stats()
    print(f"Gyan Engine — {stats['points']} knowledge pairs")
    print(f"Serving on http://0.0.0.0:{PORT}")

    server = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
