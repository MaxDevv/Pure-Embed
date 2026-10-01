"""Serve the browser benchmark and persist submissions in benchmark/results/."""
import json
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "results"


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.path = "/benchmark/browser.html"
        super().do_GET()

    def do_POST(self):
        if self.path != "/results":
            self.send_error(404)
            return
        try:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
                raise ValueError("Expected benchmark result")
        except (ValueError, KeyError) as error:
            self.send_error(400, str(error))
            return
        RESULTS.mkdir(exist_ok=True)
        filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8] + ".json"
        (RESULTS / filename).write_text(json.dumps(data, indent=2) + "\n")
        body = json.dumps({"file": filename}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), partial(Handler, directory=str(ROOT)))
    print("Browser benchmark: http://127.0.0.1:8765", flush=True)
    server.serve_forever()
