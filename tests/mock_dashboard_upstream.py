#!/usr/bin/env python3
"""Mock upstream standing in for the Flask dashboard during a proxy test."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Echo back what the proxy told us about the client, so the test can
        # confirm the real-IP headers survive.
        self._reply(200, {
            "path": self.path,
            "x_real_ip": self.headers.get("X-Real-IP"),
            "x_forwarded_for": self.headers.get("X-Forwarded-For"),
            "auth_header_present": "Authorization" in self.headers,
        })

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        # The real endpoint would queue a USDC transfer here.
        self._reply(200, {"path": self.path, "status": "WITHDRAWAL WOULD EXECUTE"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 18080), Handler).serve_forever()
