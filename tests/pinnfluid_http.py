"""Shared helpers of the web-app tests: a server on a free port and a browser-like client.

Not a test module (unittest discovers test*.py only); the tests import it after putting
tests/ on sys.path.
"""

from __future__ import annotations

import http.client
import json
import threading


class Server:
    """app.Handler (or a subclass) served on 127.0.0.1 and a free port, in a thread."""

    def __init__(self, app, handler=None):
        self.app = app
        self.httpd = app.http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler or app.Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.address = self.httpd.server_address

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class Browser:
    """An HTTP client with a cookie jar of one cookie, like one browser profile."""

    def __init__(self, server: Server, cookie: str | None = None):
        self.host, self.port = server.address
        self.cookie = cookie

    def request(self, method: str, path: str, body=None, *, raw: bytes | None = None,
                timeout: float = 60):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if self.cookie:
            headers["Cookie"] = self.cookie
        try:
            data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
            conn.request(method, path, body=data, headers=headers)
            r = conn.getresponse()
            payload = r.read()
            set_cookie = r.getheader("Set-Cookie")
            if set_cookie:
                self.cookie = set_cookie.split(";", 1)[0]
            return r.status, r.headers, payload
        finally:
            conn.close()

    def json(self, method: str, path: str, body=None):
        status, _, payload = self.request(method, path, body)
        return status, json.loads(payload)
