"""Session cookies are issued and signed by the server; it accepts no other.

A well-formed id that the server did not issue (a forged or stale one) is replaced
by a fresh, signed one. This does not stop a page of the same origin from planting
a cookie the server did issue (to itself): pages of one origin share cookies, so
the origin is the boundary. A request with two different session cookies gets a
fresh session. The signature is an HMAC with a per-instance random
secret, or PINN_WEBAPP_SESSION_SECRET when instances must share sessions.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "pinnfluid", ROOT / "pinnfluid" / "webapp", ROOT / "pinnfluid" / "domain_prep",
             ROOT / "pinnfluid" / "input_prep", Path(__file__).resolve().parent):
    sys.path.insert(0, str(path))

from webapp import app  # noqa: E402
from pinnfluid_http import Browser, Server  # noqa: E402

VALUE_RE = re.compile(r"^[0-9a-f]{32}\.[0-9a-f]{32}$")
KNOWN_ID = "0123456789abcdef0123456789abcdef"


class SignedSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.Handler.page_html = app._build_html()
        cls.server = Server(app)

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def setUp(self):
        env = patch.dict(os.environ, {"PINN_WEBAPP_SECURE_COOKIE": "0"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("PINN_WEBAPP_SESSION_SECRET", None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(app, "RESULTS_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def value(self, browser: Browser) -> str:
        return browser.cookie.split("=", 1)[1]

    def test_the_server_issues_signed_ids_and_keeps_them(self):
        browser = Browser(self.server)
        browser.request("GET", "/")
        self.assertRegex(self.value(browser), VALUE_RE)
        first = browser.cookie
        _, headers, _ = browser.request("GET", "/")
        self.assertIsNone(headers.get("Set-Cookie"))
        self.assertEqual(browser.cookie, first)

    def test_ids_it_did_not_issue_are_replaced(self):
        issued = Browser(self.server)
        issued.request("GET", "/")
        issued_id, mac = self.value(issued).split(".")
        other_mac = ("0" if mac[0] != "0" else "1") + mac[1:]
        for forged in (KNOWN_ID, f"{KNOWN_ID}.{mac}", f"{issued_id}.{other_mac}", f"{issued_id}.",
                       f"{issued_id}.{mac}.x", issued_id):
            browser = Browser(self.server, f"pinnfluid_session={forged}")
            _, headers, _ = browser.request("GET", "/")
            self.assertIsNotNone(headers.get("Set-Cookie"), forged)
            self.assertRegex(self.value(browser), VALUE_RE)
            self.assertNotEqual(self.value(browser).split(".")[0], forged.split(".")[0], forged)

    def test_a_forged_id_gives_no_known_namespace(self):
        """A cookie set to an id the server did not sign starts a fresh session, so
        whoever set it cannot compute where that browser's runs are."""
        victim = Browser(self.server, f"pinnfluid_session={KNOWN_ID}.{'0' * 32}")
        victim.request("GET", "/")
        self.assertNotEqual(app._cookie_namespace(self.value(victim)), app._session_namespace(KNOWN_ID))
        self.assertIsNone(app._cookie_namespace(f"{KNOWN_ID}.{'0' * 32}"))

    def test_another_instance_does_not_accept_the_cookie_unless_it_shares_the_secret(self):
        browser = Browser(self.server)
        browser.request("GET", "/")
        value = self.value(browser)
        with patch.object(app, "_INSTANCE_SECRET", os.urandom(32)):  # a restarted instance
            self.assertIsNone(app._cookie_namespace(value))
        with patch.dict(os.environ, {"PINN_WEBAPP_SESSION_SECRET": "shared-secret-of-the-deployment"}):
            shared = app._sign_session(KNOWN_ID)
            with patch.object(app, "_INSTANCE_SECRET", os.urandom(32)):
                self.assertEqual(app._cookie_namespace(shared), app._session_namespace(KNOWN_ID))

    def test_two_different_session_cookies_start_a_fresh_session(self):
        """A browser that sends two valid session cookies (one planted by another page
        of the same origin, for example with a longer Path) gets a fresh session, so
        its work lands in neither: the server cannot tell which one is its own."""
        own, planted = Browser(self.server), Browser(self.server)
        own.request("GET", "/")
        planted.request("GET", "/")
        both = Browser(self.server, f"{planted.cookie}; {own.cookie}")
        _, headers, _ = both.request("GET", "/")
        self.assertIsNotNone(headers.get("Set-Cookie"))
        fresh = app._cookie_namespace(self.value(both))
        self.assertNotIn(fresh, (app._cookie_namespace(self.value(own)),
                                 app._cookie_namespace(self.value(planted))))
        # the same cookie twice is not ambiguous
        twice = Browser(self.server, f"{own.cookie}; {own.cookie}")
        _, headers, _ = twice.request("GET", "/")
        self.assertIsNone(headers.get("Set-Cookie"))

    def test_a_short_shared_secret_is_reported(self):
        with patch.dict(os.environ, {"PINN_WEBAPP_SESSION_SECRET": "too-short"}):
            self.assertIn("at least 32 bytes", app._session_secret_problem())
        with patch.dict(os.environ, {"PINN_WEBAPP_SESSION_SECRET": "x" * 32}):
            self.assertIsNone(app._session_secret_problem())
        self.assertIsNone(app._session_secret_problem())  # unset: a random key of the process

    def test_job_status_of_an_unknown_job_also_starts_a_session(self):
        browser = Browser(self.server)
        status, headers, _ = browser.request("GET", "/job_status?id=000000000000")
        self.assertEqual(status, 404)
        self.assertIsNotNone(headers.get("Set-Cookie"))


if __name__ == "__main__":
    unittest.main()
