"""Tests for toolbridge authentication and bind safety helpers."""

import unittest

from toolbridge.auth import is_public_bind_host, validate_public_bind_auth
from toolbridge.config import Settings
from toolbridge.server import create_server


class TestPublicBindSafety(unittest.TestCase):
    def test_loopback_hosts_are_not_public(self) -> None:
        for host in ("127.0.0.1", "::1", "localhost"):
            with self.subTest(host=host):
                self.assertFalse(is_public_bind_host(host))

    def test_unspecified_and_lan_hosts_are_public(self) -> None:
        for host in ("0.0.0.0", "::", "192.168.1.20", "example.com"):
            with self.subTest(host=host):
                self.assertTrue(is_public_bind_host(host))

    def test_public_bind_rejects_partial_token_configuration(self) -> None:
        settings = Settings(listen_host="0.0.0.0", admin_token="admin-secret")

        with self.assertRaisesRegex(ValueError, "BRIDGE_API_KEY"):
            validate_public_bind_auth(settings)

    def test_public_bind_allows_missing_tokens_for_initial_setup(self) -> None:
        validate_public_bind_auth(Settings(listen_host="0.0.0.0"))

    def test_public_bind_allows_both_tokens(self) -> None:
        settings = Settings(
            listen_host="0.0.0.0",
            admin_token="admin-secret",
            bridge_api_key="bridge-secret",
        )

        validate_public_bind_auth(settings)

    def test_loopback_bind_allows_missing_tokens(self) -> None:
        validate_public_bind_auth(Settings(listen_host="127.0.0.1"))

    def test_create_server_allows_public_bind_for_initial_setup(self) -> None:
        server = create_server(Settings(listen_host="0.0.0.0", listen_port=0))
        server.server_close()


if __name__ == "__main__":
    unittest.main()
