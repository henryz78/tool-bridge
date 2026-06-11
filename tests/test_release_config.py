"""Tests for deployment-oriented project configuration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestReleaseConfiguration(unittest.TestCase):
    def test_dockerfile_defaults_to_public_container_bind(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("ENV HOST=0.0.0.0", dockerfile)
        self.assertIn("ENV PYTHONUNBUFFERED=1", dockerfile)
        self.assertIn('CMD ["python", "-m", "toolbridge"]', dockerfile)

    def test_railway_config_uses_healthcheck(self) -> None:
        path = ROOT / "railway.json"
        self.assertTrue(path.exists(), "railway.json is missing")
        railway = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(railway["build"]["builder"], "DOCKERFILE")
        self.assertEqual(railway["build"]["dockerfilePath"], "Dockerfile")
        self.assertEqual(railway["deploy"]["healthcheckPath"], "/health")

    def test_compose_example_exposes_required_public_bind_tokens(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("ADMIN_TOKEN=", compose)
        self.assertIn("BRIDGE_API_KEY=", compose)

    def test_dashboard_exposes_security_token_inputs(self) -> None:
        dashboard = (ROOT / "toolbridge" / "dashboard.html").read_text(encoding="utf-8")

        self.assertIn('id="admin-token"', dashboard)
        self.assertIn('id="bridge-api-key"', dashboard)
        self.assertIn("ADMIN_TOKEN:", dashboard)
        self.assertIn("BRIDGE_API_KEY:", dashboard)
        self.assertIn("result.error || '保存配置响应异常'", dashboard)


if __name__ == "__main__":
    unittest.main()
