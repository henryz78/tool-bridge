"""Tests for toolbridge.autostart (mocked platform calls)."""

import sys
import unittest
from unittest import mock

from toolbridge import autostart


class TestAutostartInterface(unittest.TestCase):
    def test_enable_disable_no_error(self) -> None:
        # Smoke test: should not raise regardless of platform
        with mock.patch.object(autostart, "_win_is_enabled", return_value=False), \
             mock.patch.object(autostart, "_mac_is_enabled", return_value=False), \
             mock.patch.object(autostart, "_linux_is_enabled", return_value=False):
            result = autostart.is_autostart_enabled()
        self.assertFalse(result)

    def test_dispatches_to_platform(self) -> None:
        with mock.patch.object(autostart, "_win_enable") as m:
            with mock.patch("sys.platform", "win32"):
                autostart.enable_autostart()
                m.assert_called_once()

        with mock.patch.object(autostart, "_mac_enable") as m:
            with mock.patch("sys.platform", "darwin"):
                autostart.enable_autostart()
                m.assert_called_once()

        with mock.patch.object(autostart, "_linux_enable") as m:
            with mock.patch("sys.platform", "linux"):
                autostart.enable_autostart()
                m.assert_called_once()


if __name__ == "__main__":
    unittest.main()
