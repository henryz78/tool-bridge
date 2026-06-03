"""Tests for toolbridge.config_file."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from toolbridge import config_file


class TestLoadConfig(unittest.TestCase):
    @mock.patch.object(config_file, "CONFIG_PATH")
    def test_missing_file(self, mock_path: mock.MagicMock) -> None:
        mock_path.read_text.side_effect = FileNotFoundError
        self.assertIsNone(config_file.load_config())

    @mock.patch.object(config_file, "CONFIG_PATH")
    def test_invalid_json(self, mock_path: mock.MagicMock) -> None:
        mock_path.read_text.return_value = "not json"
        self.assertIsNone(config_file.load_config())

    @mock.patch.object(config_file, "CONFIG_PATH")
    def test_non_dict_json(self, mock_path: mock.MagicMock) -> None:
        mock_path.read_text.return_value = "[1, 2, 3]"
        self.assertIsNone(config_file.load_config())

    @mock.patch.object(config_file, "CONFIG_PATH")
    def test_valid_config(self, mock_path: mock.MagicMock) -> None:
        data = {"UPSTREAM_BASE_URL": "http://localhost:3000", "PORT": 9090}
        mock_path.read_text.return_value = json.dumps(data)
        result = config_file.load_config()
        self.assertEqual(result, data)


class TestSaveConfig(unittest.TestCase):
    def test_roundtrip(self) -> None:
        data = {"PORT": 9999, "MODEL_MAP_JSON": {"a": "b"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with mock.patch.object(config_file, "CONFIG_DIR", Path(tmp)), \
                 mock.patch.object(config_file, "CONFIG_PATH", path):
                config_file.save_config(data)
                loaded = config_file.load_config()
        self.assertEqual(loaded, data)

    def test_creates_dir(self) -> None:
        data = {"PORT": 1234}
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp) / "sub"
            path = cfg_dir / "config.json"
            with mock.patch.object(config_file, "CONFIG_DIR", cfg_dir), \
                 mock.patch.object(config_file, "CONFIG_PATH", path):
                config_file.save_config(data)
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
