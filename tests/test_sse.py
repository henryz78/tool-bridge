"""Tests for toolbridge.sse.TriggerScanner."""

import unittest
from toolbridge.sse import TriggerScanner, ScanResult


class TestTriggerScannerBasic(unittest.TestCase):
    def test_marker_found_immediately(self):
        scanner = TriggerScanner("[[CALL-abc123]]")
        result = scanner.feed("Before [[CALL-abc123]] after")
        self.assertTrue(result.activated)
        self.assertIn("Before", result.prefix_text)

    def test_marker_in_chunks(self):
        marker = "[[CALL-abc123]]"
        scanner = TriggerScanner(marker)
        r1 = scanner.feed("Hello ")
        self.assertFalse(r1.activated)
        r2 = scanner.feed(marker)
        self.assertTrue(r2.activated)

    def test_marker_across_chunks(self):
        marker = "[[CALL-abc123]]"
        scanner = TriggerScanner(marker)
        r1 = scanner.feed("Hello [[CA")
        self.assertFalse(r1.activated)
        r2 = scanner.feed("LL-abc123]] rest")
        self.assertTrue(r2.activated)

    def test_no_marker(self):
        scanner = TriggerScanner("[[CALL-abc123]]")
        scanner.feed("Hello world")
        # Marker not found — pending_prefix holds the text
        self.assertFalse(scanner.found)
        self.assertIn("Hello", scanner.pending_prefix)


class TestTriggerScannerThinkBlock(unittest.TestCase):
    def test_marker_inside_think_ignored(self):
        marker = "[[CALL-abc123]]"
        scanner = TriggerScanner(marker)
        text = "Hello <thinking>ignore %s here</thinking> real %s after" % (marker, marker)
        r = scanner.feed(text)
        self.assertTrue(r.activated)
        # The prefix should contain "Hello" and "real" but NOT "ignore"
        self.assertIn("real", r.prefix_text)

    def test_think_block_then_marker(self):
        marker = "[[CALL-abc123]]"
        scanner = TriggerScanner(marker)
        r1 = scanner.feed("<thinking>inner</thinking>")
        # No marker yet
        self.assertFalse(r1.activated)
        # Feed the marker after think block
        r2 = scanner.feed(" real %s after" % marker)
        self.assertTrue(r2.activated)

    def test_nested_think(self):
        marker = "[[CALL-abc123]]"
        scanner = TriggerScanner(marker)
        text = "<thinking>outer <thinking>inner</thinking> still outer </thinking> %s after" % marker
        r = scanner.feed(text)
        self.assertTrue(r.activated)


class TestTriggerScannerAfterFound(unittest.TestCase):
    def test_subsequent_feeds(self):
        marker = "[[CALL-abc123]]"
        scanner = TriggerScanner(marker)
        r1 = scanner.feed("prefix %s" % marker)
        self.assertTrue(r1.activated)
        r2 = scanner.feed("more content")
        self.assertTrue(r2.activated)
        self.assertEqual(r2.prefix_text, "")

    def test_accumulated(self):
        marker = "[[CALL-abc123]]"
        scanner = TriggerScanner(marker)
        scanner.feed("prefix %s tool_data" % marker)
        self.assertIn("tool_data", scanner.accumulated)


if __name__ == "__main__":
    unittest.main()
