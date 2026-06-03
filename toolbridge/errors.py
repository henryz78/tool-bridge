"""Custom exception hierarchy for the bridge."""


class BridgeError(Exception):
    """Base exception for all bridge errors."""


class ConfigurationError(BridgeError):
    """Raised when an environment variable is invalid or missing."""


class UpstreamError(BridgeError):
    """Raised when the upstream API returns a non-success response."""

    def __init__(self, status: int, body: bytes, headers: dict | None = None):
        self.status = status
        self.body = body
        self.headers = headers or {}
        super().__init__(f"upstream returned {status}")


class ParseError(BridgeError):
    """Raised when tool invocation output cannot be parsed."""
