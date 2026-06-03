"""Embedded application icon for system tray."""

from __future__ import annotations

import base64
import io

_ICON_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAdklEQVR4nO2TwQ2AMAwD"
    "TZRRmKVj8WCszsIu8C8SJCEoj/qekeO6cgsQQsjsLOOg7cc5zvq23nRW3vwExQgDINhj"
    "1p5GzJ4epTesesTRQz5X0IPf0LKnXjPL7T2B1SqMmKcFaImdj8hfxgxgRVCMMABmr4Cg"
    "mgsoGyLTSRC3dAAAAABJRU5ErkJggg=="
)


def load_icon():
    """Load the embedded icon as a PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(_ICON_B64)))
