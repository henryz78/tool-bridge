"""Model name resolution and mapping utilities."""

from __future__ import annotations

import re

_HINT_SUFFIX = re.compile(r"\[\d+[km]\]$", re.IGNORECASE)


def strip_model_hints(name: str) -> str:
    """Remove context-window hint suffixes like ``[1m]`` or ``[128k]``."""
    return _HINT_SUFFIX.sub("", name).strip()


def resolve_model(requested: str, name_mapping: dict[str, str], allow_unmapped: bool) -> str | None:
    """Map an external model name to the upstream name.

    Returns ``None`` when the model is not in *name_mapping* and
    *allow_unmapped* is ``False``.
    """
    clean = strip_model_hints(requested)
    if clean in name_mapping:
        return name_mapping[clean]
    if allow_unmapped:
        return clean
    return None


def is_native_model(
    requested: str,
    name_mapping: dict[str, str],
    native_ids: set[str],
    allow_unmapped: bool,
) -> bool:
    """Return ``True`` if the resolved model supports native tool calling."""
    resolved = resolve_model(requested, name_mapping, allow_unmapped)
    if resolved is None:
        return False
    return resolved in native_ids


def collect_exposed_ids(
    name_mapping: dict[str, str],
    native_ids: set[str],
    explicit_ids: list[str],
) -> list[str]:
    """Build the list of model IDs to expose via ``/v1/models``."""
    seen: set[str] = set()
    result: list[str] = []
    for mid in explicit_ids:
        if mid not in seen:
            seen.add(mid)
            result.append(mid)
    for ext_name in name_mapping:
        if ext_name not in seen:
            seen.add(ext_name)
            result.append(ext_name)
    for up_name in name_mapping.values():
        if up_name not in seen:
            seen.add(up_name)
            result.append(up_name)
    return result
