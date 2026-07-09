"""Platform adapter registry for multi-platform publish."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.publish import drive, facebook, telegram

ADAPTERS: dict[str, Callable[..., dict[str, Any] | None]] = {
    "facebook": facebook.deliver_video,
    "telegram": telegram.deliver_video,
    "drive": drive.deliver_video,
}


def parse_platform_list(raw: str | None) -> list[str]:
    """Parse a comma-separated platform list (lowercase, deduplicated)."""
    if not raw or not raw.strip():
        return []
    seen: set[str] = set()
    platforms: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        platforms.append(name)
    return platforms


def get_enabled_platforms(*, cli_override: str | None = None) -> list[str]:
    """Return platform names from CLI override, env override, or auto-detect Drive."""
    if cli_override is not None:
        return parse_platform_list(cli_override)

    raw = os.environ.get("PUBLISH_PLATFORMS", "").strip()
    if raw:
        return parse_platform_list(raw)

    if drive.load_config_from_env() is not None:
        return ["drive"]
    return []


def deliver_to_platforms(
    video_path: str | Path,
    platforms: list[str],
    *,
    jobs_csv: str | Path | None = None,
    caption: str | None = None,
    payload_path: str | Path | None = None,
) -> list[tuple[str, dict[str, Any] | None]]:
    """Run each enabled platform adapter sequentially. Returns (name, result) pairs."""
    results: list[tuple[str, dict[str, Any] | None]] = []
    for name in platforms:
        adapter = ADAPTERS.get(name)
        if adapter is None:
            print(f"[publish] warning: unknown platform '{name}' — skipped")
            continue
        result = adapter(
            video_path,
            jobs_csv=jobs_csv,
            caption=caption,
            payload_path=payload_path,
        )
        results.append((name, result))
    return results
