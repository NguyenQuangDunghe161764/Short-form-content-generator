"""Tests for multi-platform publish runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.publish.registry import get_enabled_platforms, parse_platform_list
from core.publish_runner import publish_video


def test_parse_platform_list_deduplicates_and_lowercases() -> None:
    assert parse_platform_list("Facebook, telegram, FACEBOOK") == ["facebook", "telegram"]


def test_get_enabled_platforms_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLISH_PLATFORMS", "facebook,telegram")
    assert get_enabled_platforms() == ["facebook", "telegram"]


def test_get_enabled_platforms_cli_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLISH_PLATFORMS", "telegram")
    assert get_enabled_platforms(cli_override="facebook") == ["facebook"]


def test_publish_video_skips_when_no_platforms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PUBLISH_PLATFORMS", raising=False)
    video = tmp_path / "final.mp4"
    video.write_bytes(b"mp4")

    assert publish_video(video) is True
    assert "skipped" in capsys.readouterr().out


@patch("core.publish_runner.ADAPTERS", {"facebook": MagicMock(return_value={"ok": True})})
def test_publish_video_success(tmp_path: Path) -> None:
    video = tmp_path / "final.mp4"
    video.write_bytes(b"mp4")

    assert publish_video(video, platforms=["facebook"]) is True


def test_publish_video_auto_uses_drive_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PUBLISH_PLATFORMS", raising=False)
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "{}")
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "folder-123")

    video = tmp_path / "final.mp4"
    video.write_bytes(b"mp4")

    mock_drive = MagicMock(return_value={"ok": True})
    with patch("core.publish_runner.ADAPTERS", {"drive": mock_drive}):
        assert publish_video(video) is True

    mock_drive.assert_called_once()


@patch("core.publish_runner.ADAPTERS")
def test_publish_video_failure_returns_false(
    mock_adapters: MagicMock,
    tmp_path: Path,
) -> None:
    from core.publish.common import PublishError

    video = tmp_path / "final.mp4"
    video.write_bytes(b"mp4")
    mock_adapters.get.return_value.side_effect = PublishError("API failed")

    assert publish_video(video, platforms=["facebook"]) is False


@patch("core.publish_runner.ADAPTERS", {})
def test_publish_video_warns_on_unknown_platform(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "final.mp4"
    video.write_bytes(b"mp4")

    assert publish_video(video, platforms=["unknown"]) is True
    assert "unknown platform" in capsys.readouterr().out
