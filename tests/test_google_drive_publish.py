"""Tests for Google Drive publish adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.publish import drive


def test_load_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "{}")
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "folder-123")

    config = drive.load_config_from_env()

    assert config is not None
    assert config.folder_id == "folder-123"


def test_deliver_video_uploads_file(tmp_path: Path) -> None:
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake-video")

    config = drive.DriveConfig(
        credentials_json="{}",
        folder_id="folder-123",
    )

    service = MagicMock()
    service.files.return_value.create.return_value.execute.return_value = {
        "id": "123",
        "name": video.name,
        "webViewLink": "https://drive.google.com/file/d/123/view",
    }

    with patch("core.publish.drive.build_drive_service", return_value=service):
        result = drive.deliver_video(video, config=config)

    assert result["id"] == "123"
    create_call = service.files.return_value.create.call_args
    assert create_call is not None
    assert create_call.kwargs["media_body"].mimetype() == "video/mp4"
    assert create_call.kwargs["body"]["name"] == video.name
    assert create_call.kwargs["body"]["parents"] == ["folder-123"]


def test_deliver_video_uses_topic_as_description(tmp_path: Path) -> None:
    video = tmp_path / "final.mp4"
    video.write_bytes(b"fake-video")

    payload_path = tmp_path / "pipeline_payload.json"
    payload_path.write_text('{"publish": {"topic": "Cách uống nước đúng"}}', encoding="utf-8")

    service = MagicMock()
    service.files.return_value.create.return_value.execute.return_value = {"id": "123"}

    with patch("core.publish.drive.build_drive_service", return_value=service):
        drive.deliver_video(video, payload_path=payload_path, config=drive.DriveConfig("{}", "folder-123"))

    create_call = service.files.return_value.create.call_args
    assert create_call is not None
    assert create_call.kwargs["body"]["description"] == "Cách uống nước đúng"
