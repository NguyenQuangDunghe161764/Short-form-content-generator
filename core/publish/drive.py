"""Google Drive publish adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.publish.common import PublishError, assert_video_exists


@dataclass(frozen=True)
class DriveConfig:
    credentials_json: str
    folder_id: str


def load_config_from_env() -> DriveConfig | None:
    """Return config when Google Drive env vars are set; otherwise None."""
    credentials_json = os.environ.get("GOOGLE_DRIVE_CREDENTIALS_JSON", "").strip()
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    if not credentials_json or not folder_id:
        return None
    return DriveConfig(credentials_json=credentials_json, folder_id=folder_id)


def build_drive_service(credentials_json: str) -> Any:
    """Build a Google Drive API service from a service account JSON string."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover - exercised in runtime envs without deps
        raise PublishError(
            "google-api-python-client / google-auth packages are required for Drive upload"
        ) from exc

    try:
        creds_info = json.loads(credentials_json)
    except json.JSONDecodeError as exc:
        raise PublishError("GOOGLE_DRIVE_CREDENTIALS_JSON is not valid JSON") from exc

    scopes = ["https://www.googleapis.com/auth/drive.file"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=scopes,
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _resolve_upload_caption(
    video_path: Path,
    *,
    caption: str | None = None,
    payload_path: str | Path | None = None,
) -> str | None:
    if caption is not None:
        return caption.strip() or None

    if payload_path is None:
        return None

    path = Path(payload_path)
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    publish = payload.get("publish")
    if isinstance(publish, dict):
        topic = publish.get("topic")
        if isinstance(topic, str) and topic.strip():
            return topic.strip()

    topic = payload.get("topic")
    if isinstance(topic, str) and topic.strip():
        return topic.strip()

    return None


def deliver_video(
    video_path: str | Path,
    *,
    jobs_csv: str | Path | None = None,
    caption: str | None = None,
    payload_path: str | Path | None = None,
    config: DriveConfig | None = None,
) -> dict[str, Any] | None:
    """Upload a rendered MP4 to Google Drive in the configured folder."""
    resolved_config = config or load_config_from_env()
    if resolved_config is None:
        print("[drive] skipped (GOOGLE_DRIVE_CREDENTIALS_JSON / GOOGLE_DRIVE_FOLDER_ID not set)")
        return None

    path = Path(video_path)
    assert_video_exists(path)

    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:  # pragma: no cover - exercised in runtime envs without deps
        raise PublishError(
            "google-api-python-client / google-auth packages are required for Drive upload"
        ) from exc

    service = build_drive_service(resolved_config.credentials_json)
    mime_type = "video/mp4"
    media = MediaFileUpload(
        str(path),
        mimetype=mime_type,
        resumable=False,
    )
    body: dict[str, Any] = {
        "name": path.name,
        "parents": [resolved_config.folder_id],
    }
    if caption is not None:
        resolved_caption = caption.strip() or None
        if resolved_caption:
            body["description"] = resolved_caption
    else:
        resolved_caption = _resolve_upload_caption(
            path,
            caption=caption,
            payload_path=payload_path,
        )
        if resolved_caption:
            body["description"] = resolved_caption

    result = (
        service.files()
        .create(
            body=body,
            media_body=media,
            fields="id,name,webViewLink",
        )
        .execute()
    )
    print(f"[drive] uploaded video — {path.resolve()}")
    return result
