"""Master Controller — sequences script-writer and caption-styler LLMs, TTS synthesis, and payload assembly; CLI entry point for the MVP pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import APIError, OpenAI

from core.audio_generator import synthesize_speech
from core.caption_tokens import VALID_ANIMATIONS, VALID_STYLES, merge_styled_tokens_with_timestamps
from core.music_picker import attach_random_music
from core.project_schema import VALID_CAPTION_MODES

SCRIPT_WRITER_MODEL = "gemini-2.5-flash"
CAPTION_STYLER_MODEL = "gemini-2.5-flash"
DEFAULT_TTS_VOICE = "vi-VN-HoaiMyNeural"

SCRIPT_WRITER_SYSTEM_PROMPT = (
    "You are a viral short-form copywriter for Vietnamese audiences. "
    "Write punchy, high-retention narration scripts in Vietnamese for TikTok, Reels, and YouTube Shorts. "
    "Use a strong hook in the first line, short sentences, and a clear payoff. "
    "Write entirely in Vietnamese — natural spoken Vietnamese, not English. "
    "Output only the spoken script text — no titles, labels, or markdown."
)

CAPTION_STYLER_SYSTEM_PROMPT = """You are a caption styler for short-form video typography.

The narration script is in Vietnamese. Parse it into a raw dictionary object with a root key "tokens".

"tokens" must be a JSON array containing one object for every single individual word in the script, in spoken order.

Each word object must use these exact keys (same schema as caption_renderer):
- "text": the individual Vietnamese word as spoken (preserve punctuation attached to the word)
- "style": "primary" or "highlight" (use highlight for hook words, stats, and emotional peaks)
- "animation": "none" or "pop" (use pop sparingly on the highest-impact words)

Rules:
- Split on whitespace; every word gets its own object — do not merge phrases.
- Keep Vietnamese diacritics exactly as in the script (e.g. uống, không, được).
- Output valid JSON only with the root shape: {"tokens": [...]}
"""

_env_loaded = False
API_RETRY_ATTEMPTS = 2
API_RETRY_DELAY_SECONDS = 5

T = TypeVar("T")


class PipelineError(RuntimeError):
    """Raised when an LLM or TTS pipeline stage fails."""


def _load_env() -> None:
    """Load environment variables from `.env` once per process.

    Goal: Make API keys and config available before any pipeline stage runs.
    Params: None.
    Output: None; sets os.environ from the project-root .env file if present.
    """
    global _env_loaded
    if _env_loaded:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    _env_loaded = True


def _default_caption_mode() -> str:
    """Caption overlay default: ``CAPTION_MODE`` env, else ``none``."""
    mode = os.environ.get("CAPTION_MODE", "none").strip() or "none"
    return mode if mode in VALID_CAPTION_MODES else "none"


def _require_env(*names: str) -> None:
    """Verify required environment variables are set and non-empty.

    Goal: Fail fast before calling external APIs with missing credentials.
    Params: names — one or more environment variable names to check.
    Output: None; raises RuntimeError listing any missing variables.
    """
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s): {joined}. "
            "Copy .env.example to .env and fill in the values."
        )


def _get_client() -> OpenAI:
    """Build an OpenAI-compatible client from environment configuration.

    Goal: Shared LLM client for script writer and caption styler stages.
    Params: None.
    Output: Configured OpenAI client instance.
    """
    _load_env()
    _require_env("OPENAI_API_KEY", "OPENAI_BASE_URL")
    timeout = float(os.environ.get("OPENAI_TIMEOUT", "60"))
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=timeout,
    )


def _model_from_env(env_var: str, default: str) -> str:
    """Read an LLM model id from the environment with a fallback default.

    Goal: Allow per-stage model overrides without hardcoding in call sites.
    Params: env_var — environment variable name; default — value if unset.
    Output: Model id string.
    """
    return os.environ.get(env_var, default)


def _tts_voice_from_env() -> str:
    """Resolve the edge-tts voice name from environment.

    Goal: Default TTS voice for narration synthesis.
    Params: None.
    Output: Voice name string (default vi-VN-HoaiMyNeural).
    """
    return os.environ.get("TTS_VOICE", DEFAULT_TTS_VOICE)


def _api_error_text(exc: APIError) -> str:
    """Flatten API error body/message for quota and retry parsing."""
    parts: list[str] = [str(exc)]
    if exc.body is not None:
        parts.append(str(exc.body))
    return "\n".join(parts)


def _is_daily_quota_exhausted(exc: APIError) -> bool:
    """True when Gemini free-tier daily request cap is hit (retry won't help today)."""
    text = _api_error_text(exc)
    return any(
        marker in text
        for marker in (
            "PerDay",
            "free_tier_requests",
            "GenerateRequestsPerDay",
            "exceeded your current quota",
        )
    )


def _retry_delay_for_api_error(exc: APIError, default_seconds: float) -> float:
    """Use provider retry hint when present (e.g. Gemini 429 RetryInfo)."""
    match = re.search(r"retry in ([\d.]+)s", _api_error_text(exc), re.IGNORECASE)
    if match:
        return max(default_seconds, float(match.group(1)) + 1.0)
    return default_seconds


def _call_with_api_retry(
    call: Callable[[], T],
    *,
    stage_name: str,
    attempts: int = API_RETRY_ATTEMPTS,
    delay_seconds: float = API_RETRY_DELAY_SECONDS,
) -> T:
    """Run an OpenAI API call with a fixed delay between retries.

    Goal: Recover from transient provider errors without failing the whole pipeline.
    Params: call — zero-arg callable that performs the API request;
        stage_name — label for error messages; attempts — total tries (default 2);
        delay_seconds — wait before retry (default 5s).
    Output: Return value from call on success.
    """
    last_exc: APIError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except APIError as exc:
            last_exc = exc
            if _is_daily_quota_exhausted(exc):
                break
            if attempt >= attempts:
                break
            time.sleep(_retry_delay_for_api_error(exc, delay_seconds))

    assert last_exc is not None
    if _is_daily_quota_exhausted(last_exc):
        raise PipelineError(
            f"{stage_name} failed: Gemini daily quota exceeded for this API key/model. "
            "Each slideshow uses ~2 LLM calls (scene script + TTS writer). "
            "Wait for the free-tier reset, enable billing in Google AI Studio, "
            "or set SCRIPT_WRITER_MODEL / a new OPENAI_API_KEY. "
            f"Details: {last_exc}"
        ) from last_exc
    raise PipelineError(
        f"{stage_name} API call failed after {attempts} attempt(s): {last_exc}"
    ) from last_exc


def run_script_writer(client: OpenAI, topic_prompt: str) -> str:
    """Generate a Vietnamese narration script from a video topic.

    Goal: Produce raw spoken script text for caption styling and TTS.
    Params: client — OpenAI-compatible client; topic_prompt — user topic or brief.
    Output: Non-empty Vietnamese script string.
    """
    topic = topic_prompt.strip()
    if not topic:
        raise ValueError("topic_prompt must not be empty.")

    model = _model_from_env("SCRIPT_WRITER_MODEL", SCRIPT_WRITER_MODEL)
    response = _call_with_api_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SCRIPT_WRITER_SYSTEM_PROMPT},
                {"role": "user", "content": topic},
            ],
        ),
        stage_name="Script writer",
    )

    raw_script = (response.choices[0].message.content or "").strip()
    if not raw_script:
        raise PipelineError("Script writer returned an empty script.")

    return raw_script


def run_caption_styler(client: OpenAI, raw_script: str) -> str:
    """Generate styled per-word caption tokens as JSON from the script.

    Goal: Produce caption_renderer-compatible tokens (text, style, animation).
    Params: client — OpenAI-compatible client; raw_script — narration text to tokenize.
    Output: JSON string with a root "tokens" array.
    """
    model = _model_from_env("CAPTION_STYLER_MODEL", CAPTION_STYLER_MODEL)
    response = _call_with_api_retry(
        lambda: client.chat.completions.create(
            model=model,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CAPTION_STYLER_SYSTEM_PROMPT},
                {"role": "user", "content": raw_script},
            ],
        ),
        stage_name="Caption styler",
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise PipelineError("Caption styler returned an empty response.")

    return content


def parse_caption_styler_response(content: str) -> list[dict]:
    """Parse caption styler JSON and extract the tokens array.

    Goal: Turn LLM JSON output into a Python list for validation and payload write.
    Params: content — raw JSON string from the caption styler LLM.
    Output: List of token dicts from the "tokens" key.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Caption styler returned invalid JSON: {exc}") from exc

    tokens = parsed.get("tokens") if isinstance(parsed, dict) else None
    if not isinstance(tokens, list):
        raise ValueError('Caption styler JSON must include a "tokens" array.')

    return tokens


def validate_tokens(tokens: list[dict]) -> list[dict]:
    """Validate caption tokens match the shared caption_renderer schema.

    Goal: Reject malformed tokens before TTS and render stages consume them.
    Params: tokens — list of caption token dicts from the caption styler.
    Output: The same list if every token has text and valid style/animation values.
    """
    for index, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise ValueError(f"Token {index} must be an object.")
        text = token.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Token {index} must include a non-empty 'text'.")

        style = token.get("style", "primary")
        if style not in VALID_STYLES:
            raise ValueError(
                f"Token {index} style must be one of {sorted(VALID_STYLES)}; got {style!r}."
            )

        animation = token.get("animation", "none")
        if animation not in VALID_ANIMATIONS:
            raise ValueError(
                f"Token {index} animation must be one of {sorted(VALID_ANIMATIONS)}; got {animation!r}."
            )

    return tokens


def synthesize_narration(text: str, path: Path, voice: str) -> list[dict[str, Any]]:
    """Generate narration MP3 and word-level timestamps via edge-tts.

    Goal: Produce synced audio for caption timing and final video mux.
    Params: text — full narration script; path — output MP3 path; voice — edge-tts voice id.
    Output: List of {text, start_ms, end_ms} timestamp dicts.
    """
    word_timestamps = synthesize_speech(text, path, voice=voice)
    if not word_timestamps:
        raise PipelineError("TTS produced no word boundaries.")

    return word_timestamps


def save_payload(payload: dict[str, Any], path: Path) -> None:
    """Write the pipeline payload to disk as pretty-printed UTF-8 JSON.

    Goal: Persist project state for render stages and future edit UI.
    Params: payload — serializable project dict; path — output JSON file path.
    Output: None; creates parent directories and writes the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_mvp_pipeline(
    topic_prompt: str,
    *,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    """Run script writer → caption styler → TTS → pipeline_payload.json.

    Goal: End-to-end MVP content generation for downstream render stages.
    Params: topic_prompt — video topic; output_dir — folder for narration and JSON artifacts.
    Output: Payload dict with topic, raw_script, tokens, and audio sections.
    """
    client = _get_client()
    out = Path(output_dir)
    voice = _tts_voice_from_env()

    raw_script = run_script_writer(client, topic_prompt)

    print("=" * 60)
    print("SCRIPT WRITER — RAW SCRIPT")
    print("=" * 60)
    print(raw_script)
    print()

    caption_styler_content = run_caption_styler(client, raw_script)
    tokens = parse_caption_styler_response(caption_styler_content)
    tokens = validate_tokens(tokens)

    # print("=" * 60)
    # print("CAPTION STYLER — STRUCTURED TOKEN LIST")
    # print("=" * 60)
    # print(json.dumps(tokens, indent=2, ensure_ascii=False))
    # print()

    narration_path = out / "narration.mp3"
    word_timestamps = synthesize_narration(raw_script, narration_path, voice)

    try:
        tokens = merge_styled_tokens_with_timestamps(tokens, word_timestamps)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc

    payload: dict[str, Any] = {
        "topic": topic_prompt.strip(),
        "raw_script": raw_script,
        "tokens": tokens,
        "audio": {
            "path": str(narration_path.resolve()),
            "voice": voice,
            "word_timestamps": word_timestamps,
        },
    }
    music = attach_random_music(out)
    if music is not None:
        payload["audio"]["music"] = music
    payload_path = out / "pipeline_payload.json"
    save_payload(
        {
            **payload,
            "render": {
                "output_dir": str(out.resolve()),
                "preview_path": None,
                "final_path": None,
            },
        },
        payload_path,
    )

    print("=" * 60)
    print("OUTPUT ARTIFACTS")
    print("=" * 60)
    print(f"Run folder: {out.resolve()}/")
    print(f"Narration:  {narration_path.resolve()}")
    print(f"Payload:    {payload_path.resolve()}")
    print()

    return payload


def run_slideshow_with_render(
    topic_prompt: str,
    *,
    output_dir: str | Path,
    caption_mode: str = "none",
    skip_images: bool = False,
    image_provider: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run slideshow generation then Remotion export → final.mp4 in output_dir."""
    from core.publish_runner import publish_video
    from core.remotion_render_stage import render_project_video
    from core.slideshow_pipeline import run_slideshow_pipeline

    out = Path(output_dir)
    run_slideshow_pipeline(
        topic_prompt,
        output_dir=out,
        caption_mode=caption_mode,
        skip_images=skip_images,
        image_provider=image_provider,
    )
    payload_path = out / "pipeline_payload.json"
    final = render_project_video(payload_path, out / "final.mp4")
    publish_video(final, payload_path=payload_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return payload, final


def run_mvp_with_render(
    topic_prompt: str,
    *,
    output_dir: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Run MVP generation then Remotion export → final.mp4 in output_dir."""
    from core.publish_runner import publish_video
    from core.remotion_render_stage import render_project_video

    out = Path(output_dir)
    run_mvp_pipeline(topic_prompt, output_dir=out)
    payload_path = out / "pipeline_payload.json"
    final = render_project_video(payload_path, out / "final.mp4")
    publish_video(final, payload_path=payload_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return payload, final


def main() -> None:
    """CLI entry point for the MVP orchestrator.

    Goal: Run the pipeline from the command line with topic and output-dir args.
    Params: None (reads sys.argv).
    Output: None; exits 0 on success or 1 after printing an error message.
    """
    _load_env()

    parser = argparse.ArgumentParser(description="Run the short-form video MVP orchestrator.")
    parser.add_argument(
        "topic",
        nargs="?",
        default="90% mọi người đang uống nước sai cách",
        help="Video topic for the script writer",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Exact run folder for all artifacts (default: OUTPUT_DIR/final or "
            "output/final; existing files are kept for TTS resume unless --fresh)"
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the run folder before starting (wipes scenes_draft.json and scene_tts cache)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip Remotion render; only write payload, images, and audio",
    )
    parser.add_argument(
        "--mode",
        choices=("mvp", "slideshow"),
        default="slideshow",
        help="Pipeline mode: slideshow (default, 3-scene slides) or mvp (word karaoke)",
    )
    parser.add_argument(
        "--caption-mode",
        choices=("none", "sentence", "word"),
        default=_default_caption_mode(),
        help="Caption overlay for slideshow mode (default: CAPTION_MODE env or none)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Slideshow mode: skip slide image generation (TTS + payload only)",
    )
    parser.add_argument(
        "--image-provider",
        choices=("pollinations", "chatgpt", "mock"),
        default=None,
        help="Slideshow image backend (default: IMAGE_PROVIDER env or pollinations)",
    )
    args = parser.parse_args()

    from core.run_output import (
        ensure_run_dir,
        prepare_default_run_dir,
        reset_run_dir,
        resolve_final_output_dir,
        resolve_generation_output_dir,
    )

    if args.output_dir:
        resolved = resolve_generation_output_dir(args.output_dir)
        out_dir = reset_run_dir(resolved) if args.fresh else ensure_run_dir(resolved)
    elif args.fresh:
        out_dir = reset_run_dir(resolve_final_output_dir())
    else:
        out_dir = prepare_default_run_dir()
    print(f"Run folder: {out_dir}")

    try:
        if args.mode == "slideshow":
            if args.no_render:
                from core.slideshow_pipeline import run_slideshow_pipeline

                run_slideshow_pipeline(
                    args.topic,
                    output_dir=out_dir,
                    caption_mode=args.caption_mode,
                    skip_images=args.skip_images,
                    image_provider=args.image_provider,
                )
            else:
                _, final = run_slideshow_with_render(
                    args.topic,
                    output_dir=out_dir,
                    caption_mode=args.caption_mode,
                    skip_images=args.skip_images,
                    image_provider=args.image_provider,
                )
                print(f"Final video: {final}")
        elif args.no_render:
            run_mvp_pipeline(args.topic, output_dir=out_dir)
        else:
            _, final = run_mvp_with_render(args.topic, output_dir=out_dir)
            print(f"Final video: {final}")
    except (PipelineError, ValueError, RuntimeError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
