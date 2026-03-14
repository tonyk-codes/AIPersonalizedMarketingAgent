"""Media helpers for video output and image preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

import config


def sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.strip().lower())
    return "-".join(part for part in safe.split("-") if part)[:80] or "asset"


def save_video_bytes(video_bytes: bytes, prefix: str = "nike-ad") -> str:
    config.ensure_artifact_dirs()
    digest = hashlib.sha1(video_bytes).hexdigest()[:12]
    output_path = config.VIDEOS_DIR / f"{sanitize_filename(prefix)}-{digest}.mp4"
    output_path.write_bytes(video_bytes)
    return str(output_path)


def _resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _load_title_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf",
        "Arial Bold.ttf",
        "segoeuib.ttf",
        "HelveticaNeue-Bold.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _build_end_card_frames(final_slogan_text: str, width: int, height: int, fps: int) -> list[np.ndarray]:
    frame_count = max(1, int(round(config.FINAL_SLOGAN_FRAME_SECONDS * fps)))
    title_font = _load_title_font(42)
    label_font = _load_title_font(18)
    frames: list[np.ndarray] = []

    for idx in range(frame_count):
        progress = (idx + 1) / frame_count
        canvas = Image.new("RGB", (width, height), color=(8, 8, 8))
        overlay = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        accent_top = int(height * 0.18)
        accent_bottom = int(height * 0.82)
        draw.rectangle((width - 28, accent_top, width - 18, accent_bottom), fill=(255, 255, 255, 210))
        draw.rectangle((0, height - 72, width, height), fill=(18, 18, 18, 235))

        text_alpha = int(255 * min(1.0, progress * 1.35))
        text = final_slogan_text.strip() or "Move in your own way"
        bbox = draw.multiline_textbbox((0, 0), text, font=title_font, spacing=8)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = max(32, (width - text_width) // 2)
        text_y = max(48, (height - text_height) // 2 - 18)
        draw.multiline_text(
            (text_x, text_y),
            text,
            font=title_font,
            fill=(255, 255, 255, text_alpha),
            spacing=8,
            align="center",
        )
        draw.text((36, height - 52), "PERSONALIZED NIKE-STYLE AD", font=label_font, fill=(196, 196, 196, text_alpha))
        composed = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        frames.append(np.asarray(composed))

    return frames


def compose_final_ad_video(source_video_path: str, output_stem: str, final_slogan_text: str) -> str:
    config.ensure_artifact_dirs()
    width = config.DEFAULT_VIDEO_WIDTH
    height = config.DEFAULT_VIDEO_HEIGHT
    default_fps = config.DEFAULT_VIDEO_FPS
    output_path = config.VIDEOS_DIR / f"{sanitize_filename(output_stem)}.mp4"

    with VideoFileClip(source_video_path) as clip:
        fps = int(round(getattr(clip, "fps", 0) or default_fps))
        frames = [
            np.asarray(_resize_and_crop(Image.fromarray(frame).convert("RGB"), width, height))
            for frame in clip.iter_frames(fps=fps, dtype="uint8")
        ]

    frames.extend(_build_end_card_frames(final_slogan_text=final_slogan_text, width=width, height=height, fps=fps))

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio=False,
        preset="ultrafast",
        logger=None,
    )
    clip.close()
    return str(output_path)


def _placeholder_image(path: Path, title: str) -> None:
    canvas = Image.new("RGB", (1280, 720), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 580, 1280, 720), fill=(238, 29, 35))
    draw.text((60, 80), "SMART NIKE SHOE AD STUDIO", fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((60, 130), title[:70], fill=(230, 230, 230), font=ImageFont.load_default())
    canvas.save(path)


def ensure_local_image(image_path_or_url: str, name_hint: str = "product") -> str:
    """Download remote images once, otherwise return local path unchanged."""
    config.ensure_artifact_dirs()
    if not image_path_or_url:
        target = config.IMAGES_DIR / f"{sanitize_filename(name_hint)}-placeholder.png"
        _placeholder_image(target, name_hint)
        return str(target)

    parsed = urlparse(image_path_or_url)
    if parsed.scheme in ("http", "https"):
        stem = sanitize_filename(name_hint)
        suffix = Path(parsed.path).suffix or ".jpg"
        target = config.IMAGES_DIR / f"{stem}{suffix}"
        if not target.exists():
            try:
                urlretrieve(image_path_or_url, target)
            except Exception:
                _placeholder_image(target, name_hint)
        return str(target)

    path = Path(image_path_or_url)
    if path.exists():
        return str(path)

    target = config.IMAGES_DIR / f"{sanitize_filename(name_hint)}-placeholder.png"
    if not target.exists():
        _placeholder_image(target, name_hint)
    return str(target)


def create_fallback_banner_video(
    image_path_or_url: str,
    slogan: str,
    headline: str,
    output_stem: str,
    final_slogan_text: str = "",
    duration_seconds: int | None = None,
) -> str:
    """Create a short Ken-Burns-style MP4 when real video generation is unavailable."""
    config.ensure_artifact_dirs()
    image_path = ensure_local_image(image_path_or_url, name_hint=output_stem)
    width = config.DEFAULT_VIDEO_WIDTH
    height = config.DEFAULT_VIDEO_HEIGHT
    fps = config.DEFAULT_VIDEO_FPS
    duration = duration_seconds or config.DEFAULT_VIDEO_DURATION_SECONDS
    total_frames = max(1, duration * fps)

    base = Image.open(image_path).convert("RGB")
    frames: list[np.ndarray] = []

    for idx in range(total_frames):
        t = idx / max(1, total_frames - 1)
        scale = 1.0 + 0.08 * t
        scaled_w = int(base.width * scale)
        scaled_h = int(base.height * scale)
        scaled = base.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        left = max(0, (scaled_w - width) // 2)
        top = max(0, (scaled_h - height) // 2)
        frame = scaled.crop((left, top, left + width, top + height))

        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, height - 130, width, height), fill=(0, 0, 0, 150))

        slogan_text = slogan[:70] if slogan else "Move with confidence"
        headline_text = headline[:100] if headline else "Personalized Nike Shoe Story"
        draw.text((30, height - 110), slogan_text, fill=(255, 255, 255, 255), font=ImageFont.load_default())
        draw.text((30, height - 80), headline_text, fill=(225, 225, 225, 255), font=ImageFont.load_default())

        composed = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        frames.append(np.asarray(composed))

    end_text = final_slogan_text or slogan
    frames.extend(_build_end_card_frames(final_slogan_text=end_text, width=width, height=height, fps=fps))

    clip = ImageSequenceClip(frames, fps=fps)
    output_path = config.VIDEOS_DIR / f"{sanitize_filename(output_stem)}.mp4"
    clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio=False,
        preset="ultrafast",
        logger=None,
    )
    clip.close()
    return str(output_path)
