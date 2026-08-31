import os
import re
import math
import shutil
import asyncio
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import cv2
import yt_dlp

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")


# =========================================================
# FFmpeg
# =========================================================

def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


FFMPEG = get_ffmpeg()


def run_cmd(command, binary=False):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )

    if result.returncode != 0:
        error = result.stderr
        if not isinstance(error, str):
            error = error.decode("utf-8", errors="ignore")
        raise RuntimeError(error[-5000:])

    return result


# =========================================================
# Helpers
# =========================================================

def clean_name(name):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def find_tiktok_url(text):
    if not text:
        return None

    match = re.search(
        r"https?://(?:www\.)?(?:tiktok\.com|vm\.tiktok\.com)/[^\s]+",
        text,
        re.IGNORECASE,
    )

    return match.group(0) if match else None


def video_duration(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError("ویدیو قابل خواندن نیست.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if not fps or fps <= 0 or frames <= 0:
        raise RuntimeError("مدت ویدیو قابل تشخیص نیست.")

    return frames / fps


def normalize_duration(value):
    return max(0.1, float(value))


# =========================================================
# TikTok Downloader
# =========================================================

def download_tiktok(url, output_dir):
    output_template = str(Path(output_dir) / "tiktok_source.%(ext)s")

    options = {
        "outtmpl": output_template,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise RuntimeError(
            "دانلود لینک TikTok انجام نشد.\n"
            "ممکن است لینک خصوصی، منقضی یا توسط TikTok محدود شده باشد."
        ) from exc

    files = list(Path(output_dir).glob("tiktok_source.*"))

    if not files:
        raise RuntimeError("فایل ویدیوی TikTok پیدا نشد.")

    return files[0]


# =========================================================
# Audio / Beat Detection
# =========================================================

def extract_wav(audio_path, wav_path):
    run_cmd([
        FFMPEG,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "8000",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ])


def detect_beats(audio_path):
    with tempfile.TemporaryDirectory() as temp:
        wav_path = Path(temp) / "analysis.wav"

        extract_wav(audio_path, wav_path)

        with wave.open(str(wav_path), "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    if len(audio) < sample_rate:
        return []

    audio /= 32768.0

    window = 1024
    hop = 512

    if len(audio) < window:
        return []

    rms = []

    for start in range(0, len(audio) - window, hop):
        chunk = audio[start:start + window]
        rms.append(float(np.sqrt(np.mean(chunk * chunk))))

    rms = np.asarray(rms)

    if len(rms) < 5:
        return []

    # انرژی تغییر یافته برای پیدا کردن ضرب‌ها
    diff = np.diff(rms, prepend=rms[0])
    diff[diff < 0] = 0

    threshold = max(
        float(np.percentile(diff, 72)),
        float(np.mean(diff) + np.std(diff) * 0.35),
    )

    candidates = np.where(diff >= threshold)[0]

    beats = []

    min_gap = int(0.28 * sample_rate / hop)

    last_index = -min_gap

    for index in candidates:
        if index - last_index < min_gap:
            continue

        left = max(0, index - 2)
        right = min(len(diff), index + 3)

        if diff[index] >= np.max(diff[left:right]):
            time = index * hop / sample_rate
            beats.append(time)
            last_index = index

    # اگر ضرب قابل تشخیص نبود، برش‌های منظم می‌سازیم
    if len(beats) < 2:
        duration = len(audio) / sample_rate
        step = 0.8
        beats = list(np.arange(0, duration, step))

    return beats


# =========================================================
# Instruction Parser
# =========================================================

def instruction_flags(text):
    text = (text or "").lower()

    return {
        "gray": any(
            word in text
            for word in [
                "blackwhite",
                "black and white",
                "grayscale",
                "سیاه سفید",
                "سیاه‌سفید",
            ]
        ),

        "mirror": any(
            word in text
            for word in [
                "mirror",
                "flip",
                "آینه",
                "برعکس",
            ]
        ),

        "slow": any(
            word in text
            for word in [
                "slow",
                "slowmo",
                "slow motion",
                "آهسته",
                "اسلوموشن",
            ]
        ),

        "fast": any(
            word in text
            for word in [
                "fast",
                "speed",
                "سریع",
                "تند",
            ]
        ),
    }


# =========================================================
# Create Beat-Synced Video
# =========================================================

def create_segments(video_path, beats, output_dir):
    duration = video_duration(video_path)

    cuts = [0.0]

    for beat in beats:
        beat = float(beat)

        if 0.25 < beat < duration - 0.25:
            if beat - cuts[-1] >= 0.35:
                cuts.append(beat)

    if duration - cuts[-1] > 0.25:
        cuts.append(duration)

    segments = []

    for index in range(len(cuts) - 1):
        start = cuts[index]
        end = cuts[index + 1]
        length = end - start

        if length < 0.25:
            continue

        segment = Path(output_dir) / f"segment_{index:04d}.mp4"

        run_cmd([
            FFMPEG,
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(length),
            "-an",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(segment),
        ])

        segments.append(segment)

    if not segments:
        raise RuntimeError("هیچ بخش قابل استفاده‌ای از ویدیو ساخته نشد.")

    return segments


def concat_segments(segments, output_path):
    list_file = output_path.parent / "concat.txt"

    with open(list_file, "w", encoding="utf-8") as file:
        for segment in segments:
            safe_path = str(segment).replace("'", "'\\''")
            file.write(f"file '{safe_path}'\n")

    run_cmd([
        FFMPEG,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ])


# =========================================================
# Apply Visual Effects
# =========================================================

def apply_effects(video_path, output_path, instruction):
    flags = instruction_flags(instruction)

    filters_list = []

    if flags["gray"]:
        filters_list.append("hue=s=0")

    if flags["mirror"]:
        filters_list.append("hflip")

    if flags["slow"]:
        filters_list.append("setpts=1.25*PTS")

    if flags["fast"]:
        filters_list.append("setpts=0.8*PTS")

    filters_list.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    video_filter = ",".join(filters_list)

    video_speed = 1.0

    if flags["slow"]:
        video_speed = 0.8

    elif flags["fast"]:
        video_speed = 1.25

    run_cmd([
        FFMPEG,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ])

    return video_speed


# =========================================================
# Add Music
# =========================================================

def add_music(video_path, audio_path, output_path, video_speed=1.0):
    audio_filter = "apad"

    run_cmd([
        FFMPEG,
        "-y",
        "-i",
        str(video_path),
        "-stream_loop",
        "-1",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-af",
        audio_filter,
        "-movflags",
        "+faststart",
        str(output_path),
    ])


# =========================================================
# Complete Processing
# =========================================================

def process_project(video_path, audio_path, instruction, work_dir):
    work_dir = Path(work_dir)

    beats = detect_beats(audio_path)

    segments_dir = work_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    segments = create_segments(
        video_path,
        beats,
        segments_dir,
    )

    beat_video = work_dir / "beat_video.mp4"

    concat_segments(
        segments,
        beat_video,
    )

    effected_video = work_dir / "effected.mp4"

    video_speed = apply_effects(
        beat_video,
        effected_video,
        instruction,
    )

    output = work_dir / "MADRID_ERA_EDIT.mp4"

    add_music(
        effected_video,
        audio_path,
        output,
        video_speed,
    )

    if not output.exists() or output.stat().st_size < 10000:
        raise RuntimeError("فایل خروجی ساخته نشد.")

    return output


# =========================================================
# Telegram UI
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()

    await update.message.reply_text(
        "👑 MADRID ERA AI\n\n"
        "🎬 ویدیو را بفرست.\n"
        "یا 🔗 لینک TikTok را بفرست.\n\n"
        "بعد 🎵 آهنگ را بفرست.\n"
        "در آخر 📝 دستور ادیتت را بنویس.\n\n"
        "🤖 بعد از دستور، پردازش به‌صورت خودکار شروع می‌شود."
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()

    await update.message.reply_text(
        "🗑️ پروژه پاک شد.\n\n"
        "برای شروع دوباره /start را بزن."
    )


async def receive_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    work_dir = tempfile.mkdtemp(prefix="madrid_")

    try:
        video = update.message.video

        if not video:
            await update.message.reply_text(
                "❌ ویدیو دریافت نشد."
            )
            shutil.rmtree(work_dir, ignore_errors=True)
            return

        file = await context.bot.get_file(video.file_id)

        video_path = Path(work_dir) / "input_video.mp4"

        await file.download_to_drive(str(video_path))

        context.user_data["work_dir"] = work_dir
        context.user_data["video_path"] = str(video_path)
        context.user_data["video_received"] = True

        await update.message.reply_text(
            "✅ 🎬 ویدیو دریافت شد.\n\n"
            "حالا 🎵 آهنگ را بفرست."
        )

    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)

        await update.message.reply_text(
            f"❌ دریافت ویدیو ناموفق بود:\n{str(exc)[:1500]}"
        )


async def receive_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    if not context.user_data.get("video_received"):
        await update.message.reply_text(
            "اول 🎬 ویدیو را بفرست."
        )
        return

    work_dir = context.user_data.get("work_dir")

    if not work_dir:
        await update.message.reply_text(
            "❌ پروژه پیدا نشد. /start را بزن."
        )
        return

    try:
        audio = update.message.audio or update.message.voice

        if not audio:
            await update.message.reply_text(
                "❌ آهنگ دریافت نشد."
            )
            return

        file = await context.bot.get_file(audio.file_id)

        audio_path = Path(work_dir) / "music"

        if update.message.audio:
            audio_path = audio_path.with_suffix(".mp3")
        else:
            audio_path = audio_path.with_suffix(".ogg")

        await file.download_to_drive(str(audio_path))

        context.user_data["audio_path"] = str(audio_path)
        context.user_data["audio_received"] = True

        await update.message.reply_text(
            "✅ 🎵 آهنگ دریافت شد.\n\n"
            "حالا 📝 دستور ادیتت را بنویس.\n\n"
            "مثلاً:\n"
            "«کات‌ها با ضرب آهنگ هماهنگ باشد، "
            "حس سریع و فوتبالی داشته باشد.»"
        )

    except Exception as exc:
        await update.message.reply_text(
            f"❌ دریافت آهنگ ناموفق بود:\n{str(exc)[:1500]}"
        )


async def receive_instruction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    text = update.message.text or ""

    # TikTok URL
    tiktok_url = find_tiktok_url(text)

    if tiktok_url and not context.user_data.get("video_received"):

        work_dir = tempfile.mkdtemp(prefix="madrid_tiktok_")

        await update.message.reply_text(
            "🔗 لینک TikTok دریافت شد.\n\n"
            "⬇️ در حال دریافت ویدیو..."
        )

        try:
            video_path = await asyncio.to_thread(
                download_tiktok,
                tiktok_url,
                work_dir,
            )

            context.user_data["work_dir"] = work_dir
            context.user_data["video_path"] = str(video_path)
            context.user_data["video_received"] = True
            context.user_data["tiktok_url"] = tiktok_url

            await update.message.reply_text(
                "✅ ویدیوی TikTok آماده شد.\n\n"
                "حالا 🎵 آهنگ را بفرست."
            )

        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)

            await update.message.reply_text(
                f"❌ دریافت TikTok انجام نشد:\n{str(exc)[:1500]}"
            )

        return

    if not context.user_data.get("video_received"):
        await update.message.reply_text(
            "اول 🎬 ویدیو یا 🔗 لینک TikTok را بفرست."
        )
        return

    if not context.user_data.get("audio_received"):
        await update.message.reply_text(
            "اول 🎵 آهنگ را بفرست."
        )
        return

    context.user_data["instruction"] = text

    await update.message.reply_text(
        "✅ پروژه کامل دریافت شد!\n\n"
        "🎬 ویدیو: آماده\n"
        "🎵 آهنگ: آماده\n"
        "📝 دستور: دریافت شد\n\n"
        "🤖 حالت Auto Edit فعال شد.\n"
        "🎵 در حال تحلیل ضرب آهنگ...\n"
        "🎬 در حال ساخت کات‌ها...\n"
        "⚙️ در حال پردازش ویدیو...\n\n"
        "⏳ لطفاً صبر کن..."
    )

    work_dir = context.user_data.get("work_dir")

    video_path = context.user_data.get("video_path")
    audio_path = context.user_data.get("audio_path")

    if not work_dir or not video_path or not audio_path:
        await update.message.reply_text(
            "❌ اطلاعات پروژه ناقص است. /start را بزن."
        )
        return

    try:
        output = await asyncio.to_thread(
            process_project,
            Path(video_path),
            Path(audio_path),
            text,
            Path(work_dir),
        )

        await update.message.reply_text(
            "🎉 پردازش با موفقیت تمام شد!\n\n"
            "📤 خروجی آماده است. در حال ارسال..."
        )

        with open(output, "rb") as video_file:
            await update.message.reply_video(
                video=video_file,
                caption=(
                    "👑 MADRID ERA AI\n\n"
                    "✅ Auto Edit انجام شد.\n"
                    "🎵 کات‌ها بر اساس ریتم آهنگ ساخته شدند."
                ),
                supports_streaming=True,
            )

        tiktok_url = context.user_data.get("tiktok_url")

        if tiktok_url:
            await update.message.reply_text(
                "🔗 لینک TikTok ورودی:\n"
                f"{tiktok_url}"
            )

        await update.message.reply_text(
            "✅ پروژه تمام شد.\n\n"
            "برای ادیت جدید /start را بزن."
        )

    except Exception as exc:
        await update.message.reply_text(
            "❌ پردازش انجام نشد.\n\n"
            "خطا:\n"
            f"{str(exc)[:3000]}\n\n"
            "اگر دوباره تکرار شد، Logs Railway را بفرست."
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        context.user_data.clear()


# =========================================================
# Main
# =========================================================

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))

    app.add_handler(
        MessageHandler(
            filters.VIDEO,
            receive_video,
        )
    )

    app.add_handler(
        MessageHandler(
            filters.AUDIO | filters.VOICE,
            receive_audio,
        )
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            receive_instruction,
        )
    )

    print("MADRID ERA AI is running...")

    app.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
