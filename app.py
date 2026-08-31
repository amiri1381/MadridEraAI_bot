import os
import re
import math
import shutil
import asyncio
import subprocess
import tempfile
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------

def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


FFMPEG = get_ffmpeg()


def run_cmd(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])

    return result


# ---------------------------------------------------------
# Beat detection
# ---------------------------------------------------------

def detect_beats(audio_file):
    """
    Extract audio to PCM and estimate beat positions.
    This is intentionally lightweight so it can run on Railway.
    """

    temp_wav = audio_file.parent / "analysis.wav"

    run_cmd([
        FFMPEG,
        "-y",
        "-i",
        str(audio_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        "-f",
        "wav",
        str(temp_wav),
    ])

    import wave

    with wave.open(str(temp_wav), "rb") as wav:
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    if len(audio) == 0:
        return []

    audio /= 32768.0

    # Short-time energy
    window = int(rate * 0.05)
    hop = int(rate * 0.025)

    energies = []

    for i in range(0, len(audio) - window, hop):
        chunk = audio[i:i + window]
        energies.append(float(np.sqrt(np.mean(chunk * chunk))))

    energies = np.array(energies)

    if len(energies) < 10:
        return []

    # Dynamic threshold
    baseline = np.median(energies)
    deviation = np.std(energies)

    threshold = baseline + deviation * 0.8

    candidates = []

    for i in range(1, len(energies) - 1):
        if (
            energies[i] > threshold
            and energies[i] >= energies[i - 1]
            and energies[i] >= energies[i + 1]
        ):
            t = i * hop / rate
            candidates.append(t)

    # Prevent too many cuts
    beats = []

    for t in candidates:
        if not beats or t - beats[-1] >= 0.35:
            beats.append(t)

    return beats


# ---------------------------------------------------------
# Media information
# ---------------------------------------------------------

def get_duration(file):
    result = run_cmd([
        FFMPEG,
        "-i",
        str(file),
    ])

    text = result.stderr

    match = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        text
    )

    if not match:
        return 0.0

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return hours * 3600 + minutes * 60 + seconds


# ---------------------------------------------------------
# Main video processor
# ---------------------------------------------------------

def create_edit(video, audio, instruction, output):
    duration = get_duration(video)

    if duration <= 0:
        raise RuntimeError("Could not read video duration.")

    beats = detect_beats(audio)

    # Limit processing complexity
    if len(beats) > 40:
        beats = beats[:40]

    # If beat detection fails, create safe cuts
    if len(beats) < 2:
        cuts = [
            0,
            duration * 0.25,
            duration * 0.50,
            duration * 0.75,
        ]
    else:
        cuts = [0.0]

        for beat in beats:
            if beat > 0.25 and beat < duration - 0.15:
                cuts.append(beat)

    # Remove duplicates and invalid positions
    clean = []

    for value in cuts:
        value = float(value)

        if value < duration and (
            not clean or value - clean[-1] >= 0.30
        ):
            clean.append(value)

    cuts = clean

    # We don't create hundreds of FFmpeg filters.
    # Instead we create a rhythm-based selection of segments.
    segments = []

    for i in range(len(cuts)):
        start = cuts[i]

        if i + 1 < len(cuts):
            end = cuts[i + 1]
        else:
            end = duration

        length = end - start

        if length >= 0.35:
            segments.append((start, length))

    # Maximum number of segments for Railway stability
    segments = segments[:30]

    if not segments:
        segments = [(0, duration)]

    # Create individual clips
    clip_files = []

    for index, (start, length) in enumerate(segments):
        clip = video.parent / f"clip_{index}.mp4"

        # Different small visual treatments based on segment index.
        # The instruction controls the overall style.
        if "سیاه" in instruction or "dark" in instruction.lower():
            vf = (
                "eq=brightness=-0.03:"
                "contrast=1.08:"
                "saturation=0.95,"
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920"
            )
        else:
            vf = (
                "eq=contrast=1.06:"
                "saturation=1.08:"
                "brightness=0.01,"
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920"
            )

        # Small alternating zoom effect
        if index % 2 == 0:
            vf += ",zoompan=z='min(zoom+0.0015,1.08)':"
            vf += "d=1:s=1080x1920:fps=30"
        else:
            vf += ",fps=30"

        run_cmd([
            FFMPEG,
            "-y",
            "-ss",
            str(start),
            "-t",
            str(length),
            "-i",
            str(video),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ])

        clip_files.append(clip)

    # Create concat list
    concat_file = video.parent / "concat.txt"

    with open(concat_file, "w", encoding="utf-8") as f:
        for clip in clip_files:
            safe_path = str(clip).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    joined = video.parent / "joined.mp4"

    run_cmd([
        FFMPEG,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(joined),
    ])

    # Final output:
    # video + user-selected music
    run_cmd([
        FFMPEG,
        "-y",
        "-i",
        str(joined),
        "-i",
        str(audio),
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
        "-movflags",
        "+faststart",
        str(output),
    ])

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Output video was not created.")

    return output


# ---------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()

    await update.message.reply_text(
        "👑 MADRID ERA AI\n\n"
        "🎬 ویدیو را بفرست.\n"
        "🎵 بعد آهنگ را بفرست.\n"
        "📝 در آخر دستور ادیتت را بنویس.\n\n"
        "مثال:\n"
        "«ادیت فوتبالی هیجانی، کات روی ضرب، "
        "زوم روی لحظه‌های مهم و خروجی مناسب TikTok»"
    )


async def receive_video(update: Update, context: ContextTypes.DEFAULT_TYPE):

    file = await update.message.video.get_file()

    workdir = Path(tempfile.mkdtemp(prefix="madrid_"))
    video_path = workdir / "input.mp4"

    await file.download_to_drive(str(video_path))

    context.user_data["workdir"] = str(workdir)
    context.user_data["video"] = str(video_path)
    context.user_data["video_received"] = True

    await update.message.reply_text(
        "🎬 ویدیو دریافت شد.\n\n"
        "حالا 🎵 آهنگت را بفرست."
    )


async def receive_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not context.user_data.get("video_received"):
        await update.message.reply_text(
            "اول 🎬 ویدیو را بفرست."
        )
        return

    workdir = Path(context.user_data["workdir"])

    if update.message.audio:
        telegram_file = await update.message.audio.get_file()
    elif update.message.document:
        telegram_file = await update.message.document.get_file()
    else:
        await update.message.reply_text(
            "لطفاً فایل آهنگ را بفرست."
        )
        return

    audio_path = workdir / "music"

    await telegram_file.download_to_drive(str(audio_path))

    context.user_data["audio"] = str(audio_path)
    context.user_data["audio_received"] = True

    await update.message.reply_text(
        "🎵 آهنگ دریافت شد.\n\n"
        "حالا 📝 دستور ادیتت را بنویس."
    )


async def receive_instruction(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not context.user_data.get("video_received"):
        await update.message.reply_text(
            "اول 🎬 ویدیو را بفرست."
        )
        return

    if not context.user_data.get("audio_received"):
        await update.message.reply_text(
            "اول 🎵 آهنگ را بفرست."
        )
        return

    instruction = update.message.text.strip()

    if not instruction:
        await update.message.reply_text(
            "📝 دستور ادیت نمی‌تواند خالی باشد."
        )
        return

    context.user_data["instruction"] = instruction

    await update.message.reply_text(
        "✅ پروژه کامل شد.\n\n"
        "🎬 ویدیو: آماده\n"
        "🎵 آهنگ: آماده\n"
        "📝 دستور: دریافت شد\n\n"
        "🤖 Auto Edit شروع شد...\n"
        "🥁 در حال تحلیل ریتم\n"
        "✂️ در حال ساخت کات‌ها\n"
        "🎨 در حال اعمال افکت‌ها\n"
        "🎥 در حال ساخت خروجی"
    )

    workdir = Path(context.user_data["workdir"])
    video = Path(context.user_data["video"])
    audio = Path(context.user_data["audio"])
    output = workdir / "MADRID_ERA_FINAL.mp4"

    try:
        # Run CPU-heavy work outside Telegram event loop
        await asyncio.to_thread(
            create_edit,
            video,
            audio,
            instruction,
            output,
        )

        await update.message.reply_text(
            "✅ ادیت با موفقیت تمام شد!\n\n"
            "🎥 خروجی آماده است. در حال ارسال..."
        )

        with open(output, "rb") as video_file:
            await update.message.reply_video(
                video=video_file,
                caption=(
                    "👑 MADRID ERA AI\n"
                    "🤖 Auto Edit completed"
                ),
                supports_streaming=True,
            )

        await update.message.reply_text(
            "🔥 ادیت آماده شد."
        )

    except Exception as e:

        error_text = str(e)

        if len(error_text) > 1200:
            error_text = error_text[-1200:]

        await update.message.reply_text(
            "❌ پردازش متوقف شد.\n\n"
            "خطای فنی:\n"
            f"{error_text}"
        )

    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

        context.user_data.clear()


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):

    workdir = context.user_data.get("workdir")

    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    context.user_data.clear()

    await update.message.reply_text(
        "🗑️ پروژه پاک شد.\n"
        "برای شروع دوباره /start را بزن."
    )


# ---------------------------------------------------------
# TikTok / sample link detection
# ---------------------------------------------------------

async def receive_link(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = update.message.text.strip()

    if "tiktok.com" not in text:
        await receive_instruction(update, context)
        return

    await update.message.reply_text(
        "🔗 لینک TikTok دریافت شد.\n\n"
        "این لینک به‌عنوان نمونه سبک ادیت ثبت شد.\n"
        "برای اجرای پروژه، ویدیو و آهنگ را هم بفرست."
    )

    context.user_data["reference_link"] = text


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))

    app.add_handler(
        MessageHandler(
            filters.VIDEO,
            receive_video
        )
    )

    app.add_handler(
        MessageHandler(
            filters.AUDIO | filters.Document.AUDIO,
            receive_audio
        )
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            receive_link
        )
    )

    print("👑 MADRID ERA AI IS RUNNING...")

    app.run_polling()


if __name__ == "__main__":
    main()
