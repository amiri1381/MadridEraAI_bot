import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👑 MADRID ERA CONTROL\n\n"
        "🎬 ویدیو را بفرست.\n"
        "🎵 بعد آهنگ را بفرست.\n"
        "📝 در آخر دستور ادیتت را بنویس."
    )


async def receive_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["video_received"] = True

    await update.message.reply_text(
        "🎬 ویدیو دریافت شد.\n\n"
        "حالا 🎵 آهنگت را بفرست."
    )


async def receive_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("video_received"):
        await update.message.reply_text("اول 🎬 ویدیو را بفرست.")
        return

    context.user_data["audio_received"] = True

    await update.message.reply_text(
        "🎵 آهنگ دریافت شد.\n\n"
        "حالا 📝 دستور ادیتت را بنویس."
    )


async def receive_instruction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("video_received"):
        await update.message.reply_text("اول 🎬 ویدیو را بفرست.")
        return

    if not context.user_data.get("audio_received"):
        await update.message.reply_text("اول 🎵 آهنگ را بفرست.")
        return

    context.user_data["instruction"] = update.message.text

    await update.message.reply_text(
        "✅ پروژه دریافت شد!\n\n"
        "🎬 ویدیو: آماده\n"
        "🎵 آهنگ: آماده\n"
        "📝 دستور: دریافت شد\n\n"
        "🤖 آماده مرحله بعد است."
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "🗑️ پروژه پاک شد.\n"
        "برای شروع دوباره /start را بزن."
    )


def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))

    app.add_handler(MessageHandler(filters.VIDEO, receive_video))
    app.add_handler(MessageHandler(filters.AUDIO, receive_audio))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, receive_instruction)
    )

    print("MADRID ERA CONTROL is running...")
    app.run_polling()


if __name__ == "__main__":
    main()