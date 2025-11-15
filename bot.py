import os
import json
import logging
from functools import lru_cache
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes,
)
from groq import Groq
from rapidfuzz import fuzz

# === ЛОГИРОВАНИЕ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# === ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===
SHEET_ID = "1HBdZBWjlplVdZ4a7A5hdXxPyb2vyQ68ntIJ-oPfRwhA"
RANGE_NAME = "Support!A:B"

# === ПРОВЕРКА КАЖДОЙ ПЕРЕМЕННОЙ ОТДЕЛЬНО ===
errors = []

# 1. TELEGRAM_TOKEN
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    errors.append("TELEGRAM_TOKEN не задан")
elif not TELEGRAM_TOKEN.strip():
    errors.append("TELEGRAM_TOKEN пустой")
else:
    logger.info("TELEGRAM_TOKEN загружен")

# 2. GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    errors.append("GROQ_API_KEY не задан")
elif not GROQ_API_KEY.strip():
    errors.append("GROQ_API_KEY пустой")
else:
    logger.info("GROQ_API_KEY загружен")

# 3. GOOGLE_CREDENTIALS
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
if not GOOGLE_CREDENTIALS:
    errors.append("GOOGLE_CREDENTIALS не задан")
elif not GOOGLE_CREDENTIALS.strip():
    errors.append("GOOGLE_CREDENTIALS пустой")
else:
    try:
        creds_info = json.loads(GOOGLE_CREDENTIALS)
        required_keys = ["type", "project_id", "private_key", "client_email"]
        missing = [k for k in required_keys if k not in creds_info]
        if missing:
            errors.append(f"GOOGLE_CREDENTIALS: отсутствуют ключи: {', '.join(missing)}")
        else:
            logger.info("GOOGLE_CREDENTIALS — валидный JSON")
    except json.JSONDecodeError as e:
        errors.append(f"GOOGLE_CREDENTIALS — невалидный JSON: {e}")

# 4. ADMIN_ID
ADMIN_ID_STR = os.getenv("ADMIN_ID")
if not ADMIN_ID_STR:
    errors.append("ADMIN_ID не задан")
elif not ADMIN_ID_STR.strip():
    errors.append("ADMIN_ID пустой")
else:
    try:
        ADMIN_IDS = [int(uid.strip()) for uid in ADMIN_ID_STR.split(",") if uid.strip()]
        if not ADMIN_IDS:
            raise ValueError
        logger.info(f"Админы загружены: {ADMIN_IDS}")
    except ValueError:
        errors.append("ADMIN_ID должен быть числом или списком чисел через запятую")

# === ВЫВОД ОШИБОК ===
if errors:
    logger.error("ОШИБКИ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ:")
    for err in errors:
        logger.error(f"  → {err}")
    logger.error("Исправь переменные в Render → Environment и сделай Manual Deploy!")
    exit(1)
else:
    logger.info("Все переменные окружения успешно загружены!")

# === GOOGLE SHEETS ===
try:
    creds = Credentials.from_service_account_info(
        json.loads(GOOGLE_CREDENTIALS),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    logger.info("Google Sheets подключён.")
except Exception as e:
    logger.error(f"Ошибка Google Auth: {e}")
    exit(1)

# === GROQ ===
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq API подключён.")
except Exception as e:
    logger.error(f"Ошибка Groq: {e}")
    exit(1)

# === КЕШИРОВАННАЯ БАЗА ЗНАНИЙ ===
@lru_cache(maxsize=1)
def get_knowledge_base():
    try:
        logger.info(f"Читаем Google Sheets: {RANGE_NAME}")
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = result.get("values", [])[1:]
        kb_entries = []
        for r in rows:
            problem = r[0].strip() if len(r) > 0 else ""
            solution = r[1].strip() if len(r) > 1 else "Нет решения"
            kb_entries.append(f"Проблема: {problem}\nРешение: {solution}")
        kb = "\n\n".join(kb_entries)
        logger.info(f"База знаний загружена: {len(rows)} записей.")
        return kb or "База знаний пуста."
    except Exception as e:
        logger.error(f"Ошибка чтения Sheets: {e}")
        return "Ошибка доступа к базе знаний."

# === СИСТЕМНЫЙ ПРОМПТ ===
SYSTEM_PROMPT = """
Ты — бот техподдержки. Используй ТОЛЬКО эту базу знаний:
{kb}
Если не нашёл — скажи: "Не нашёл точного решения. Опиши подробнее или обратись к модератору."
Отвечай кратко, по-русски, шаг за шагом.
"""

# === КОМАНДЫ АДМИНА ===
async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    context.bot_data["paused"] = True
    await update.message.reply_text("Бот приостановлен. Используй /resume")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    context.bot_data["paused"] = False
    await update.message.reply_text("Бот возобновлён!")

async def status_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    status = "приостановлен" if context.bot_data.get("paused", False) else "работает"
    await update.message.reply_text(f"Бот {status}")

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Доступ запрещён.")
        return
    get_knowledge_base.cache_clear()
    logger.info("Кеш сброшен по команде /reload")
    await update.message.reply_text("База знаний обновлена!")

# === ОБРАБОТКА СООБЩЕНИЙ ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    if context.bot_data.get("paused", False):
        if message.text == "/status" and update.effective_user.id in ADMIN_IDS:
            await message.reply_text("Бот приостановлен.")
        return

    text = (message.text or message.caption or "").strip()
    if not text or text.startswith("/"):
        return

    logger.info(f"Сообщение: {text[:50]}...")

    kb = get_knowledge_base()
    kb_blocks = [b.strip() for b in kb.split("\n\n") if b.strip()]
    user_query = text.lower().strip()
    best_score = 0
    best_solution = None

    for block in kb_blocks:
        lines = [line.strip() for line in block.split("\n")]
        if len(lines) < 2 or not lines[0].lower().startswith("проблема:"):
            continue
        problem = lines[0][10:].strip()
        score = fuzz.ratio(user_query, problem.lower())
        if score > 85 and score > best_score:
            best_score = score
            solution_lines = [line.replace("Решение: ", "", 1).strip() for line in lines[1:] if line.strip()]
            best_solution = "\n".join(solution_lines)

    if best_solution:
        await message.reply_text(best_solution)
        logger.info(f"Ответ из базы (схожесть: {best_score}%)")
        return

    full_prompt = SYSTEM_PROMPT.format(kb=kb) + f"\n\nЗапрос: {text}"
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": full_prompt}],
            max_tokens=250,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        await message.reply_text(reply)
        logger.info("Ответ от Groq")
    except Exception as e:
        logger.error(f"Ошибка Groq: {e}")
        await message.reply_text("Извини, временная ошибка.")

# === АВТООБНОВЛЕНИЕ ===
def schedule_auto_reload(app):
    def clear_cache(ctx):
        get_knowledge_base.cache_clear()
        logger.info("Автообновление: кеш сброшен")
    app.job_queue.run_repeating(clear_cache, interval=300, first=10)

# === ЗАПУСК ===
if __name__ == "__main__":
    logger.info("Запуск бота...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    if os.getenv("BOT_PAUSED", "false").lower() == "true":
        app.bot_data["paused"] = True
        logger.info("Бот в режиме паузы (BOT_PAUSED=true)")

    app.add_handler(MessageHandler(
        (filters.TEXT & ~filters.COMMAND) |
        (filters.CAPTION & ~filters.COMMAND),
        handle_message
    ))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_bot))

    schedule_auto_reload(app)
    app.run_polling(drop_pending_updates=True)