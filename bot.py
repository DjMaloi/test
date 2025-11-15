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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
SHEET_ID = "1HBdZBWjlplVdZ4a7A5hdXxPyb2vyQ68ntIJ-oPfRwhA"
RANGE_NAME = "Support!A:B"  # ← Проверьте точное имя листа!

# Проверка переменных
if not all([TELEGRAM_TOKEN, GROQ_API_KEY, GOOGLE_CREDENTIALS]):
    logger.error("ОШИБКА: Не заданы переменные в Render!")
    exit(1)
logger.info("Переменные загружены.")

# === GOOGLE SHEETS ===
try:
    creds_info = json.loads(GOOGLE_CREDENTIALS)
    creds = Credentials.from_service_account_info(
        creds_info,
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

# === КЕШИРОВАННАЯ БАЗА ЗНАНИЙ (с \n) ===
@lru_cache(maxsize=1)
def get_knowledge_base():
    try:
        logger.info(f"Читаем Google Sheets: {RANGE_NAME}")
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = result.get("values", [])[1:]  # Пропускаем заголовок
        kb_entries = []
        for r in rows:
            problem = r[0].strip() if len(r) > 0 else ""
            solution = r[1].strip() if len(r) > 1 else "Нет решения"
            kb_entries.append(f"Проблема: {problem}\nРешение: {solution}")
        kb = "\n\n".join(kb_entries)  # Двойной \n — разделитель записей
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

# === КОМАНДА /reload ===
async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    get_knowledge_base.cache_clear()
    logger.info("Кеш сброшен по команде /reload")
    await update.message.reply_text("База знаний обновлена!")

# === ОБРАБОТКА СООБЩЕНИЙ ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text or text.startswith("/"):
        return

    logger.info(f"Сообщение: {text[:50]}...")

    kb = get_knowledge_base()
    kb_blocks = [b.strip() for b in kb.split("\n\n") if b.strip()]

    user_query = text.lower().strip()
    best_score = 0
    best_solution = None

    # Поиск по базе
    for block in kb_blocks:
        lines = [line.strip() for line in block.split("\n")]
        if len(lines) < 2:
            continue
        if not lines[0].lower().startswith("проблема:"):
            continue

        problem = lines[0][10:].strip()
        score = fuzz.ratio(user_query, problem.lower())
        if score > 85 and score > best_score:
            best_score = score
            solution_lines = []
            for line in lines[1:]:
                cleaned = line.replace("Решение: ", "", 1).strip()
                if cleaned:
                    solution_lines.append(cleaned)
            best_solution = "\n".join(solution_lines)

    # Ответ из базы
    if best_solution:
        await update.message.reply_text(best_solution)
        logger.info(f"Ответ из базы (схожесть: {best_score}%)")
        return

    # Groq
    full_prompt = SYSTEM_PROMPT.format(kb=kb) + f"\n\nЗапрос: {text}"
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": full_prompt}],
            max_tokens=250,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        await update.message.reply_text(reply)
        logger.info("Ответ от Groq")
    except Exception as e:
        logger.error(f"Ошибка Groq: {e}")
        await update.message.reply_text("Извини, временная ошибка.")

# === АВТООБНОВЛЕНИЕ КЕША КАЖДЫЕ 5 МИНУТ ===
def schedule_auto_reload(app):
    def clear_cache(ctx):
        get_knowledge_base.cache_clear()
        logger.info("Автообновление: кеш базы знаний сброшен")

    app.job_queue.run_repeating(clear_cache, interval=300, first=10)

# === ЗАПУСК ===
if __name__ == "__main__":
    logger.info("Запуск бота...")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Хендлеры
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("reload", reload_kb))

    # Автообновление
    schedule_auto_reload(app)

    app.run_polling(drop_pending_updates=True)