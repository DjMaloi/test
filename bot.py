import os
import json
import logging
from functools import lru_cache
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from groq import Groq
from rapidfuzz import fuzz

# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
SHEET_ID = "1HBdZBWjlplVdZ4a7A5hdXxPyb2vyQ68ntIJ-oPfRwhA"
RANGE_NAME = "Sheet1!A:B"

# Проверка переменных
if not all([TELEGRAM_TOKEN, GROQ_API_KEY, GOOGLE_CREDENTIALS]):
    logger.error("ОШИБКА: Не заданы переменные в Render!")
    exit(1)
logger.info("Переменные загружены.")

# Google Sheets
try:
    creds_info = json.loads(GOOGLE_CREDENTIALS)
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    logger.info("Google Sheets подключён через env!")
except Exception as e:
    logger.error(f"Ошибка Google Auth: {e}")
    exit(1)

# Groq
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq API подключён.")
except Exception as e:
    logger.error(f"Ошибка Groq: {e}")
    exit(1)

# Кешированная база знаний
@lru_cache(maxsize=1)
def get_knowledge_base():
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = result.get('values', [])[1:]  # Пропускаем заголовок
        kb = "\n".join([
            f"Проблема: {r[0]}\nРешение: {r[1] if len(r) > 1 else 'Нет решения'}"
            for r in rows if r
        ])
        logger.info(f"База знаний загружена: {len(rows)} записей.")
        return kb or "База знаний пуста."
    except Exception as e:
        logger.error(f"Ошибка чтения Sheets: {e}")
        return "Ошибка доступа к базе знаний."

# Системный промпт
SYSTEM_PROMPT = """
Ты — бот техподдержки. Используй ТОЛЬКО эту базу знаний для ответов (не придумывай ничего лишнего):
{kb}
Если проблема не найдена, скажи: "Не нашёл точного решения. Опиши подробнее или обратись к модератору."
Отвечай кратко, по-русски, шаг за шагом. Будь полезным.
"""

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text or text.startswith('/'):
        return

    logger.info(f"Сообщение: {text[:50]}...")

    # Загружаем базу (из кеша)
    kb = get_knowledge_base()
    kb_lines = [line.strip() for line in kb.split('\n') if line.strip()]

    user_query = text.lower().strip()
    best_score = 0
    best_solution = None

    # Поиск по базе знаний (нечёткий)
    for i in range(0, len(kb_lines), 2):
        problem_line = kb_lines[i]
        if problem_line.lower().startswith("проблема:"):
            problem = problem_line[9:].strip()  # Убираем "Проблема: "
            score = fuzz.ratio(user_query, problem.lower())
            if score > 85 and score > best_score:
                best_score = score
                solution_line = kb_lines[i + 1] if i + 1 < len(kb_lines) else "Решение: Нет данных"
                best_solution = solution_line.replace("Решение: ", "").strip()

    # Если нашли — отвечаем из базы
    if best_solution:
        await update.message.reply_text(best_solution)
        logger.info(f"Ответ из базы (схожесть: {best_score}%)")
        return

    # Иначе — Groq
    full_prompt = SYSTEM_PROMPT.format(kb=kb) + f"\n\nЗапрос пользователя: {text}"
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": full_prompt}],
            max_tokens=250,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        await update.message.reply_text(reply)
        logger.info("Ответ от Groq")
    except Exception as e:
        logger.error(f"Ошибка Groq: {e}")
        await update.message.reply_text("Извини, временная ошибка. Попробуй позже.")

# Запуск
if __name__ == "__main__":
    logger.info("Запуск бота...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)