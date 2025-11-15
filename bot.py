import os
import json
import logging
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from groq import Groq

# Логирование (для Render logs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Переменные из Render env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
SHEET_ID = "1HBdZBWjlplVdZ4a7A5hdXxPyb2vyQ68ntIJ-oPfRwhA"  # Ваша таблица
RANGE_NAME = "Sheet1!A:B"

# Проверка переменных
if not all([TELEGRAM_TOKEN, GROQ_API_KEY, GOOGLE_CREDENTIALS]):
    logger.error("ОШИБКА: Не заданы переменные в Render Variables!")
    exit(1)

logger.info("Переменные загружены успешно.")

# Google Sheets (через env JSON)
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

def get_knowledge_base():
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = result.get('values', [])[1:]  # Пропускаем заголовок
        kb = "\n".join([f"Проблема: {r[0]}\nРешение: {r[1] if len(r) > 1 else 'Нет решения'}" for r in rows if r])
        logger.info(f"База знаний загружена: {len(rows)} записей.")
        return kb or "База знаний пуста."
    except Exception as e:
        logger.error(f"Ошибка чтения Sheets: {e}")
        return "Ошибка доступа к базе знаний."

# Groq клиент
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq API подключён.")
except Exception as e:
    logger.error(f"Ошибка Groq: {e}")
    exit(1)

SYSTEM_PROMPT = """
Ты — бот техподдержки. Используй ТОЛЬКО эту базу знаний для ответов (не придумывай ничего лишнего):
{kb}

Если проблема не найдена, скажи: "Не нашёл точного решения. Опиши подробнее или обратись к модератору."
Отвечай кратко, по-русски, шаг за шагом. Будь полезным.
"""

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text or text.startswith('/'):
        return

    logger.info(f"Получено сообщение: {text[:50]}...")
    
    kb = get_knowledge_base()
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
        logger.info("Ответ отправлен.")
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        await update.message.reply_text("Извини, временная ошибка. Попробуй позже.")

# Запуск бота (с фиксом для polling)
if __name__ == "__main__":
    logger.info("🚀 Запуск бота...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)  # Игнорирует старые обновления