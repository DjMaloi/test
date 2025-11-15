import os
import logging
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from groq import Groq

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменные (замени на свои или используй env в Railway)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "ТВОЙ_ТОКЕН_ЗДЕСЬ")  # Из @BotFather
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "ТВОЙ_GROQ_КЛЮЧ_ЗДЕСЬ")  # Из console.groq.com
SHEET_ID = "1HBdZBWjlplVdZ4a7A5hdXxPyb2vyQ68ntIJ-oPfRwhA"  # Твоя таблица
RANGE_NAME = "Sheet1!A:B"  # A=Проблема, B=Решение (измени, если структура другая)

# Проверка переменных
if "ТВОЙ_ТОКЕН_ЗДЕСЬ" in TELEGRAM_TOKEN or "ТВОЙ_GROQ_КЛЮЧ_ЗДЕСЬ" in GROQ_API_KEY:
    logger.error("ЗАМЕНИ ТОКЕНЫ В КОДЕ ИЛИ В ENV!")
    exit(1)

# Google Sheets подключение
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
creds = Credentials.from_service_account_file('service_account.json', scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)
sheet_service = service.spreadsheets()

def get_knowledge_base():
    """Читает базу из твоей таблицы"""
    try:
        result = sheet_service.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = result.get('values', [])
        if not rows or len(rows) < 2:
            return "База знаний пуста. Добавь строки в таблицу."
        # Пропускаем заголовок
        kb_rows = rows[1:]
        kb_text = "\n".join([f"Проблема: {row[0]}\nРешение: {row[1] if len(row) > 1 else 'Нет решения'}" for row in kb_rows if row])
        logger.info(f"Загружено {len(kb_rows)} проблем из таблицы.")
        return kb_text
    except Exception as e:
        logger.error(f"Ошибка чтения таблицы: {e}")
        return "Ошибка доступа к базе. Проверь шаринг с Service Account."

# Groq клиент
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
Ты — умный бот техподдержки. Используй ТОЛЬКО эту базу знаний для ответов (не придумывай ничего лишнего):
{kb}

Если запрос не совпадает с базой, скажи: "Не нашёл точного решения. Опиши проблему подробнее или обратись к модератору."
Отвечай кратко, по-русски, в 2–3 шага. Будь полезным и дружелюбным.
"""

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text or text.startswith('/'):
        return

    logger.info(f"Новое сообщение: {text[:50]}...")
    
    # Загружаем базу
    kb = get_knowledge_base()
    full_prompt = SYSTEM_PROMPT.format(kb=kb) + f"\n\nЗапрос пользователя: {text}"

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Бесплатная быстрая модель
            messages=[{"role": "system", "content": full_prompt}],
            max_tokens=250,
            temperature=0.7  # Для естественности
        )
        reply = response.choices[0].message.content.strip()
        await update.message.reply_text(reply)
        logger.info("Ответ отправлен успешно.")
    except Exception as e:
        logger.error(f"Ошибка Groq: {e}")
        await update.message.reply_text("Извини, временная ошибка. Попробуй перефразировать вопрос.")

# Запуск бота
if __name__ == "__main__":
    logger.info("🚀 Бот запускается...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)