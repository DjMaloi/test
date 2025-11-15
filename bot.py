import os
import logging
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === БЕЗОПАСНО: Только из env (Railway) ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHEET_ID = "1HBdZBWjlplVdZ4a7A5hdXxPyb2vyQ68ntIJ-oPfRwhA"  # Твоя таблица
RANGE_NAME = "Sheet1!A:B"

# Проверка: если нет переменных — остановка
if not TELEGRAM_TOKEN:
    logger.error("ОШИБКА: TELEGRAM_TOKEN не задан в Railway Variables!")
    exit(1)
if not GROQ_API_KEY:
    logger.error("ОШИБКА: GROQ_API_KEY не задан в Railway Variables!")
    exit(1)

# Google Sheets
creds = Credentials.from_service_account_file('service_account.json',
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly'])
service = build('sheets', 'v4', credentials=creds)
sheet_service = service.spreadsheets()

def get_knowledge_base():
    try:
        result = sheet_service.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = result.get('values', [])[1:]  # без заголовка
        kb = "\n".join([f"Проблема: {r[0]}\nРешение: {r[1] if len(r)>1 else '—'}" for r in rows])
        return kb or "База пуста."
    except Exception as e:
        logger.error(f"Sheets error: {e}")
        return "Ошибка чтения базы."

# Groq
client = Groq(api_key=GROQ_API_KEY)

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text.startswith('/'): return

    kb = get_knowledge_base()
    prompt = f"Ты — бот поддержки. Используй ТОЛЬКО базу:\n{kb}\n\nВопрос: {text}\nОтветь кратко по-русски."

    try:
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        await update.message.reply_text(resp.choices[0].message.content.strip())
    except Exception as e:
        await update.message.reply_text("Ошибка. Попробуй позже.")

# Запуск
app = Application.builder().token(TELEGRAM_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
logger.info("Бот запущен через Railway env!")
app.run_polling()