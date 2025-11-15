import os
import json
import logging
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ПЕРЕМЕННЫЕ ИЗ RAILWAY ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
SHEET_ID = "1HBdZBWjlplVdZ4a7A5hdXxPyb2vyQ68ntIJ-oPfRwhA"  # Твоя таблица
RANGE_NAME = "Sheet1!A:B"

if not all([TELEGRAM_TOKEN, GROQ_API_KEY, GOOGLE_CREDENTIALS]):
    logger.error("ОШИБКА: Не заданы переменные в Railway!")
    exit(1)

# === Google Sheets через строку JSON ===
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

def get_kb():
    try:
        data = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = data.get('values', [])[1:]
        kb = "\n".join([f"Q: {r[0]}\nA: {r[1] if len(r)>1 else '—'}" for r in rows])
        return kb or "База пуста."
    except Exception as e:
        return "Ошибка чтения таблицы."

client = Groq(api_key=GROQ_API_KEY)

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text.startswith('/'): return
    kb = get_kb()
    prompt = f"База:\n{kb}\n\nВопрос: {text}\nОтветь кратко по-русски."
    try:
        resp = client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], max_tokens=200)
        await update.message.reply_text(resp.choices[0].message.content.strip())
    except Exception as e:
        await update.message.reply_text("Ошибка. Попробуй позже.")

app = Application.builder().token(TELEGRAM_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
logger.info("Бот запущен — безопасно!")
app.run_polling()