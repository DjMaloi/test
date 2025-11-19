import os
import json
import logging
from functools import lru_cache
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest
from groq import Groq
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PAUSED = False

# === ПЕРЕМЕННЫЕ ===
errors = []
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "service_account.json")

# ADMIN_IDS
ADMIN_ID_STR = os.getenv("ADMIN_ID", "")
if not ADMIN_ID_STR or not ADMIN_ID_STR.strip():
    errors.append("ADMIN_ID не задан")
else:
    try:
        ADMIN_IDS = [int(x.strip()) for x in ADMIN_ID_STR.split(",") if x.strip()]
        logger.info(f"Админы: {ADMIN_IDS}")
    except:
        errors.append("ADMIN_ID — только цифры через запятую")
        ADMIN_IDS = []

# ALLOWED_GROUP_IDS
ALLOWED_GROUP_IDS_STR = os.getenv("ALLOWED_GROUP_IDS", "")
if ALLOWED_GROUP_IDS_STR.strip():
    try:
        ALLOWED_GROUP_IDS = [int(x.strip()) for x in ALLOWED_GROUP_IDS_STR.split(",") if x.strip()]
        logger.info(f"Разрешённые группы: {ALLOWED_GROUP_IDS}")
    except:
        ALLOWED_GROUP_IDS = []
else:
    ALLOWED_GROUP_IDS = []
    logger.info("ALLOWED_GROUP_IDS не задан — бот работает везде")

if errors:
    for e in errors: logger.error(e)
    exit(1)

# === GOOGLE & GROQ ===
try:
    with open(GOOGLE_CREDENTIALS_PATH) as f:
        creds_info = json.load(f)
    creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
except Exception as e:
    logger.error(f"Google error: {e}")
    exit(1)

try:
    client = Groq(api_key=GROQ_API_KEY, timeout=10)
except Exception as e:
    logger.error(f"Groq error: {e}")
    exit(1)

# === БАЗА ЗНАНИЙ ===
@lru_cache(maxsize=1)
def get_knowledge_base():
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = result.get("values", [])[1:]
        entries = [f"Проблема: {r[0].strip()}\nРешение: {r[1].strip() if len(r)>1 else 'Нет решения'}" for r in rows if r and r[0].strip()]
        kb = "\n\n".join(entries)
        logger.info(f"Загружено {len(rows)} записей")
        return kb or "База пуста"
    except Exception as e:
        logger.error(f"Sheets error: {e}")
        get_knowledge_base.cache_clear()
        return "Ошибка базы знаний"

SYSTEM_PROMPT = """Ты — бот техподдержки. Используй ТОЛЬКО эту базу знаний:\n{kb}\nЕсли не нашёл — скажи: "Не нашёл точного решения. Опиши подробнее или обратись к модератору."\nОтвечай кратко, по-русски, шаг за шагом."""

# === 1. ПРОВЕРКА ГРУПП (включая топики!) ===
async def restrict_groups(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if chat.type not in ("group", "supergroup"):
        return
    if not ALLOWED_GROUP_IDS:
        return
    if chat.id not in ALLOWED_GROUP_IDS:
        try:
            await context.bot.send_message(chat.id, "Бот работает только в официальных чатах проекта.")
        except:
            pass
        await context.bot.leave_chat(chat.id)
        logger.warning(f"Покинул группу {chat.title} ({chat.id})")

# === 2. ЛИЧКА: НЕ-АДМИНЫ — КНОПКА, АДМИНЫ — ПОЛНЫЙ ДОСТУП ===
async def private_access_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        return
    if update.effective_user.id in ADMIN_IDS:
        return  # админы проходят дальше — всё работает
    # Не-админы — только кнопка
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])  # ← замени
    await update.message.reply_text(
        "Писать боту в личку могут только администраторы.\nНужна помощь — нажми кнопку:",
        reply_markup=keyboard
    )
    return  # блокируем дальнейшую обработку

# === ОБРАБОТКА СООБЩЕНИЙ (для групп и админов в личке) ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PAUSED and update.effective_user.id not in ADMIN_IDS:
        return

    text = (update.message.text or update.message.caption or "").strip()
    if not text or text.startswith("/") or len(text) > 1000:
        return

    kb = get_knowledge_base()
    # твой поиск по базе и Groq — без изменений
    # ... (весь твой код из handle_message оставляем как есть) ...

    # Пример (оставь свой оригинальный код ниже):
    kb_blocks = [b.strip() for b in kb.split("\n\n") if b.strip()]
    user_query = text.lower()
    best_solution = None
    best_score = 0

    for block in kb_blocks:
        lines = [l.strip() for l in block.split("\n")]
        if len(lines) < 2 or not lines[0].lower().startswith("проблема:"): continue
        problem = lines[0][10:].strip()
        score = fuzz.ratio(user_query, problem.lower())
        if score > 85 and score > best_score:
            best_score = score
            best_solution = "\n".join(line[9:].strip() if line.lower().startswith("решение:") else line for line in lines[1:])

    if best_solution:
        await update.message.reply_text(best_solution)
        return

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": SYSTEM_PROMPT.format(kb=kb) + f"\n\nЗапрос: {text}"}],
            max_tokens=250,
            temperature=0.7,
        )
        await update.message.reply_text(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"Groq error: {e}")
        await update.message.reply_text("Временная ошибка")

# === АДМИН-КОМАНДЫ ===
async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    global PAUSED; PAUSED = True
    await update.message.reply_text("Бот приостановлен")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    global PAUSED; PAUSED = False
    await update.message.reply_text("Бот возобновлён")

async def status_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    await update.message.reply_text(f"Бот {'приостановлен' if PAUSED else 'работает'}")

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    get_knowledge_base.cache_clear()
    await update.message.reply_text("База обновлена")

async def health_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("OK" if not PAUSED else "PAUSED")

# === ЗАПУСК ===
if __name__ == "__main__":
    PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

    app = Application.builder().token(TELEGRAM_TOKEN).request(HTTPXRequest(connection_pool_size=100)).build()

    # КЛЮЧЕВОЙ ПОРЯДОК:
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS | filters.ChatType.GROUPS, restrict_groups), group=0)
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE, private_access_filter))  # личка: админы — всё, остальные — кнопка
    app.add_handler(MessageHandler(filters.TEXT | filters.CAPTION, handle_message))  # основной обработчик

    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_bot))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("health", health_check))

    def auto_reload(app):
        async def job(_): get_knowledge_base.cache_clear()
        app.job_queue.run_repeating(job, interval=300)
    auto_reload(app)

    logger.info("Бот запущен!")
    app.run_polling(drop_pending_updates=True)