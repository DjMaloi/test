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
    DispatcherHandlerStop,  # ← ЭТО ВАЖНО!
)
from telegram.request import HTTPXRequest
from groq import Groq
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PAUSED = False

# =========================== ПЕРЕМЕННЫЕ ===========================
errors = []
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN: errors.append("TELEGRAM_TOKEN пустой")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY: errors.append("GROQ_API_KEY пустой")

SHEET_ID = os.getenv("SHEET_ID")
if not SHEET_ID: errors.append("SHEET_ID пустой")

# ADMIN_IDS
ADMIN_ID_STR = os.getenv("ADMIN_ID", "")
ADMIN_IDS = [int(x.strip()) for x in ADMIN_ID_STR.split(",") if x.strip()] if ADMIN_ID_STR.strip() else []
if not ADMIN_IDS:
    errors.append("ADMIN_ID не задан")

# ALLOWED_GROUP_IDS
ALLOWED_GROUP_IDS = []
if os.getenv("ALLOWED_GROUP_IDS", "").strip():
    try:
        ALLOWED_GROUP_IDS = [int(x.strip()) for x in os.getenv("ALLOWED_GROUP_IDS").split(",") if x.strip()]
    except:
        errors.append("ALLOWED_GROUP_IDS — ошибка формата")

if errors:
    for e in errors: logger.error(e)
    exit(1)

logger.info(f"Админы: {ADMIN_IDS}")
logger.info(f"Разрешённые группы: {ALLOWED_GROUP_IDS or 'все'}")

# =========================== GOOGLE & GROQ ===========================
with open(os.getenv("GOOGLE_CREDENTIALS_PATH", "service_account.json")) as f:
    creds_info = json.load(f)
creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
service = build("sheets", "v4", credentials=creds)
sheet = service.spreadsheets()

client = Groq(api_key=GROQ_API_KEY, timeout=10)

# =========================== БАЗА ЗНАНИЙ ===========================
@lru_cache(maxsize=1)
def get_knowledge_base():
    try:
        rows = sheet.values().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute().get("values", [])[1:]
        entries = [f"Проблема: {r[0].strip()}\nРешение: {r[1].strip() if len(r)>1 else '—'}" for r in rows if r and r[0].strip()]
        return "\n\n".join(entries) or "База пуста"
    except Exception as e:
        logger.error(e)
        return "Ошибка загрузки базы знаний"

SYSTEM_PROMPT = """Ты — бот техподдержки. Используй ТОЛЬКО эту базу знаний:
{kb}
Если не нашёл — ответь: "Не нашёл точного решения. Опиши подробнее или обратись к модератору."
Отвечай кратко, по-русски, шаг за шагом."""

# =========================== 1. ЗАЩИТА ГРУПП ===========================
async def restrict_groups(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if chat.type not in ("group", "supergroup"):
        return
    if ALLOWED_GROUP_IDS and chat.id not in ALLOWED_GROUP_IDS:
        try:
            await context.bot.send_message(chat.id, "Бот работает только в официальных чатах проекта.")
        except:
            pass
        await context.bot.leave_chat(chat.id)
        logger.warning(f"Покинул группу {chat.id}")

# =========================== 2. ЛИЧКА — ТОЛЬКО АДМИНЫ ===========================
async def private_non_admin_block(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        return
    if update.effective_user.id in ADMIN_IDS:
        return  # админ — пропускаем дальше
    # не-админ — кнопка + стоп
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])  # ← ЗАМЕНИ!
    await update.message.reply_text(
        "Писать боту в личку могут только администраторы.\nНужна помощь — нажми кнопку ниже:",
        reply_markup=keyboard
    )
    raise DispatcherHandlerStop  # ← полностью глушим дальнейшую обработку

# =========================== 3. ОСНОВНАЯ ЛОГИКА ===========================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PAUSED and update.effective_user.id not in ADMIN_IDS:
        return

    text = (update.message.text or update.message.caption or "").strip()
    if not text or text.startswith("/"):
        return

    if len(text) > 1000:
        await update.message.reply_text("Слишком длинное сообщение")
        return

    kb = get_knowledge_base()
    best_solution = None
    best_score = 0
    for block in [b.strip() for b in kb.split("\n\n") if b.strip()]:
        lines = [l.strip() for l in block.split("\n")]
        if len(lines) < 2 or not lines[0].lower().startswith("проблема:"):
            continue
        problem = lines[0][10:].strip()
        score = fuzz.ratio(text.lower(), problem.lower())
        if score > 85 and score > best_score:
            best_score = score
            best_solution = "\n".join(l[9:].strip() if l.lower().startswith("решение:") else l for l in lines[1:])

    if best_solution:
        await update.message.reply_text(best_solution)
        return

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": SYSTEM_PROMPT.format(kb=kb) + f"\n\nЗапрос: {text}"}],
            max_tokens=300,
            temperature=0.7,
        )
        await update.message.reply_text(resp.choices[0].message.content.strip())
    except Exception as e:
        logger.error(e)
        await update.message.reply_text("Временная ошибка, попробуй позже")

# =========================== АДМИН-КОМАНДЫ ===========================
async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    global PAUSED; PAUSED = True
    await update.message.reply_text("Бот приостановлен")

async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    global PAUSED; PAUSED = False
    await update.message.reply_text("Бот возобновлён")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    await update.message.reply_text(f"Статус: {'приостановлен' if PAUSED else 'работает'}")

async def reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    get_knowledge_base.cache_clear()
    await update.message.reply_text("База обновлена")

# =========================== ЗАПУСК ===========================
if __name__ == "__main__":
    PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

    app = Application.builder().token(TELEGRAM_TOKEN).request(HTTPXRequest(connection_pool_size=100)).build()

    # КРИТИЧЕСКИ ВАЖНЫЙ ПОРЯДОК:
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS | filters.ChatType.GROUPS, restrict_groups), group=0)
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, private_non_admin_block))
    app.add_handler(MessageHandler(filters.TEXT | filters.CAPTION, handle_message))

    app.add_handler(CommandHandler("pause", pause))
    app.add_handler(CommandHandler("resume", resume))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("reload", reload))

    app.job_queue.run_repeating(lambda _: get_knowledge_base.cache_clear(), interval=300, first=10)

    logger.info("Бот успешно запущен!")
    app.run_polling(drop_pending_updates=True)