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

# === ЛОГИ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PAUSED = False

# === ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===
SHEET_ID = os.getenv("SHEET_ID")
RANGE_NAME = "Support!A:B"
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "service_account.json")

errors = []

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN or not TELEGRAM_TOKEN.strip():
    errors.append("TELEGRAM_TOKEN не задан")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or not GROQ_API_KEY.strip():
    errors.append("GROQ_API_KEY не задан")

if not SHEET_ID or not SHEET_ID.strip():
    errors.append("SHEET_ID не задан")

# ADMIN_IDS
ADMIN_ID_STR = os.getenv("ADMIN_ID", "")
if not ADMIN_ID_STR.strip():
    errors.append("ADMIN_ID не задан")
else:
    try:
        ADMIN_IDS = [int(x.strip()) for x in ADMIN_ID_STR.split(",") if x.strip()]
        logger.info(f"Админы: {ADMIN_IDS}")
    except ValueError:
        errors.append("ADMIN_ID — только цифры через запятую")
        ADMIN_IDS = []

# ALLOWED_GROUP_IDS — группы, где бот может жить (включая топики/форумы)
ALLOWED_GROUP_IDS_STR = os.getenv("ALLOWED_GROUP_IDS", "")
if ALLOWED_GROUP_IDS_STR.strip():
    try:
        ALLOWED_GROUP_IDS = [int(x.strip()) for x in ALLOWED_GROUP_IDS_STR.split(",") if x.strip()]
        logger.info(f"Разрешённые группы: {ALLOWED_GROUP_IDS}")
    except ValueError:
        errors.append("ALLOWED_GROUP_IDS — только цифры через запятую")
        ALLOWED_GROUP_IDS = []
else:
    ALLOWED_GROUP_IDS = []
    logger.info("ALLOWED_GROUP_IDS не задан — бот работает везде")

if errors:
    for e in errors:
        logger.error(e)
    exit(1)

# === GOOGLE SHEETS ===
try:
    with open(GOOGLE_CREDENTIALS_PATH) as f:
        creds_info = json.load(f)
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    logger.info("Google Sheets подключён")
except Exception as e:
    logger.error(f"Google error: {e}")
    exit(1)

# === GROQ ===
try:
    client = Groq(api_key=GROQ_API_KEY, timeout=10)
    logger.info("Groq подключён")
except Exception as e:
    logger.error(f"Groq error: {e}")
    exit(1)

# === БАЗА ЗНАНИЙ ===
@lru_cache(maxsize=1)
def get_knowledge_base():
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        rows = result.get("values", [])[1:]
        entries = []
        for r in rows:
            problem = r[0].strip() if len(r) > 0 else ""
            solution = r[1].strip() if len(r) > 1 else "Нет решения"
            if problem:
                entries.append(f"Проблема: {problem}\nРешение: {solution}")
        kb = "\n\n".join(entries)
        logger.info(f"Загружено записей: {len(rows)}")
        return kb or "База пуста"
    except Exception as e:
        logger.error(f"Sheets error: {e}")
        get_knowledge_base.cache_clear()
        return "Ошибка базы знаний"

SYSTEM_PROMPT = """
Ты — бот техподдержки. Используй ТОЛЬКО эту базу знаний:
{kb}
Если не нашёл — скажи: "Не нашёл точного решения. Опиши подробнее или обратись к модератору."
Отвечай кратко, по-русски, шаг за шагом.
"""

# === 1. УНИВЕРСАЛЬНАЯ ПРОВЕРКА ГРУПП (работает с топиками!) ===
async def restrict_groups(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if chat.type not in ("group", "supergroup"):
        return

    if not ALLOWED_GROUP_IDS:  # если список пуст — пропускаем все группы
        return

    if chat.id not in ALLOWED_GROUP_IDS:
        try:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Этот бот работает только в официальных чатах проекта.\nДобавление в другие группы запрещено."
            )
        except:
            pass
        await context.bot.leave_chat(chat.id)
        logger.warning(f"Покинул запрещённую группу: {chat.title or 'Без названия'} ({chat.id})")

# === 2. БЛОКИРОВКА ЛИЧКИ ОТ НЕ-АДМИНОВ ===
async def block_private_non_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        return
    if update.effective_user.id in ADMIN_IDS:
        return

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            text="Связаться с поддержкой",
            url="https://t.me/alexeymaloi"  # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
            # Замени на свой ник или ссылку в группу поддержки
        )
    ]])

    await update.message.reply_text(
        "Писать боту в личку могут только администраторы.\nНужна помощь — нажми кнопку:",
        reply_markup=keyboard
    )

# === АДМИН-КОМАНДЫ ===
async def admin_only_in_private(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if update.effective_chat.type != "private":
        await update.message.reply_text("Команда только в личке")
        return False
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Доступ запрещён")
        return False
    return True

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context): return
    global PAUSED; PAUSED = True
    await update.message.reply_text("Бот приостановлен")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context): return
    global PAUSED; PAUSED = False
    await update.message.reply_text("Бот возобновлён")

async def status_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context): return
    await update.message.reply_text(f"Бот {'приостановлен' if PAUSED else 'работает'}")

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context): return
    get_knowledge_base.cache_clear()
    await update.message.reply_text("База обновлена")

async def health_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("OK" if not PAUSED else "PAUSED")

# === ОБРАБОТКА СООБЩЕНИЙ ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PAUSED:
        if update.effective_chat.type == "private" and update.effective_user.id in ADMIN_IDS:
            if update.message.text == "/status":
                await update.message.reply_text("Бот приостановлен")
        return

    text = (update.message.text or update.message.caption or "").strip()
    if not text or text.startswith("/") or len(text) > 1000:
        return

    return

    kb = get_knowledge_base()
    kb_blocks = [b.strip() for b in kb.split("\n\n") if b.strip()]
    user_query = text.lower()
    best_score = 0
    best_solution = None

    for block in kb_blocks:
        lines = [l.strip() for l in block.split("\n")]
        if len(lines) < 2 or not lines[0].lower().startswith("проблема:"):
            continue
        problem = lines[0][10:].strip()
        score = fuzz.ratio(user_query, problem.lower())
        if score > 85 and score > best_score:
            best_score = score
            best_solution = "\n".join(
                line[9:].strip() if line.lower().startswith("решение:") else line
                for line in lines[1:]
            )

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
        await update.message.reply_text("Временная ошибка, попробуй позже")

# === АВТООБНОВЛЕНИЕ ===
def schedule_auto_reload(app):
    async def job(_):
        get_knowledge_base.cache_clear()
        logger.info("Кеш очищен")
    app.job_queue.run_once(job, 10)
    app.job_queue.run_repeating(job, interval=300)

# === ЗАПУСК ===
if __name__ == "__main__":
    logger.info("Запуск бота...")
    PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

    app = Application.builder().token(TELEGRAM_TOKEN).request(HTTPXRequest(connection_pool_size=100)).build()

    # САМЫЕ ВАЖНЫЕ — ПЕРВЫМИ!
    app.add_handler(MessageHandler(
        filters.ChatType.GROUPS | filters.StatusUpdate.NEW_CHAT_MEMBERS,
        restrict_groups
    ), group=0)

    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_private_non_admin))

    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | (filters.CAPTION & ~filters.COMMAND), handle_message))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_bot))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("health", health_check))

    schedule_auto_reload(app)
    logger.info("Бот успешно запущен!")
    app.run_polling(drop_pending_updates=True)