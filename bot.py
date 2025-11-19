
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

# === ЛОГИРОВАНИЕ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# === ГЛОБАЛЬНЫЙ ФЛАГ ПАУЗЫ ===
PAUSED = False

# === ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===
SHEET_ID = os.getenv("SHEET_ID")
RANGE_NAME = "Support!A:B"
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "service_account.json")

# === ПРОВЕРКА ПЕРЕМЕННЫХ ===
errors = []

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN or not TELEGRAM_TOKEN.strip():
    errors.append("TELEGRAM_TOKEN не задан или пустой")
else:
    logger.info("TELEGRAM_TOKEN загружен")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or not GROQ_API_KEY.strip():
    errors.append("GROQ_API_KEY не задан или пустой")
else:
    logger.info("GROQ_API_KEY загружен")

if not SHEET_ID or not SHEET_ID.strip():
    errors.append("SHEET_ID не задан")
else:
    logger.info("SHEET_ID загружен")

if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    errors.append(f"Файл credentials не найден: {GOOGLE_CREDENTIALS_PATH}")
else:
    try:
        with open(GOOGLE_CREDENTIALS_PATH) as f:
            creds_info = json.load(f)
        required_keys = ["type", "project_id", "private_key_id", "private_key", "client_email"]
        missing = [k for k in required_keys if k not in creds_info]
        if missing:
            errors.append(f"GOOGLE_CREDENTIALS: отсутствуют ключи: {', '.join(missing)}")
        else:
            logger.info("GOOGLE_CREDENTIALS — валидный JSON")
    except json.JSONDecodeError:
        errors.append("GOOGLE_CREDENTIALS — невалидный JSON")
    except Exception as e:
        errors.append(f"Ошибка чтения credentials: {e}")

ADMIN_ID_STR = os.getenv("ADMIN_ID", "")
if not ADMIN_ID_STR.strip():
    errors.append("ADMIN_ID не задан")
else:
    try:
        ADMIN_IDS = [int(uid.strip()) for uid in ADMIN_ID_STR.split(",") if uid.strip()]
        if not ADMIN_IDS:
            raise ValueError
        logger.info(f"Админы загружены: {ADMIN_IDS}")
    except ValueError:
        errors.append("ADMIN_ID должен быть числом или списком через запятую")

if errors:
    logger.error("ОШИБКИ ПЕРЕМЕННЫХ:")
    for err in errors:
        logger.error(f" → {err}")
    exit(1)
else:
    logger.info("Все переменные окружения загружены!")

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
    logger.info("Google Sheets подключён.")
except Exception as e:
    logger.error(f"Ошибка Google Auth: {e}")
    exit(1)

# === GROQ ===
try:
    client = Groq(api_key=GROQ_API_KEY, timeout=10)
    logger.info("Groq API подключён.")
except Exception as e:
    logger.error(f"Ошибка Groq: {e}")
    exit(1)

# === КЕШ БАЗЫ ЗНАНИЙ ===
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
            if problem:
                kb_entries.append(f"Проблема: {problem}\nРешение: {solution}")
        kb = "\n\n".join(kb_entries)
        logger.info(f"База знаний загружена: {len(rows)} записей.")
        return kb or "База знаний пуста."
    except Exception as e:
        logger.error(f"Ошибка чтения Sheets: {e}")
        get_knowledge_base.cache_clear()
        return "Временная ошибка базы знаний. Попробуй позже."

SYSTEM_PROMPT = """
Ты — бот техподдержки. Используй ТОЛЬКО эту базу знаний:
{kb}
Если не нашёл — скажи: "Не нашёл точного решения. Опиши подробнее или обратись к модератору."
Отвечай кратко, по-русски, шаг за шагом.
"""

# === БЛОКИРОВКА ЛИЧКИ ОТ НЕ-АДМИНОВ + КНОПКА ===
async def block_non_admin_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        return  # в группах всё работает как обычно

    if update.effective_user.id in ADMIN_IDS:
        return  # админы могут писать всё

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            text="Связаться с поддержкой",
            url="https://t.me/alexeymaloi"  # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
            # Замени на свой ник, ссылку в группу или на другого админа
            # Пример: "https://t.me/your_nick" или "https://t.me/+abc123"
        )
    ]])

    await update.message.reply_text(
        "Писать боту в личку могут только администраторы.\n"
        "Если нужна помощь — нажми кнопку ниже:",
        reply_markup=keyboard
    )
    return  # останавливаем дальнейшую обработку

# === АДМИН-КОМАНДЫ ===
async def admin_only_in_private(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if update.effective_chat.type != "private":
        await update.message.reply_text("Эта команда работает только в личных сообщениях.")
        return False
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Доступ запрещён.")
        return False
    return True

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context):
        return
    global PAUSED
    PAUSED = True
    await update.message.reply_text("Бот приостановлен. Используй /resume")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context):
        return
    global PAUSED
    PAUSED = False
    await update.message.reply_text("Бот возобновлён!")

async def status_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context):
        return
    status = "приостановлен" if PAUSED else "работает"
    await update.message.reply_text(f"Бот {status}")

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context):
        return
    get_knowledge_base.cache_clear()
    logger.info("Кеш сброшен по команде /reload")
    await update.message.reply_text("База знаний обновлена!")

async def health_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id in ADMIN_IDS:
        await update.message.reply_text("OK" if not PAUSED else "PAUSED")
    else:
        await update.message.reply_text("OK")

# === ОБРАБОТКА СООБЩЕНИЙ ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PAUSED:
        if update.effective_chat.type == "private" and update.effective_user.id in ADMIN_IDS:
            if update.message.text == "/status":
                await update.message.reply_text("Бот приостановлен.")
        return

    text = (update.message.text or update.message.caption or "").strip()
    if not text or text.startswith("/") or len(text) > 1000:
        return

    logger.info(f"Сообщение: {text[:50]}...")
    kb = get_knowledge_base()
    kb_blocks = [b.strip() for b in kb.split("\n\n") if b.strip()]
    user_query = text.lower()
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
            solution_lines = []
            for line in lines[1:]:
                if line.lower().startswith("решение:"):
                    solution_lines.append(line[9:].strip())
                else:
                    solution_lines.append(line)
            best_solution = "\n".join(solution_lines)

    if best_solution:
        await update.message.reply_text(best_solution)
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
        await update.message.reply_text(reply)
        logger.info("Ответ от Groq")
    except Exception as e:
        logger.error(f"Ошибка Groq: {e}")
        await update.message.reply_text("Извини, временная ошибка. Попробуй позже.")

# === АВТООБНОВЛЕНИЕ КЕША ===
def schedule_auto_reload(app):
    async def clear_cache(ctx):
        get_knowledge_base.cache_clear()
        logger.info("Автообновление: кеш сброшен")
    app.job_queue.run_once(clear_cache, when=10)
    app.job_queue.run_repeating(clear_cache, interval=300)

# === ЗАПУСК БОТА ===
if __name__ == "__main__":
    logger.info("Запуск бота...")
    PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"
    if PAUSED:
        logger.warning("Бот запущен в режиме ПАУЗЫ")

    request = HTTPXRequest(connection_pool_size=100)
    app = Application.builder().token(TELEGRAM_TOKEN).request(request).build()

    # ВАЖНО: ЭТОТ ХЕНДЛЕР ПЕРВЫЙ — блокирует личку от всех кроме админов
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_non_admin_private))

    # Основные хендлеры
    app.add_handler(MessageHandler(
        (filters.TEXT & ~filters.COMMAND) |
        (filters.CAPTION & ~filters.COMMAND),
        handle_message
    ))

    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_bot))
    app.add_handler(CommandHandler("health", health_check))

    schedule_auto_reload(app)

    logger.info("Бот запущен!")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)