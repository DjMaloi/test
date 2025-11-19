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

# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ===
PAUSED = False

# === ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===
SHEET_ID = os.getenv("SHEET_ID")
RANGE_NAME = "Support!A:B"          # можно и A2:B — но с A:B тоже ок
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "service_account.json")

# === ПРОВЕРКА ПЕРЕМЕННЫХ ===
errors = []
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN or not TELEGRAM_TOKEN.strip():
    errors.append("TELEGRAM_TOKEN не задан")
else:
    logger.info("TELEGRAM_TOKEN загружен")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or not GROQ_API_KEY.strip():
    errors.append("GROQ_API_KEY не задан")
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
        required = ["type", "project_id", "private_key_id", "private_key", "client_email"]
        missing = [k for k in required if k not in creds_info]
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
        logger.info(f"Админы: {ADMIN_IDS}")
    except ValueError:
        errors.append("ADMIN_ID должен быть числом или списком через запятую")

if errors:
    logger.error("ОШИБКИ ЗАПУСКА:")
    for err in errors:
        logger.error(f" → {err}")
    exit(1)
else:
    logger.info("Все переменные загружены успешно!")

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
    logger.info("Google Sheets API подключён")
except Exception as e:
    logger.error(f"Ошибка Google Auth: {e}")
    exit(1)

# === GROQ ===
try:
    client = Groq(api_key=GROQ_API_KEY, timeout=15)
    logger.info("Groq API подключён")
except Exception as e:
    logger.error(f"Ошибка Groq: {e}")
    exit(1)

# === БАЗА ЗНАНИЙ С КЭШЕМ ===
@lru_cache(maxsize=1)
def get_knowledge_base():
    try:
        logger.info(f"Загружаем базу знаний из {RANGE_NAME}")
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        logger.info(f"Сырой ответ Google: {result}")

        values = result.get("values", [])
        if not values:
            logger.warning("Таблица пуста или недоступна")
            return "База знаний временно недоступна."

        # Пропускаем заголовок "keywords | answer", если он есть
        rows = values
        if len(rows) > 0 and len(rows[0]) >= 2:
            first_cell = str(rows[0][0]).lower()
            if "keyword" in first_cell or "проблема" in first_cell:
                rows = rows[1:]
                logger.info("Заголовок обнаружен и пропущен")

        kb_entries = []
        for idx, row in enumerate(rows, start=2):  # нумерация с 2-й строки
            if not row:
                continue
            problem = row[0].strip() if len(row) > 0 else ""
            solution = row[1].strip() if len(row) > 1 else "Нет решения"
            if problem:
                kb_entries.append(f"Проблема: {problem}\nРешение: {solution}")
                logger.debug(f"Строка {idx}: {problem[:50]}... → добавлена")

        kb = "\n\n".join(kb_entries)
        logger.info(f"База знаний загружена: {len(kb_entries)} записей")
        return kb or "База знаний пуста."

    except Exception as e:
        logger.error(f"Ошибка чтения Sheets: {type(e).__name__}: {e}", exc_info=True)
        if "403" in str(e):
            logger.error("Проверь, расшарен ли сервисный аккаунт на таблицу!")
        get_knowledge_base.cache_clear()
        return "Временная ошибка базы знаний."

SYSTEM_PROMPT = """
Ты — бот техподдержки. Отвечай ТОЛЬКО на основе этой базы знаний:
{kb}

Если запрос очень похож на одну из «Проблема: …» — дай точное «Решение: …».
Если точного совпадения нет — скажи: "Не нашёл точного решения. Опиши подробнее или обратись к модератору."
Отвечай кратко, по-русски, шаг за шагом.
"""

# === БЛОКИРОВКА ЛИЧКИ ДЛЯ НЕ-АДМИНОВ ===
async def block_non_admin_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        return
    if update.effective_user.id in ADMIN_IDS:
        return
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(
            text="Связаться с поддержкой",
            url="https://t.me/alexeymaloi"
        )
    ]])
    await update.message.reply_text(
        "Писать боту в личку могут только администраторы.\n"
        "Нужна помощь — нажми кнопку ниже:",
        reply_markup=keyboard
    )

# === АДМИН-КОМАНДЫ ===
async def admin_only_in_private(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if update.effective_chat.type != "private":
        await update.message.reply_text("Команда работает только в ЛС с ботом.")
        return False
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("Доступ запрещён.")
        return False
    return True

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context): return
    global PAUSED
    PAUSED = True
    await update.message.reply_text("Бот приостановлен (/resume для возобновления)")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context): return
    global PAUSED
    PAUSED = False
    await update.message.reply_text("Бот снова работает!")

async def status_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context): return
    await update.message.reply_text("Пауза" if PAUSED else "Работает")

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await admin_only_in_private(update, context): return
    get_knowledge_base.cache_clear()
    await update.message.reply_text("Кэш базы знаний сброшен и перезагружен!")

async def health_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("OK" if not PAUSED else "PAUSED")

# === ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ СООБЩЕНИЙ ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PAUSED and update.effective_user.id not in ADMIN_IDS:
        return

    text = (update.message.text or update.message.caption or "").strip()
    if not text or text.startswith("/") or len(text) > 1000:
        return

    logger.info(f"Запрос: {text[:70]}...")

    kb = get_knowledge_base()
    kb_blocks = [b.strip() for b in kb.split("\n\n") if b.strip()]

    user_query = text.lower()
    best_score = 0
    best_solution = None

    # FUZZY-ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ
    for block in kb_blocks:
        lines = [line.strip() for line in block.split("\n")]
        if len(lines) < 2:
            continue

        # Берем текст после "Проблема: "
        problem_line = lines[0]
        if problem_line.lower().startswith("проблема:"):
            problem_text = problem_line[9:].strip()  # без "Проблема: "
        else:
            problem_text = problem_line

        score = fuzz.ratio(user_query, problem_text.lower())
        # можно понизить до 75, если хочешь больше совпадений
        if score > 82 and score > best_score:
            best_score = score
            best_solution = "\n".join(lines[1:]).replace("Решение: ", "", 1)

    if best_solution:
        await update.message.reply_text(best_solution.strip())
        logger.info(f"Ответ из fuzzy-поиска (score {best_score}%)")
        return

    # Если fuzzy не нашёл — идём в Groq с полной базой
    full_prompt = SYSTEM_PROMPT.format(kb=kb) + f"\n\nЗапрос пользователя: {text}"
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": full_prompt}],
            max_tokens=300,
            temperature=0.6,
        )
        reply = response.choices[0].message.content.strip()
        await update.message.reply_text(reply)
        logger.info("Ответ от Groq")
    except Exception as e:
        logger.error(f"Groq ошибка: {e}")
        await update.message.reply_text("Извини, сейчас не могу ответить. Попробуй позже или напиши админу.")

# === АВТООБНОВЛЕНИЕ КЭША ===
def schedule_auto_reload(app):
    async def clear_cache(ctx):
        get_knowledge_base.cache_clear()
        logger.info("Автообновление: кэш базы знаний сброшен")
    app.job_queue.run_once(clear_cache, when=15)
    app.job_queue.run_repeating(clear_cache, interval=600)  # каждые 10 минут

# === ЗАПУСК ===
if __name__ == "__main__":
    logger.info("Запуск бота...")
    PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

    request = HTTPXRequest(connection_pool_size=100)
    app = Application.builder().token(TELEGRAM_TOKEN).request(request).build()

    # Блокировка лички от всех кроме админов
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_non_admin_private))

    # Основные обработчики
    app.add_handler(MessageHandler(
        (filters.TEXT & ~filters.COMMAND) | (filters.CAPTION & ~filters.COMMAND),
        handle_message
    ))

    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_bot))
    app.add_handler(CommandHandler("health", health_check))

    schedule_auto_reload(app)

    logger.info("Бот запущен и готов к работе!")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)