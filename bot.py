import os
import json
import logging
import asyncio
import re
import ssl
import certifi
from hashlib import md5
from cachetools import TTLCache
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler
)
from telegram.request import HTTPXRequest
from groq import AsyncGroq
import chromadb
from sentence_transformers import SentenceTransformer

# ====================== SSL ФИКС ======================
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ====================== ЛОГИ ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
telegram_logger = logging.getLogger("telegram")
telegram_logger.setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("telegram.bot").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ====================== КОНФИГ ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "service_account.json")
ADMIN_ID_STR = os.getenv("ADMIN_ID", "")

errors = []
for var, name in [(TELEGRAM_TOKEN, "TELEGRAM_TOKEN"), (GROQ_API_KEY, "GROQ_API_KEY"), (SHEET_ID, "SHEET_ID")]:
    if not var or not var.strip():
        errors.append(f"{name} не задан")
if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    errors.append(f"Файл credentials не найден: {GOOGLE_CREDENTIALS_PATH}")
if not ADMIN_ID_STR.strip():
    errors.append("ADMIN_ID не задан")
else:
    try:
        ADMIN_IDS = [int(i.strip()) for i in ADMIN_ID_STR.split(",") if i.strip()]
        logger.info(f"Админы: {ADMIN_IDS}")
    except ValueError:
        errors.append("ADMIN_ID — некорректный формат")

if errors:
    logger.error("ОШИБКИ ЗАПУСКА:\n" + "\n".join(f"→ {e}" for e in errors))
    exit(1)

# ====================== ПАУЗА И СТАТИСТИКА ======================
PAUSE_FILE = "/app/paused.flag"
STATS_FILE = "/app/stats.json"

def is_paused() -> bool:
    return os.path.exists(PAUSE_FILE)

def set_paused(state: bool):
    if state:
        open(PAUSE_FILE, "w").close()
        logger.info("БОТ НА ПАУЗЕ — отвечает только админам")
    else:
        try:
            os.remove(PAUSE_FILE)
        except:
            pass
        logger.info("Пауза снята")

stats = {"total": 0, "cached": 0, "groq": 0, "vector": 0, "keyword": 0}
if os.path.exists(STATS_FILE):
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except:
        pass

def save_stats():
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)
    except:
        pass

response_cache = TTLCache(maxsize=5000, ttl=86400)

# ====================== GOOGLE SHEETS ======================
creds = Credentials.from_service_account_file(
    GOOGLE_CREDENTIALS_PATH,
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)
service = build("sheets", "v4", credentials=creds)
sheet = service.spreadsheets()

# ====================== CHROMA ======================
chroma_client = chromadb.PersistentClient(path="/app/chroma")
collection = None
embedder = None

# ====================== ЗАГРУЗКА БАЗЫ ======================
async def update_vector_db():
    global collection
    logger.info("=== Загрузка базы знаний ===")
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute()
        values = result.get("values", [])
        logger.info(f"Получено строк: {len(values)}")

        if len(values) < 2:
            logger.warning("Таблица пуста")
            collection = None
            return

        docs, ids, metadatas = [], [], []
        for i, row in enumerate(values[1:], start=1):
            if len(row) < 2: continue
            q = row[0].strip()
            a = row[1].strip()
            if q and a:
                docs.append(q)
                ids.append(f"kb_{i}")
                metadatas.append({"question": q.split("\n")[0], "answer": a})

        try:
            chroma_client.delete_collection("support_kb")
        except:
            pass

        collection = chroma_client.get_or_create_collection("support_kb", metadata={"hnsw:space": "cosine"})
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.info(f"БАЗА ЗАГРУЖЕНА: {len(docs)} записей ✅")

    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}", exc_info=True)
        collection = None

# ====================== GROQ ======================
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_SEM = asyncio.Semaphore(8)

def preprocess(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^а-яa-z0-9\s]', ' ', text.lower())).strip()

async def safe_typing(bot, chat_id):
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except:
        pass

#=================Добавляем функцию для проверки, является ли запрос техническим=======

def match_technical_problem(text: str) -> bool:
    patterns = [
        r"(пк|компьютер|биос|экран|перезагрузка|ошибка|запуск)",
        r"(не работает|плохо работает|синий экран|зависает)",
    ]
    return any(re.search(pattern, text.lower()) for pattern in patterns)

# ====================== ХРАНЕНИЕ КОНТЕКСТА ПОЛЬЗОВАТЕЛЯ =====================

user_context = {}

def get_user_context(user_id: int):
    return user_context.get(user_id, [])

def set_user_context(user_id: int, context: list):
    user_context[user_id] = context

# ====================== ОПРОС ДЛЯ УТОЧНЕНИЯ ВОПРОСА ======================
async def ask_for_clarification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Да, это техническая проблема", callback_data="tech")],
        [InlineKeyboardButton("Нет", callback_data="non_tech")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Вы хотите сообщить о технической проблеме?", reply_markup=reply_markup)

async def handle_clarification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    choice = query.data
    if choice == "tech":
        # Проводим поиск с учетом, что это техническая проблема
        await handle_technical_query(update, context)
    else:
        # Проводим обычный поиск
        await handle_non_technical_query(update, context)

# ====================== ОБРАБОТКА СОобщЕНИЙ ======================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return

    raw_text = (update.message.text or update.message.caption or "").strip()
    if not raw_text or raw_text.startswith("/") or len(raw_text) > 1500:
        return

    user = update.effective_user
    username = f"@{user.username}" if user.username else ""
    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    display_name = f"{name} {username}".strip() or "Без имени"
    logger.info(f"ЗАПРОС → user={user.id} | {display_name} | \"{raw_text[:130]}{'...' if len(raw_text) > 130 else ''}\"")

    stats["total"] += 1
    save_stats()

    clean_text = preprocess(raw_text)
    user_id = update.effective_user.id
    user_history = get_user_context(user_id)
    user_history.append(raw_text)
    set_user_context(user_id, user_history[-5:])  # Храним только последние 5 запросов

    previous_queries = get_user_context(user_id)
    if previous_queries:
        additional_terms = " ".join(previous_queries)
        clean_text += " " + additional_terms

    # Проверяем, является ли запрос техническим
    is_technical = match_technical_problem(raw_text)

    if is_technical:
        await ask_for_clarification(update, context)
        return

    # Выбор модели для поиска
    def get_embedder(is_technical: bool = False):
        return SentenceTransformer("all-mpnet-base-v2" if is_technical else "paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

    embedder = get_embedder(is_technical)
    emb = embedder.encode(clean_text).tolist()

    # Ваш код для поиска в базе данных...

# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    app = Application.builder()\
        .token(TELEGRAM_TOKEN)\
        .request(HTTPXRequest(connection_pool_size=100))\
        .concurrent_updates(False)\
        .build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_clarification))

    logger.info("Бот 2.4 запущен — пауза работает, Alt+Enter поддерживается, всё идеально! Новая логика")
    app.run_polling(drop_pending_updates=True)
