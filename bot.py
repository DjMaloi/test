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
general_kb_path = os.getenv("GENERAL_KB_PATH", "/app/chroma/general_kb")
technical_kb_path = os.getenv("TECHNICAL_KB_PATH", "/app/chroma/technical_kb")

chroma_client_general = chromadb.PersistentClient(path=general_kb_path)
chroma_client_technical = chromadb.PersistentClient(path=technical_kb_path)

collection_general = chroma_client_general.get_or_create_collection("general_kb", metadata={"hnsw:space": "cosine"})
collection_technical = chroma_client_technical.get_or_create_collection("technical_kb", metadata={"hnsw:space": "cosine"})

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

# ====================== ОБРАБОТКА СООБЩЕНИЙ ======================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return

    raw_text = (update.message.text or update.message.caption or "").strip()
    if not raw_text or raw_text.startswith("/") or len(raw_text) > 1500:
        return

    # === ЛОГИРУЕМ КАЖДЫЙ ВХОДЯЩИЙ ЗАПРОС ===
    user = update.effective_user
    username = f"@{user.username}" if user.username else ""
    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    display_name = f"{name} {username}".strip() or "Без имени"
    logger.info(f"ЗАПРОС → user={user.id} | {display_name} | \"{raw_text[:130]}{'...' if len(raw_text) > 130 else ''}\"")

    stats["total"] += 1
    save_stats()

    clean_text = preprocess(raw_text)
    # Сохраняем историю запросов
    user_id = update.effective_user.id
    user_history = get_user_context(user_id)
    user_history.append(raw_text)
    set_user_context(user_id, user_history[-5:])  # Храним только последние 5 запросов
    
    previous_queries = get_user_context(user_id)
    if previous_queries:
        additional_terms = " ".join(previous_queries)
        clean_text += " " + additional_terms  # Добавляем к запросу

    # Проверяем, является ли запрос техническим
    is_technical = match_technical_problem(raw_text)

    # Выбираем модель в зависимости от типа запроса
    def get_embedder(is_technical: bool = False):
        if is_technical:
            return SentenceTransformer("all-mpnet-base-v2", device="cpu")
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

    embedder = get_embedder(is_technical)

    cache_key = md5(clean_text.encode()).hexdigest()

    if cache_key in response_cache:
        stats["cached"] += 1
        save_stats()
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response_cache[cache_key])
        return

    await safe_typing(context.bot, update.effective_chat.id)

    best_answer = None
    source = "fallback"

    if collection_general and collection_general.count() > 0:
        try:
            emb = embedder.encode(clean_text).tolist()
            results = collection_general.query(
                query_embeddings=[emb],
                n_results=10,
                include=["metadatas", "distances"]
            )
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            # Векторный поиск для общего хранилища
            for i, (dist, meta) in enumerate(zip(distances, metadatas), 1):
                q_preview = meta["question"].split("\n")[0][:80].replace("\n", " ")
                if dist < 0.4 and best_answer is None:
                    best_answer = meta["answer"]
                    source = "vector"
                    stats["vector"] += 1

        except Exception as e:
            logger.error(f"Chroma ошибка: {e}", exc_info=True)

    if not best_answer and collection_technical and collection_technical.count() > 0:
        try:
            emb = embedder.encode(clean_text).tolist()
            results = collection_technical.query(
                query_embeddings=[emb],
                n_results=10,
                include=["metadatas", "distances"]
            )
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            # Векторный поиск для технического хранилища
            for i, (dist, meta) in enumerate(zip(distances, metadatas), 1):
                q_preview = meta["question"].split("\n")[0][:80].replace("\n", " ")
                if dist < 0.4 and best_answer is None:
                    best_answer = meta["answer"]
                    source = "vector"
                    stats["vector"] += 1

        except Exception as e:
            logger.error(f"Chroma ошибка: {e}", exc_info=True)

    # Кэшируем ответ и отправляем
    if best_answer:
        response_cache[cache_key] = best_answer
        await context.bot.send_message(chat_id=update.effective_chat.id, text=best_answer)

# ====================== ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ======================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return

    if update.effective_chat.type == "private" and update.effective_user.id not in ADMIN_IDS:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
        await update.message.reply_text(
            "Писать боту в личку могут только администраторы.\nНужна помощь — нажми ниже:",
            reply_markup=keyboard
        )

# ====================== АДМИНКИ ======================
async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    await update_vector_db()
    await update.message.reply_text("База перезагружена!")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    set_paused(True)
    await update.message.reply_text("Бот на паузе — обычные пользователи не получают ответы")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    set_paused(False)
    await update.message.reply_text("Бот снова работает")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    paused = "Пауза" if is_paused() else "Работает"
    count = collection.count() if collection else 0
    await update.message.reply_text(
        f"Статус: {paused}\n"
        f"Записей: {count}\n"
        f"Запросов: {stats['total']} (кэш: {stats['cached']})\n"
        f"Вектор: {stats['vector']} | Ключи: {stats['keyword']} | Groq: {stats['groq']}"
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Глобальная ошибка: {context.error}", exc_info=True)

# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    app = Application.builder()\
        .token(TELEGRAM_TOKEN)\
        .request(HTTPXRequest(connection_pool_size=100))\
        .concurrent_updates(False)\
        .build()

    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & ~filters.COMMAND & ~filters.User(user_id=ADMIN_IDS),
        block_private
    ))

    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND &
        (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP | filters.User(user_id=ADMIN_IDS)),
        handle_message
    ))
    app.add_handler(MessageHandler(
        filters.CAPTION & ~filters.COMMAND &
        (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP | filters.User(user_id=ADMIN_IDS)),
        handle_message
    ))

    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_error_handler(error_handler)

    app.job_queue.run_once(lambda _: asyncio.create_task(update_vector_db()), when=15)

    logger.info("2.6 Бот запущен — новая логика, 2 языковые базы")

    app.run_polling(drop_pending_updates=True)
