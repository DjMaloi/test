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

# ====================== SSL FIX ======================
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ====================== CONFIG ======================
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

# ====================== STATS ======================
stats = {"total": 0, "cached": 0, "groq": 0, "vector": 0, "keyword": 0}
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

# ====================== EMBEDDERS ======================
embedder_general = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
embedder_technical = SentenceTransformer("all-mpnet-base-v2", device="cpu")
def preprocess(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^а-яa-z0-9\s]', ' ', text.lower())).strip()

async def safe_typing(bot, chat_id):
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except:
        pass

# ====================== UPDATE VECTOR DB ======================
async def update_vector_db():
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="A:B").execute()
        values = result.get("values", [])
        if not values:
            logger.warning("Google Sheets пустой")
            return

        # очищаем коллекции
        collection_general.delete()
        collection_technical.delete()

        for row in values:
            if len(row) >= 2:
                keyword, answer = row[0].strip(), row[1].strip()
                emb_general = embedder_general.encode(keyword).tolist()
                emb_technical = embedder_technical.encode(keyword).tolist()
                collection_general.add(
                    documents=[keyword],
                    metadatas=[{"question": keyword, "answer": answer}],
                    embeddings=[emb_general]
                )
                collection_technical.add(
                    documents=[keyword],
                    metadatas=[{"question": keyword, "answer": answer}],
                    embeddings=[emb_technical]
                )

        logger.info(f"Загружено {len(values)} записей из Google Sheets в ChromaDB")
    except Exception as e:
        logger.error(f"Ошибка загрузки базы: {e}", exc_info=True)

# ====================== MESSAGE HANDLER ======================
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

    cache_key = md5(clean_text.encode()).hexdigest()
    if cache_key in response_cache:
        stats["cached"] += 1
        save_stats()
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response_cache[cache_key])
        return

    await safe_typing(context.bot, update.effective_chat.id)

    best_answer = None
    source = "fallback"

    # === Google Sheets поиск ===
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="A:B").execute()
        values = result.get("values", [])
        for row in values:
            if len(row) >= 2:
                keyword, answer = row[0].strip().lower(), row[1].strip()
                if keyword in clean_text:
                    best_answer = answer
                    source = "keyword"
                    stats["keyword"] += 1
                    break
    except Exception as e:
        logger.error(f"Ошибка Google Sheets: {e}", exc_info=True)

    # === Векторный поиск ===
    if not best_answer and collection_general and collection_general.count() > 0:
        try:
            emb = embedder_general.encode(clean_text).tolist()
            results = collection_general.query(query_embeddings=[emb], n_results=10, include=["metadatas", "distances"])
            for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
                if dist < 0.4 and best_answer is None:
                    best_answer = meta["answer"]
                    source = "vector"
                    stats["vector"] += 1
        except Exception as e:
            logger.error(f"Chroma ошибка: {e}", exc_info=True)

    if not best_answer and collection_technical and collection_technical.count() > 0:
        try:
            emb = embedder_technical.encode(clean_text).tolist()
            results = collection_technical.query(query_embeddings=[emb], n_results=10, include=["metadatas", "distances"])
            for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
                if dist < 0.4 and best_answer is None:
                    best_answer = meta["answer"]
                    source = "vector"
                    stats["vector"] += 1
        except Exception as e:
            logger.error(f"Chroma ошибка: {e}", exc_info=True)

    # === Отправка ответа ===
    if best_answer:
        response_cache[cache_key] = best_answer
        await context.bot.send_message(chat_id=update.effective_chat.id, text=best_answer)

# ====================== BLOCK PRIVATE ======================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return

    if update.effective_chat.type == "private" and update.effective_user.id not in ADMIN_IDS:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
        await update.message.reply_text(
            "Писать боту в личку могут только администраторы.\nНужна помощь — нажми ниже:",
            reply_markup=keyboard
        )
# ====================== АДМИН-КОМАНДЫ ======================
async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    await update_vector_db()
    await update.message.reply_text("База перезагружена!")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    set_paused(True)
    await update.message.reply_text("Бот на паузе — обычные пользователи не получают ответы")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    set_paused(False)
    await update.message.reply_text("Бот снова работает")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    paused = "Пауза" if is_paused() else "Работает"
    count_general = collection_general.count() if collection_general else 0
    count_technical = collection_technical.count() if collection_technical else 0
    await update.message.reply_text(
        f"Статус: {paused}\n"
        f"Записей: общая={count_general}, тех={count_technical}\n"
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

    # блокируем личные чаты для не-админов
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & ~filters.COMMAND & ~filters.User(user_id=ADMIN_IDS),
        block_private
    ))

    # обработка сообщений в группах и от админов
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

    # команды админов
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_error_handler(error_handler)

    # первая загрузка базы через 15 секунд после старта
    app.job_queue.run_once(lambda _: asyncio.create_task(update_vector_db()), when=15)

    logger.info("2.8 Бот запущен — логика с Google Sheets и ChromaDB")

    app.run_polling(drop_pending_updates=True)
