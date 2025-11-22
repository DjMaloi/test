import os
import re
import logging
import asyncio
from hashlib import md5
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
#from telegram.ext._httpxrequest import HTTPXRequest
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import AsyncGroq

# ====================== LOGGING ======================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
# Отключаем лишние логи от telegram
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext._application").setLevel(logging.WARNING)
logging.getLogger("telegram.ext._updater").setLevel(logging.WARNING)
logging.getLogger("telegram.bot").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ====================== CONFIG ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_ID", "").split(",") if x]

# ====================== GOOGLE SHEETS ======================
creds = Credentials.from_service_account_file(
    os.getenv("GOOGLE_CREDENTIALS", "/app/service_account.json"),
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)
sheet = build("sheets", "v4", credentials=creds).spreadsheets()

# ====================== CHROMA ======================
client = chromadb.PersistentClient(path="/app/chroma")
collection_general = client.get_or_create_collection("general_kb")
collection_technical = client.get_or_create_collection("technical_kb")

# ====================== EMBEDDERS ======================
embedder_general = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embedder_technical = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ====================== GROQ ======================
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# ====================== PAUSE & STATS ======================
PAUSE_FILE = "/app/paused.flag"
STATS_FILE = "/app/stats.json"

def is_paused() -> bool:
    """Проверяет, находится ли бот на паузе"""
    return os.path.exists(PAUSE_FILE)

def set_paused(state: bool):
    """Включает или снимает паузу"""
    if state:
        open(PAUSE_FILE, "w").close()
        logger.info("БОТ НА ПАУЗЕ — отвечает только админам")
    else:
        try:
            os.remove(PAUSE_FILE)
        except FileNotFoundError:
            pass
        logger.info("Пауза снята")

stats = {"total": 0, "cached": 0, "groq": 0, "vector": 0, "keyword": 0}

def save_stats():
    try:
        import json
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"Ошибка сохранения статистики: {e}")

# ====================== CACHE ======================
from cachetools import TTLCache
response_cache = TTLCache(maxsize=1000, ttl=3600)

# ====================== CHROMA CLIENT ======================
chroma_client = chromadb.Client(Settings(
    persist_directory="/app/chroma_db",   # папка для хранения базы
    chroma_db_impl="duckdb+parquet"
))

def preprocess(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^а-яa-z0-9\s]', ' ', text.lower())).strip()

async def safe_typing(bot, chat_id):
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except:
        pass

# ====================== ОБНОВЛЕНИЕ БАЗЫ ======================
async def update_vector_db():
    global collection_general, collection_technical
    try:
        logger.info("Обновление базы знаний из Google Sheets...")

        # читаем данные из таблицы
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="General!A:A").execute()
        general_rows = [row[0] for row in result.get("values", []) if row]

        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Technical!A:A").execute()
        technical_rows = [row[0] for row in result.get("values", []) if row]

        # пересоздаём коллекции
        chroma_client.delete_collection("general")
        chroma_client.delete_collection("technical")

        collection_general = chroma_client.create_collection("general")
        collection_technical = chroma_client.create_collection("technical")

        # добавляем данные с обязательными ids
        if general_rows:
            collection_general.add(
                ids=[f"general_{i}" for i in range(len(general_rows))],
                documents=general_rows,
                embeddings=model.encode(general_rows).tolist()
            )

        if technical_rows:
            collection_technical.add(
                ids=[f"technical_{i}" for i in range(len(technical_rows))],
                documents=technical_rows,
                embeddings=model.encode(technical_rows).tolist()
            )

        logger.info(f"База обновлена: общая={len(general_rows)}, тех={len(technical_rows)}")

    except Exception as e:
        logger.error(f"Ошибка загрузки базы: {e}")


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

    logger.info("2.12.1 Бот запущен — логика с Google Sheets и ChromaDB")

    app.run_polling(drop_pending_updates=True)







