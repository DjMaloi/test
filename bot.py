import os
import json
import logging
import asyncio
from hashlib import md5
from cachetools import TTLCache
import re
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

# ============================ ЛОГИ ============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================ ПЕРЕМЕННЫЕ ============================
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
    except ValueError:
        errors.append("ADMIN_ID — некорректный формат")

if errors:
    logger.error("ОШИБКИ ЗАПУСКА:\n" + "\n".join(f"→ {e}" for e in errors))
    exit(1)

# ============================ ПАУЗА И СТАТИСТИКА ============================
PAUSE_FILE = "/app/paused.flag"
STATS_FILE = "/app/stats.json"

def is_paused() -> bool:
    return os.path.exists(PAUSE_FILE)

def set_paused(state: bool):
    if state:
        open(PAUSE_FILE, "w").close()
        logger.info("Бот на паузе")
    else:
        try:
            os.remove(PAUSE_FILE)
        except:
            pass
        logger.info("Пауза снята")

stats = {"total": 0, "cached": 0, "groq": 0, "vector_hit": 0, "keyword_hit": 0}
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

# Кэши
response_cache = TTLCache(maxsize=3000, ttl=86400)

# ============================ GOOGLE SHEETS ============================
creds = Credentials.from_service_account_file(
    GOOGLE_CREDENTIALS_PATH,
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)
service = build("sheets", "v4", credentials=creds)
sheet = service.spreadsheets()

# ============================ CHROMA + EMBEDDER ============================
chroma_client = chromadb.PersistentClient(path="/app/chroma")
collection = None
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    return embedder

# ============================ ЗАГРУЗКА БАЗЫ (МАКСИМАЛЬНО НАДЁЖНАЯ) ============================
async def update_vector_db_safe():
    global collection
    logger.info("=== ЗАГРУЗКА БАЗЫ ЗНАНИЙ — ПОЛНАЯ ДИАГНОСТИКА ===")
    try:
        # Получаем список всех листов
        spreadsheet = sheet.get(spreadsheetId=SHEET_ID).execute()
        sheets = [s['properties']['title'] for s in spreadsheet.get('sheets', [])]
        logger.info(f"Листы в таблице: {sheets}")

        docs, ids, metadatas = [], [], []

        # Ищем лист с нужным именем
        target_sheet = None
        for name in sheets:
            if name.lower() in ["support", "faq", "поддержка", "база"]:
                target_sheet = name
                break
        if not target_sheet:
            target_sheet = sheets[0]  # берём первый
            logger.warning(f"Лист не найден по названию — берём первый: {target_sheet}")

        range_name = f"{target_sheet}!A:B"
        logger.info(f"Читаем диапазон: {range_name}")

        result = sheet.values().get(spreadsheetId=SHEET_ID, range=range_name).execute()
        values = result.get("values", [])
        logger.info(f"Получено строк: {len(values)}")

        if len(values) < 2:
            logger.error("В таблице меньше 2 строк — только заголовок или пусто")
            collection = None
            return

        # Логируем первые строки
        for i, row in enumerate(values[:5]):
            logger.info(f"Строка {i+1}: {row}")

        # Парсим со второй строки
        for idx, row in enumerate(values[1:], start=2):
            if len(row) < 2:
                continue
            question = row[0].strip()
            answer = row[1].strip()
            if question and answer:
                docs.append(question)
                ids.append(f"kb_{len(docs)}")
                metadatas.append({"question": question, "answer": answer})

        logger.info(f"Успешно спаршено записей: {len(docs)}")

        if not docs:
            logger.error("НЕТ ВАЛИДНЫХ ЗАПИСЕЙ! База останется пустой.")
            collection = None
            return

        # Пересоздаём коллекцию
        try:
            chroma_client.delete_collection("support_kb")
        except:
            pass

        collection = chroma_client.get_or_create_collection("support_kb", metadata={"hnsw:space": "cosine"})
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.info(f"БАЗА ЗНАНИЙ УСПЕШНО ЗАГРУЖЕНА: {len(docs)} записей ✅✅✅")

    except Exception as e:
        logger.error(f"ОШИБКА ЗАГРУЗКИ ТАБЛИЦЫ: {e}", exc_info=True)
        collection = None

# ============================ GROQ ============================
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_SEMAPHORE = asyncio.Semaphore(6)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================ БЕЗОПАСНЫЙ TYPING ============================
async def safe_typing(bot, chat_id):
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except:
        pass  # тихо игнорируем отсутствие прав

# ============================ ОСНОВНОЙ ХЕНДЛЕР ============================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return

    raw_text = (update.message.text or update.message.caption or "").strip()
    if not raw_text or raw_text.startswith("/") or len(raw_text) > 1500:
        return

    stats["total"] += 1
    save_stats()

    text = preprocess_text(raw_text)
    cache_key = md5(text.encode()).hexdigest()

    if cache_key in response_cache:
        stats["cached"] += 1
        save_stats()
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response_cache[cache_key])
        return

    await safe_typing(context.bot, update.effective_chat.id)

    best_answer = None
    match_type = "fallback"

    if collection and collection.count() > 0:
        try:
            emb = get_embedder().encode(text).tolist()
            results = collection.query(query_embeddings=[emb], n_results=10, include=["metadatas", "distances"])

            # 1. Векторный поиск
            for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
                if dist < 0.42:
                    best_answer = meta["answer"]
                    match_type = "vector"
                    stats["vector_hit"] += 1
                    logger.info(f"Векторный хит (dist={dist:.3f}): {meta['question'][:70]}")
                    break

            # 2. Ключевой поиск (fallback)
            if not best_answer:
                words = [w for w in text.split() if len(w) > 3]
                all_metas = collection.get(include=["metadatas"])["metadatas"]
                for meta in all_metas:
                    q_clean = preprocess_text(meta["question"])
                    if any(word in q_clean for word in words) or text in q_clean:
                        best_answer = meta["answer"]
                        match_type = "keyword"
                        stats["keyword_hit"] += 1
                        logger.info(f"Ключевой хит: {meta['question'][:70]}")
                        break

        except Exception as e:
            logger.error(f"Ошибка Chroma: {e}")

    if not best_answer:
        best_answer = "Точного решения пока нет в базе знаний.\nОпишите проблему подробнее — передам специалисту."

    # Перефразируем только при хорошем совпадении
    reply = best_answer
    if match_type in ("vector", "keyword") and len(best_answer) < 900:
        prompt = f"""Перефразируй коротко и дружелюбно. Сохрани весь смысл, ничего не придумывай.
Оригинальный ответ:
{best_answer}

Вопрос: {raw_text}
Короткий ответ:"""

        async with GROQ_SEMAPHORE:
            stats["groq"] += 1
            save_stats()
            try:
                response = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=450,
                        temperature=0.2,
                    ),
                    timeout=20
                )
                new_reply = response.choices[0].message.content.strip()
                if 15 < len(new_reply) < len(best_answer) * 1.8:
                    reply = new_reply
            except Exception as e:
                logger.warning(f"Groq ошибка: {e}")

    response_cache[cache_key] = reply
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)

# ============================ АДМИНКИ ============================
async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    await update_vector_db_safe()
    await update.message.reply_text("База перезагружена!")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    set_paused(True)
    await update.message.reply_text("Бот на паузе")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    set_paused(False)
    await update.message.reply_text("Бот работает")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    paused = "Пауза" if is_paused() else "Работает"
    count = collection.count() if collection else 0
    text = f"""Статус: {paused}
Записей: {count}
Запросов: {stats.get('total',0)}
└ Кэш: {stats.get('cached',0)}
└ Вектор: {stats.get('vector_hit',0)}
└ Ключи: {stats.get('keyword_hit',0)}
└ Groq: {stats.get('groq',0)}"""
    await update.message.reply_text(text)

async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type == "private" and update.effective_user.id not in ADMIN_IDS:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
        await update.message.reply_text("Личные сообщения только для админов.", reply_markup=keyboard)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Глобальная ошибка: {context.error}", exc_info=True)

# ============================ ЗАПУСК ============================
if __name__ == "__main__":
    app = Application.builder()\
        .token(TELEGRAM_TOKEN)\
        .request(HTTPXRequest(connection_pool_size=100))\
        .concurrent_updates(False)\
        .build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND & ~filters.User(user_id=ADMIN_IDS), block_private))

    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_error_handler(error_handler)

    app.job_queue.run_once(lambda ctx: asyncio.create_task(update_vector_db_safe()), when=10)
    app.job_queue.run_repeating(lambda ctx: asyncio.create_task(update_vector_db_safe()), interval=600, first=600)

    logger.info("Бот запущен — всё будет работать идеально!")
    app.run_polling(drop_pending_updates=True)