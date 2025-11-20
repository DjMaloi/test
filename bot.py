import os
import json
import logging
import asyncio
from functools import lru_cache
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

# ============================ КОНФИГ ============================
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "service_account.json")
ADMIN_ID_STR = os.getenv("ADMIN_ID", "")

# Проверка переменных
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

# Паузa и статистика
PAUSE_FILE = "/app/paused.flag"
STATS_FILE = "/app/stats.json"

def is_paused() -> bool:
    return os.path.exists(PAUSE_FILE)

def set_paused(state: bool):
    if state:
        open(PAUSE_FILE, "w").close()
        logger.info("Бот поставлен на паузу")
    else:
        try:
            os.remove(PAUSE_FILE)
        except FileNotFoundError:
            pass
        logger.info("Пауза снята")

if os.getenv("BOT_PAUSED", "").lower() == "true":
    set_paused(True)

def load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE) as f:
                return json.load(f)
        except:
            return {"total": 0, "cached": 0, "groq_calls": 0}
    return {"total": 0, "cached": 0, "groq_calls": 0}

def save_stats(s):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(s, f)
    except:
        pass

stats = load_stats()

# Кэши
query_cache = TTLCache(maxsize=5000, ttl=3600)
response_cache = TTLCache(maxsize=3000, ttl=86400)

# Google Sheets
try:
    creds = Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    logger.info("Google Sheets подключён")
except Exception as e:
    logger.error(f"Google Sheets ошибка: {e}")
    exit(1)

# Chroma + модель эмбеддингов
chroma_client = chromadb.PersistentClient(path="/app/chroma")
collection = None
embedder = None
MODEL_CACHE_DIR = "/app/model_cache"

def get_embedder():
    global embedder
    if embedder is None:
        logger.info("Грузим модель эмбеддингов...")
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        try:
            embedder = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                cache_folder=MODEL_CACHE_DIR,
                device="cpu"
            )
            logger.info("Модель загружена из локального кэша")
        except Exception:
            logger.warning("Кэша нет — скачиваем модель один раз...")
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            embedder = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                cache_folder=MODEL_CACHE_DIR
            )
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("Модель скачана и закэширована")
    return embedder

@lru_cache(maxsize=1)
def get_knowledge_base() -> str:
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute()
        values = result.get("values", [])
        if not values:
            logger.warning("Таблица пуста")
            return ""
        rows = values[1:] if len(values) > 1 and "проблема" in str(values[0][0]).lower() else values
        entries = []
        for row in rows:
            if len(row) >= 2 and row[0].strip():
                entries.append(f"Проблема: {row[0].strip()}\nРешение: {row[1].strip()}")
        kb = "\n\n".join(entries)
        logger.info(f"Загружено {len(entries)} записей из таблицы")
        return kb
    except Exception as e:
        logger.error(f"Ошибка чтения таблицы: {e}")
        return ""

async def update_vector_db_safe():
    global collection
    logger.info("=== Обновление векторной базы ===")
    kb = get_knowledge_base()
    if not kb:
        logger.warning("База знаний пуста")
        collection = None
        return

    blocks = [b.strip() for b in kb.split("\n\n") if b.strip()]
    docs, ids, metadatas = [], [], []
    for i, block in enumerate(blocks):
        lines = [l.strip() for l in block.split("\n")]
        if len(lines) < 2:
            continue
        problem = lines[0].replace("Проблема:", "", 1).strip()
        solution = "\n".join(lines[1:]).replace("Решение:", "", 1).strip()
        full_text = f"Проблема: {problem}\nРешение: {solution}"
        docs.append(full_text)
        ids.append(f"kb_{i}")
        metadatas.append({"problem": problem, "solution": solution})

    try:
        chroma_client.delete_collection("support_kb")
    except:
        pass

    collection = chroma_client.get_or_create_collection(
        "support_kb", metadata={"hnsw:space": "cosine"}
    )
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    logger.info(f"Векторная база обновлена: {len(docs)} документов ✅")

# Groq
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_SEMAPHORE = asyncio.Semaphore(6)

# ============================ ОСНОВНОЙ ХЕНДЛЕР ============================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return
    text = (update.message.text or update.message.caption or "").strip()
    if not text or text.startswith("/") or len(text) > 1500:
        return

    chat_id = update.effective_chat.id
    stats["total"] = stats.get("total", 0) + 1
    save_stats(stats)

    cache_key = md5(text.lower().encode()).hexdigest()
    if cache_key in response_cache:
        stats["cached"] = stats.get("cached", 0) + 1
        save_stats(stats)
        await context.bot.send_message(chat_id=chat_id, text=response_cache[cache_key])
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    relevant = []
    if collection is not None and cache_key not in query_cache:
        try:
            emb = get_embedder().encode(text).tolist()
            results = collection.query(
                query_embeddings=[emb],
                n_results=12,
                include=["metadatas", "distances"]
            )

            hard_threshold = 0.52
            soft_threshold = 0.68

            candidates = list(zip(results["distances"][0], results["metadatas"][0]))
            candidates.sort(key=lambda x: x[0])

            for dist, meta in candidates:
                if dist <= hard_threshold:
                    relevant.append(meta)
                    logger.info(f"✓ ТОП (dist={dist:.4f}): {meta['problem'][:90]}")
                elif dist <= soft_threshold and not relevant:
                    relevant.append(meta)
                    logger.info(f"↗ Мягкое (dist={dist:.4f}): {meta['problem'][:90]}")

            if not relevant:
                logger.info(f"Нет релевантных записей для «{text}»")

            query_cache[cache_key] = relevant

        except Exception as e:
            logger.error(f"Ошибка Chroma: {e}")

    else:
        relevant = query_cache.get(cache_key, [])

    if not relevant:
        reply = "Точного решения пока нет в базе знаний.\nОпишите подробнее — передам специалисту."
        response_cache[cache_key] = reply
        await context.bot.send_message(chat_id=chat_id, text=reply)
        return

    context_str = "\n\n".join([f"Проблема: {m['problem']}\nРешение: {m['solution']}" for m in relevant])

    prompt = f"""Ты — специалист техподдержки. Отвечай ТОЛЬКО на основе предоставленной базы знаний.
Если в базе нет точного ответа — скажи: «Точного решения пока нет в базе знаний. Опишите подробнее — передам специалисту.»
НЕ ПРИДУМЫВАЙ ничего от себя.

База знаний:
{context_str}

Вопрос пользователя: {text}

Ответ:"""

    async with GROQ_SEMAPHORE:
        stats["groq_calls"] = stats.get("groq_calls", 0) + 1
        save_stats(stats)
        try:
            response = awaitAdm groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=600,
                temperature=0.1,
                timeout=25,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq ошибка: {e}")
            reply = "Сервис временно недоступен, попробуйте позже."

    response_cache[cache_key] = reply
    await context.bot.send_message(chat_id=chat_id, text=reply)

# ============================ АДМИНКИ ============================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type == "private" and update.effective_user.id not in ADMIN_IDS:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
        await update.message.reply_text("Личные сообщения только для админов.", reply_markup=keyboard)

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    get_knowledge_base.cache_clear()
    await update_vector_db_safe()
    await update.message.reply_text("База знаний обновлена!")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    set_paused(True)
    await update.message.reply_text("Бот на паузе")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    set_paused(False)
    await update.message.reply_text("Бот работает")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    s = stats
    paused = "На паузе" if is_paused() else "Работает"
    coll_count = collection.count() if collection else 0
    await update.message.reply_text(
        f"Статус: {paused}\n"
        f"Записей в базе: {coll_count}\n"
        f"Запросов: {s.get('total',0)} | Кэш: {s.get('cached',0)} | Groq: {s.get('groq_calls',0)}"
    )

# ============================ ЗАПУСК ============================
if __name__ == "__main__":
    app = Application.builder()\
        .token(TELEGRAM_TOKEN)\
        .request(HTTPXRequest(connection_pool_size=100))\
        .concurrent_updates(False)\
        .build()

    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_private))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_message))

    app.job_queue.run_once(lambda ctx: asyncio.create_task(update_vector_db_safe()), when=10)
    app.job_queue.run_repeating(lambda ctx: asyncio.create_task(update_vector_db_safe()), interval=600, first=600)

    logger.info("Бот запущен — финальная версия (hard_threshold=0.52)")
    app.run_polling(drop_pending_updates=True)