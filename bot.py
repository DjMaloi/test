import os
import json
import logging
import asyncio
from functools import lru_cache
from hashlib import md5
from cachetools import TTLCache

from google.oauth2.service import Credentials
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

# ============================ ПАУЗА + СТАТИСТИКА ============================
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

# ============================ КЭШИ ============================
query_cache = TTLCache(maxsize=5000, ttl=3600)      # эмбеддинги + результаты поиска
response_cache = TTLCache(maxsize=3000, ttl=86400)  # готовые ответы

# ============================ GOOGLE SHEETS ============================
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

# ============================ CHROMA (с volume) ============================
chroma_client = chromadb.PersistentClient(path="/app/chroma")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection = None

@lru_cache(maxsize=1)
def get_knowledge_base() -> str:
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute()
        values = result.get("values", [])
        rows = values[1:] if values and "проблема" in str(values[0][0]).lower() else values
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

async def update_vector_db():
    global collection
    kb = get_knowledge_base()
    if not kb:
        return

    blocks = [b.strip() for b in kb.split("\n\n") if b.strip()]
    docs, ids, metadatas = [], [], []

    for i, block in enumerate(blocks):
        lines = [l.strip() for l in block.split("\n")]
        if len(lines) < 2:
            continue
        problem = lines[0].replace("Проблема:", "", 1).strip()
        solution = "\n".join(lines[1:]).replace("Решение:", "", 1).strip()
        docs.append(f"Проблема: {problem}\nРешение: {solution}")
        ids.append(f"kb_{i}")
        metadatas.append({"problem": problem, "solution": solution})

    try:
        chroma_client.delete_collection("support_kb")
    except:
        pass

    collection = chroma_client.get_or_create_collection("support_kb")
    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    logger.info(f"Векторная база обновлена: {len(docs)} документов")

# ============================ GROQ ============================
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_SEMAPHORE = asyncio.Semaphore(6)

# ============================ ОБРАБОТКА СООБЩЕНИЙ ============================
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

    # Кэш готового ответа
    if cache_key in response_cache:
        stats["cached"] = stats.get("cached", 0) + 1
        save_stats(stats)
        await context.bot.send_message(chat_id=chat_id, text=response_cache[cache_key])
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        # Кэш поиска
        if cache_key not in query_cache and collection is not None:
            emb = embedder.encode(text).tolist()
            results = collection.query(
                query_embeddings=[emb],
                n_results=10,
                include=["metadatas", "distances"]
            )
            relevant = []
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                if dist < 0.38:
                    relevant.append(meta)
                if len(relevant) >= 4:
                    break
            query_cache[cache_key] = relevant
        else:
            relevant = query_cache.get(cache_key, [])

        if not relevant:
            reply = "Точного решения пока нет в базе знаний.\nОпишите подробнее — передам специалисту."
            response_cache[cache_key] = reply
            await context.bot.send_message(chat_id=chat_id, text=reply)
            return

        context_str = "\n\n".join([f"Проблема: {m['problem']}\nРешение: {m['solution']}" for m in relevant])

        prompt = f"""Ты — опытный русскоязычный специалист техподдержки.
Отвечай ТОЛЬКО по базе знаний ниже. Ничего не придумывай.
База знаний:
{context_str}

Вопрос: {text}
Ответ:"""

        async with GROQ_SEMAPHORE:
            stats["groq_calls"] = stats.get("groq_calls", 0) + 1
            save_stats(stats)
            response = await groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
                timeout=15,
            )
        reply = response.choices[0].message.content.strip()

        if any(w in reply.lower() for w in ["не знаю", "не уверен", "не могу найти"]):
            reply = "Точного решения пока нет в базе знаний.\nПередам ваш запрос специалисту."

        response_cache[cache_key] = reply
        await context.bot.send_message(chat_id=chat_id, text=reply)

    except Exception as e:
        logger.error(f"Ошибка обработки: {e}", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text="Сервис временно недоступен.")

# ============================ АДМИН КОМАНДЫ ============================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type == "private" and update.effective_user.id not in ADMIN_IDS:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
        await update.message.reply_text("Личные сообщения только для админов.", reply_markup=keyboard)

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    get_knowledge_base.cache_clear()
    await update_vector_db()
    await update.message.reply_text("База знаний обновлена!")

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
    s = stats
    await update.message.reply_text(
        f"Статус: {'На паузе' if is_paused() else 'Работает'}\n"
        f"Запросов: {s.get('total',0)} | Кэш: {s.get('cached',0)} | Groq: {s.get('groq_calls',0)}"
    )

async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PAUSED" if is_paused() else "OK")

# ============================ ЗАПУСК ============================
if __name__ == "__main__":
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .request(HTTPXRequest(connection_pool_size=100))
        .concurrent_updates(False)
        .build()
    )

    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_private))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("health", health_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_message))

    app.job_queue.run_once(lambda _: asyncio.create_task(update_vector_db()), when=2)
    app.job_queue.run_repeating(lambda _: asyncio.create_task(update_vector_db()), interval=600, first=600)

    logger.info("Бот запущен — финальная версия с volume")
    app.run_polling(drop_pending_updates=True)