import os
import json
import logging
import asyncio
from functools import lru_cache
from hashlib import md5
from cachetools import TTLCache
import re  # <-- новое
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

# (проверка переменных — без изменений)
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

PAUSE_FILE = "/app/paused.flag"
STATS_FILE = "/app/stats.json"

def is_paused(): return os.path.exists(PAUSE_FILE)
def set_paused(state: bool):
    if state: open(PAUSE_FILE, "w").close()
    else:
        try: os.remove(PAUSE_FILE)
        except FileNotFoundError: pass

if os.getenv("BOT_PAUSED", "").lower() == "true":
    set_paused(True)

# статистика и кэши (без изменений)
def load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE) as f: return json.load(f)
        except: return {"total": 0, "cached": 0, "groq_calls": 0}
    return {"total": 0, "cached": 0, "groq_calls": 0}

def save_stats(s):
    try:
        with open(STATS_FILE, "w") as f: json.dump(s, f)
    except: pass

stats = load_stats()
query_cache = TTLCache(maxsize=5000, ttl=3600)
response_cache = TTLCache(maxsize=3000, ttl=86400)

# Google Sheets, Chroma, модель — без изменений
# (тот же код, что и в последней версии)

# <<< НОВАЯ ФУНКЦИЯ ПРЕПРОЦЕССИНГА >>>
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# <<< ОСНОВНОЙ ХЕНДЛЕР С УЛУЧШЕННЫМ ПОИСКОМ >>>
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return
    raw_text = (update.message.text or update.message.caption or "").strip()
    if not raw_text or raw_text.startswith("/") or len(raw_text) > 1500:
        return

    text = preprocess_text(raw_text)  # <-- новая предобработка

    chat_id = update.effective_chat.id
    stats["total"] = stats.get("total", 0) + 1
    save_stats(stats)

    cache_key = md5(text.encode()).hexdigest()
    if cache_key in response_cache:
        stats["cached"] = stats.get("cached", 0) + 1
        save_stats(stats)
        await context.bot.send_message(chat_id=chat_id, text=response_cache[cache_key])
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    relevant = []
    if collection and cache_key not in query_cache:
        try:
            emb = get_embedder().encode(text).tolist()
            results = collection.query(
                query_embeddings=[emb],
                n_results=15,
                include=["metadatas", "distances"]
            )

            hard_threshold = 0.48   # ← понижен для коротких запросов
            soft_threshold = 0.78   # ← сильно повышен

            candidates = list(zip(results["distances"][0], results["metadatas"][0]))
            candidates.sort(key=lambda x: x[0])

            for dist, meta in candidates:
                if dist <= hard_threshold:
                    relevant.append(meta)
                    logger.info(f"✓ ТОП (dist={dist:.4f}): {meta['problem'][:90]}")
                elif dist <= soft_threshold and len(relevant) < 3:  # до 3 мягких если мало топовых
                    relevant.append(meta)
                    logger.info(f"↗ Мягкое (dist={dist:.4f}): {meta['problem'][:90]}")

            if relevant:
                relevant = relevant[:6]  # максимум 6 чанков

            query_cache[cache_key] = relevant

        except Exception as e:
            logger.error(f"Chroma ошибка: {e}")

    else:
        relevant = query_cache.get(cache_key, [])

    if not relevant:
        reply = "Точного решения пока нет в базе знаний.\nОпишите подробнее — передам специалисту."
        response_cache[cache_key] = reply
        await context.bot.send_message(chat_id=chat_id, text=reply)
        return

    context_str = "\n\n".join([f"Проблема: {m['problem']}\nРешение: {m['solution']}" for m in relevant])

    prompt = f"""Ты — опытный русскоязычный специалист техподдержки.
Отвечай ТОЛЬКО по базе знаний ниже. Используй решение из базы, адаптируй под вопрос пользователя, но ничего не придумывай.

База знаний:
{context_str}

Вопрос: {raw_text}

Ответ:"""

    async with GROQ_SEMAPHORE:
        stats["groq_calls"] = stats.get("groq_calls", 0) + 1
        save_stats(stats)
        try:
            response = await groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=600,
                temperature=0.05,   # ещё меньше креатива
                timeout=25,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq ошибка: {e}")
            reply = "Сервис временно недоступен, попробуйте позже."

    response_cache[cache_key] = reply
    await context.bot.send_message(chat_id=chat_id, text=reply)

# Остальной код (админки, запуск) — полностью из последней версии

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).request(HTTPXRequest(connection_pool_size=100)).concurrent_updates(False).build()
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_private))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_message))

    app.job_queue.run_once(lambda ctx: asyncio.create_task(update_vector_db_safe()), when=10)
    app.job_queue.run_repeating(lambda ctx: asyncio.create_task(update_vector_db_safe()), interval=600, first=600)

    logger.info("Бот запущен — УЛУЧШЕННАЯ ВЕРСИЯ ДЛЯ ТВОЕЙ БАЗЫ (ноябрь 2025)")
    app.run_polling(drop_pending_updates=True)