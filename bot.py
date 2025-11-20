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
from telegram.error import NetworkError
from groq import AsyncGroq
import chromadb
from sentence_transformers import SentenceTransformer

# ============================ ОТКЛЮЧЕНИЕ ТЕЛЕМЕТРИИ CHROMADB ============================
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "true"

# ============================ КОНФИГ ============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

stats = {"total": 0, "cached": 0, "groq_calls": 0, "groq_fallback": 0}
if os.path.exists(STATS_FILE):
    try:
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)
    except:
        pass

def save_stats():
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
    except:
        pass

query_cache = TTLCache(maxsize=5000, ttl=3600)
response_cache = TTLCache(maxsize=3000, ttl=86400)

# Google Sheets
creds = Credentials.from_service_account_file(
    GOOGLE_CREDENTIALS_PATH,
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)
service = build("sheets", "v4", credentials=creds)
sheet = service.spreadsheets()

# Chroma
chroma_client = chromadb.PersistentClient(path="/app/chroma")
collection = None
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    return embedder

async def update_vector_db_safe():
    global collection
    logger.info("=== Обновление векторной базы ===")
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute()
        values = result.get("values", [])
        if len(values) < 2:
            logger.warning("Таблица пуста")
            collection = None
            return

        docs, ids, metadatas = [], [], []
        for i, row in enumerate(values):
            if i == 0: continue
            if len(row) < 2: continue
            question = row[0].strip()
            answer = row[1].strip()
            if not question or not answer: continue

            docs.append(question)
            ids.append(f"kb_{i}")
            metadatas.append({"question": question, "answer": answer})

        try:
            chroma_client.delete_collection("support_kb")
        except:
            pass

        collection = chroma_client.get_or_create_collection("support_kb", metadata={"hnsw:space": "cosine"})
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.info(f"База загружена: {len(docs)} записей ✅")
    except Exception as e:
        logger.error(f"Ошибка загрузки таблицы: {e}")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_SEMAPHORE = asyncio.Semaphore(6)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text or "пустой"

# ============================ БЛОКИРОВКА ЛИЧКИ ДЛЯ НЕ-АДМИНОВ ============================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type == "private" and update.effective_user.id not in ADMIN_IDS:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
        await update.message.reply_text("Личные сообщения только для админов.", reply_markup=keyboard)

# ============================ ОСНОВНОЙ ХЕНДЛЕР ============================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return

    raw_text = (update.message.text or update.message.caption or "").strip()
    if not raw_text or raw_text.startswith("/") or len(raw_text) > 1500:
        return

    text = preprocess_text(raw_text)
    chat_id = update.effective_chat.id
    stats["total"] = stats.get("total", 0) + 1
    save_stats()

    cache_key = md5(text.encode()).hexdigest()
    if cache_key in response_cache:
        stats["cached"] = stats.get("cached", 0) + 1
        save_stats()
        await context.bot.send_message(chat_id=chat_id, text=response_cache[cache_key])
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    best_answer = None
    if collection:
        try:
            emb = get_embedder().encode(text).tolist()
            results = collection.query(query_embeddings=[emb], n_results=1, include=["metadatas", "distances"])
            if results["distances"][0][0] < 0.85:
                best_answer = results["metadatas"][0][0]["answer"]
        except Exception as e:
            logger.error(f"Chroma ошибка: {e}")

    if not best_answer:
        reply = "Ваш вопрос зафиксирован. Специалист ответит в ближайшее время."
    else:
        prompt = f"""Перефразируй коротко и дружелюбно. Не придумывай ничего нового.

Оригинальный ответ:
{best_answer}

Вопрос пользователя: {raw_text}

Короткий ответ:"""

        reply = best_answer
        async with GROQ_SEMAPHORE:
            stats["groq_calls"] = stats.get("groq_calls", 0) + 1
            save_stats()
            try:
                response = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=500,
                        temperature=0.1,
                    ),
                    timeout=25
                )
                new_reply = response.choices[0].message.content.strip()
                if 10 < len(new_reply) < len(best_answer) * 2 and "к сожалению" not in new_reply.lower():
                    reply = new_reply
            except Exception as e:
                logger.warning(f"Groq упал: {e}")
                stats["groq_fallback"] = stats.get("groq_fallback", 0) + 1
                save_stats()

    response_cache[cache_key] = reply
    await context.bot.send_message(chat_id=chat_id, text=reply)

# ============================ АДМИНКИ ============================
async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    try:
        await update_vector_db_safe()
        await update.message.reply_text("База обновлена!")
    except Exception as e:
        logger.warning(f"Не удалось отправить ответ на /reload: {e}")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    set_paused(True)
    try:
        await update.message.reply_text("Бот на паузе")
    except: pass

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    set_paused(False)
    try:
        await update.message.reply_text("Бот работает")
    except: pass

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    try:
        s = stats
        paused = "Пауза" if is_paused() else "Работает"
        coll_count = collection.count() if collection else 0
        text = f"Статус: {paused}\nЗаписей: {coll_count}\nВсего: {s.get('total',0)} | Кэш: {s.get('cached',0)} | Groq: {s.get('groq_calls',0)} | Fallback: {s.get('groq_fallback',0)}"
        await update.message.reply_text(text)
    except Exception as e:
        logger.warning(f"Не удалось отправить /status: {e}")

# ============================ ГЛОБАЛЬНЫЙ ОБРАБОТЧИК ОШИБОК ============================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.warning(f"Telegram ошибка: {context.error}")

# ============================ ЗАПУСК ============================
if __name__ == "__main__":
    app = Application.builder()\
        .token(TELEGRAM_TOKEN)\
        .request(HTTPXRequest(connection_pool_size=100))\
        .concurrent_updates(False)\
        .build()

    # Основной обработчик — первым (чтобы админы получали ответы в личке)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_message))

    # Блокировка лички только для НЕ-АДМИНОВ
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND & ~filters.User(user_id=ADMIN_IDS), block_private))

    # Команды
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))

    # Обработка ошибок сети
    app.add_error_handler(error_handler)

    app.job_queue.run_once(lambda ctx: asyncio.create_task(update_vector_db_safe()), when=10)
    app.job_queue.run_repeating(lambda ctx: asyncio.create_task(update_vector_db_safe()), interval=600, first=600)

    logger.info("Бот запущен — ВСЁ РАБОТАЕТ: админы в личке отвечают, ошибки не падают, точные ответы из таблицы")
    app.run_polling(drop_pending_updates=True)