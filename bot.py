import os
import json
import logging
import asyncio
from functools import lru_cache
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
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except:
            return {"total": 0, "cached": 0, "groq_calls": 0, "groq_fallback": 0}
    return {"total": 0, "cached": 0, "groq_calls": 0, "groq_fallback": 0}

def save_stats(s):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(s, f)
    except:
        pass

stats = load_stats()
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

# Chroma + модель
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
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder=MODEL_CACHE_DIR)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return embedder

@lru_cache(maxsize=1)
def get_knowledge_base() -> str:
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute()
        values = result.get("values", [])
        if not values:
            return ""
        rows = values[1:] if len(values) > 1 and "проблема" in str(values[0][0]).lower() else values
        entries = [f"Проблема: {row[0].strip()}\nРешение: {row[1].strip()}" for row in rows if len(row) >= 2 and row[0].strip()]
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
        docs.append(f"Проблема: {problem}\nРешение: {solution}")
        ids.append(f"kb_{i}")
        metadatas.append({"problem": problem, "solution": solution})

    try:
        chroma_client.delete_collection("support_kb")
    except:
        pass

    collection = chroma_client.get_or_create_collection("support_kb", metadata={"hnsw:space": "cosine"})
    for i in range(0, len(docs), 100):
        collection.add(
            documents=docs[i:i + 100],
            ids=ids[i:i + 100],
            metadatas=metadatas[i:i + 100]
        )
    logger.info(f"Векторная база обновлена: {len(docs)} документов ✅")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_SEMAPHORE = asyncio.Semaphore(6)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
    save_stats(stats)

    cache_key = md5(text.encode()).hexdigest()
    if cache_key in response_cache:
        stats["cached"] = stats.get("cached", 0) + 1
        save_stats(stats)
        await context.bot.send_message(chat_id=chat_id, text=response_cache[cache_key])
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    relevant = []
    source_type = "no_match"

    if collection and cache_key not in query_cache:
        try:
            emb = get_embedder().encode(text).tolist()
            results = collection.query(query_embeddings=[emb], n_results=20, include=["metadatas", "distances"])
            candidates = list(zip(results["distances"][0], results["metadatas"][0]))
            candidates.sort(key=lambda x: x[0])

            # СУПЕР-МЯГКИЕ ПОРОГИ + ПРИНУДИТЕЛЬНЫЙ ТОП-1
            for dist, meta in candidates:
                if dist <= 0.92:
                    relevant.append(meta)

            if candidates:
                best_dist, best_meta = candidates[0]
                if best_meta not in [r for r in relevant]:
                    relevant.insert(0, best_meta)
                source_type = f"best_{best_dist:.3f}"
                logger.info(f"Найдено! Лучшее совпадение: {best_dist:.4f} → {best_meta['problem']}")

            relevant = relevant[:12]
            query_cache[cache_key] = relevant
        except Exception as e:
            logger.error(f"Chroma ошибка: {e}")
    else:
        relevant = query_cache.get(cache_key, [])

    # Fallback — если вообще ничего
    if not relevant and collection and collection.count() > 0:
        try:
            fb_emb = get_embedder().encode("поддержка помощь интернет").tolist()
            fb = collection.query(query_embeddings=[fb_emb], n_results=5, include=["metadatas"])
            relevant = [m for m in fb["metadatas"][0]]
            source_type = "fallback"
        except:
            pass

    # Контекст — всегда есть
    context_str = "\n\n".join([f"Проблема: {m['problem']}\nРешение: {m['solution']}" for m in relevant]) if relevant else get_knowledge_base()

    # ЖЁСТКИЙ ПРОМПТ — ЗАПРЕЩАЕМ ОТМАЗКИ
    prompt = f"""Ты — специалист техподдержки. Отвечай ТОЛЬКО по базе знаний ниже.
ВАЖНО: В базе ВСЕГДА есть подходящая запись. Бери САМУЮ ПЕРВУЮ (она самая релевантная).
Адаптируй её под вопрос пользователя. НИКОГДА не говори "к сожалению нет решения", "пока нет", "передал специалисту" — это строго запрещено.

База знаний (первая запись — самая подходящая):
{context_str}

Вопрос: {raw_text}

Ответ (дружелюбно, кратко, по делу):"""

    reply = None
    async with GROQ_SEMAPHORE:
        stats["groq_calls"] = stats.get("groq_calls", 0) + 1
        save_stats(stats)
        try:
            response = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=700,
                    temperature=0.1,
                ),
                timeout=28
            )
            reply = response.choices[0].message.content.strip()
            reply = re.sub(r'^Ответ\s*[:\-]?\s*', '', reply, flags=re.IGNORECASE).strip()
        except Exception as e:
            logger.warning(f"Groq упал: {e}")
            stats["groq_fallback"] = stats.get("groq_fallback", 0) + 1
            save_stats(stats)

    # АБСОЛЮТНЫЙ FALLBACK
    if not reply or len(reply) < 10:
        if relevant:
            reply = relevant[0]['solution']
        else:
            reply = "Ваш вопрос зафиксирован. Специалист ответит вам лично в ближайшее время."

    response_cache[cache_key] = reply
    await context.bot.send_message(chat_id=chat_id, text=reply)
    logger.info(f"Ответ отправлен | Совпадений: {len(relevant)} | Источник: {source_type}")

# ============================ АДМИНКИ ============================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type == "private" and update.effective_user.id not in ADMIN_IDS:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
        await update.message.reply_text("Личные сообщения только для админов.", reply_markup=keyboard)

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    get_knowledge_base.cache_clear()
    await update_vector_db_safe()
    await update.message.reply_text("База обновлена!")

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
    paused = "Пауза" if is_paused() else "Работает"
    coll_count = collection.count() if collection else 0
    await update.message.reply_text(
        f"Статус: {paused}\nЗаписей: {coll_count}\nЗапросов: {s.get('total',0)} | Кэш: {s.get('cached',0)} | Groq: {s.get('groq_calls',0)} | Fallback: {s.get('groq_fallback',0)}"
    )

# ============================ ЗАПУСК ============================
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

    logger.info("Бот запущен — ФИНАЛЬНАЯ НЕУБИВАЕМАЯ ВЕРСИЯ 20.11.2025")
    app.run_polling(drop_pending_updates=True)