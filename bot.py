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
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from groq import AsyncGroq
import chromadb
from sentence_transformers import SentenceTransformer

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
    except:
        errors.append("ADMIN_ID — некорректный формат")

if errors:
    logger.error("ОШИБКИ ЗАПУСКА:\n" + "\n".join(errors))
    exit(1)

PAUSE_FILE = "/app/paused.flag"
STATS_FILE = "/app/stats.json"

def is_paused(): return os.path.exists(PAUSE_FILE)
def set_paused(state: bool):
    if state: open(PAUSE_FILE, "w").close()
    else:
        try: os.remove(PAUSE_FILE)
        except: pass

stats = json.load(open(STATS_FILE, "r")) if os.path.exists(STATS_FILE) else {"total":0, "cached":0, "groq":0, "fallback":0}
def save_stats(): 
    try: json.dump(stats, open(STATS_FILE, "w"))
    except: pass

query_cache = TTLCache(maxsize=5000, ttl=3600)
response_cache = TTLCache(maxsize=3000, ttl=86400)

# Google Sheets
creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
service = build("sheets", "v4", credentials=creds)
sheet = service.spreadsheets()

# Chroma
chroma_client = chromadb.PersistentClient(path="/app/chroma")
collection = None
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    return embedder

async def update_vector_db_safe():
    global collection
    logger.info("=== Обновление базы ===")
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute()
        values = result.get("values", [])
        if len(values) < 2:
            collection = None
            return

        docs, ids, metadatas = [], [], []
        for i, row in enumerate(values):
            if i == 0: continue  # пропуск заголовка, если есть
            if len(row) < 2: continue
            question = row[0].strip()
            answer = row[1].strip()
            if not question or not answer: continue

            # Эмбеддим ТОЛЬКО колонку A — поиск строго по вопросам/ключам
            docs.append(question)
            ids.append(f"kb_{i}")
            metadatas.append({"question": question, "answer": answer})

        try: chroma_client.delete_collection("support_kb")
        except: pass
        collection = chroma_client.get_or_create_collection("support_kb", metadata={"hnsw:space": "cosine"})
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.info(f"База загружена: {len(docs)} записей")
    except Exception as e:
        logger.error(f"Ошибка загрузки таблицы: {e}")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
GROQ_SEMAPHORE = asyncio.Semaphore(6)

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================ ХЕНДЛЕР ============================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_paused() and update.effective_user.id not in ADMIN_IDS: return

    raw = (update.message.text or update.message.caption or "").strip()
    if not raw or raw.startswith("/") or len(raw) > 1500: return

    text = preprocess(raw)
    chat_id = update.effective_chat.id
    stats["total"] += 1
    save_stats()

    key = md5(text.encode()).hexdigest()
    if key in response_cache:
        stats["cached"] += 1
        save_stats()
        await context.bot.send_message(chat_id=chat_id, text=response_cache[key])
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    best_answer = None
    if collection:
        try:
            emb = get_embedder().encode(text).tolist()
            results = collection.query(query_embeddings=[emb], n_results=1, include=["metadatas", "distances"])
            if results["distances"][0][0] < 0.9:  # очень мягкий порог
                best_answer = results["metadatas"][0][0]["answer"]
                logger.info(f"Найдено точное совпадение: {results['distances'][0][0]:.3f}")
        except Exception as e:
            logger.error(f"Chroma ошибка: {e}")

    # Если ничего не нашлось — fallback на всю базу (редко)
    if not best_answer and collection:
        try:
            results = collection.query(query_embeddings=[get_embedder().encode("помощь").tolist()], n_results=1, include=["metadatas"])
            best_answer = results["metadatas"][0][0]["answer"]
        except: pass

    if not best_answer:
        reply = "Ваш вопрос зафиксирован, специалист ответит в ближайшее время."
    else:
        # Используем Groq только для лёгкой перефразировки ОДНОГО ответа
        prompt = f"""Перефразируй следующий ответ техподдержки коротко и дружелюбно. Не добавляй ничего нового, не смешивай с другими ответами.

Оригинальный ответ:
{best_answer}

Вопрос пользователя: {raw}

Короткий ответ:"""

        reply = best_answer  # по умолчанию — чистый ответ из B
        async with GROQ_SEMAPHORE:
            stats["groq"] += 1
            save_stats()
            try:
                resp = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=500,
                        temperature=0.1,
                    ),
                    timeout=25
                )
                new_reply = resp.choices[0].message.content.strip()
                if len(new_reply) > 10 and "к сожалению" not in new_reply.lower():
                    reply = new_reply
            except Exception as e:
                logger.warning(f"Groq fallback: {e}")
                stats["fallback"] += 1
                save_stats()

    response_cache[key] = reply
    await context.bot.send_message(chat_id=chat_id, text=reply)

# ============================ АДМИНКИ И ЗАПУСК ============================
# (остальные функции без изменений — block_private, reload_kb, pause/resume, status — как в предыдущей версии)

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).request(HTTPXRequest(connection_pool_size=100)).concurrent_updates(False).build()

    #app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_private))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_message))

    app.job_queue.run_once(lambda ctx: asyncio.create_task(update_vector_db_safe()), when=10)
    app.job_queue.run_repeating(lambda ctx: asyncio.create_task(update_vector_db_safe()), interval=600, first=600)

    logger.info("Бот запущен — ТОЧНЫЕ ОТВЕТЫ ИЗ КОЛОНКИ B + ЛЁГКАЯ ПЕРЕФРАЗИРОВКА")
    app.run_polling(drop_pending_updates=True)