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
chroma_client = chromadb.PersistentClient(path="/app/chroma")
collection = None
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
       # embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        embedder = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
    return embedder

# ====================== ЗАГРУЗКА БАЗЫ (поддерживает Alt+Enter) ======================
async def update_vector_db():
    global collection
    logger.info("=== Перезагрузка базы знаний (фикс размерности) ===")
    try:
        result = sheet.tables().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute()
        values = result.get("values", [])
        logger.info(f"Получено строк: {len(values)}")

        if len(values) < 2:
            logger.warning("Таблица пуста")
            collection = None
            return

        docs, ids, metadatas = [], [], []

        for i, row in enumerate(values[1:], start=1):
            if len(row) < 2: continue
            raw_q = row[0].strip()
            raw_a = row[1].strip()
            if not raw_q or not raw_a: continue

            clean_q = preprocess(raw_q)
            docs.append(clean_q)
            ids.append(f"kb_{i}")
            metadatas.append({
                "question": raw_q.split("\n")[0][:200],
                "answer": raw_a
            })

        # ←←←← ГЛАВНОЕ: УДАЛЯЕМ СТАРУЮ КОЛЛЕКЦИЮ ОБЯЗАТЕЛЬНО!
        try:
            chroma_client.delete_collection("support_kb")
            logger.info("Старая коллекция удалена — несовместимая размерность векторов")
        except:
            pass  # если нет — ок

        collection = chroma_client.get_or_create_collection(
            "support_kb",
            metadata={"hnsw:space": "cosine"}
        )
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.info(f"БАЗА ПЕРЕЗАГРУЖЕНА УСПЕШНО ✅ | записей: {len(docs)} | размерность векторов: {len(get_embedder().encode('тест')[0])}")

    except Exception as e:
        logger.error(f"Ошибка загрузки базы: {e}", exc_info=True)
        collection = None

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
    # ПАУЗА — САМАЯ ПЕРВАЯ ПРОВЕРКА
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
    cache_key = md5(clean_text.encode()).hexdigest()

    if cache_key in response_cache:
        stats["cached"] += 1
        save_stats()
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response_cache[cache_key])
        return

    await safe_typing(context.bot, update.effective_chat.id)

    best_answer = None
    source = "fallback"

    if collection and collection.count() > 0:
        try:
            emb = get_embedder().encode(clean_text).tolist()
            results = collection.query(
                query_embeddings=[emb],
                n_results=10,
                include=["metadatas", "distances"]
            )
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            top_log = []
            selected_dist = None
            selected_preview = None

            for i, (dist, meta) in enumerate(zip(distances, metadatas), 1):
                q_preview = meta["question"].split("\n")[0][:80].replace("\n", " ")
                top_log.append(f"#{i} dist={dist:.4f} \"{q_preview}\"")

                if dist < 0.42 and best_answer is None:
                    best_answer = meta["answer"]
                    source = "vector"
                    stats["vector"] += 1
                    selected_dist = dist
                    selected_preview = q_preview

            if best_answer:
                logger.info(f"ВЕКТОР ✓ | distance={selected_dist:.4f} | user={user.id} ({display_name}) | "
                            f"запрос=\"{raw_text[:100]}{'...' if len(raw_text)>100 else ''}\" | "
                            f"→ \"{selected_preview}\" | топ-3: {' | '.join(top_log[:3])}")
            else:
                best_dist = distances[0] if distances else 1.0
                best_q = metadatas[0]["question"].split("\n")[0][:80] if metadatas else "—"
                logger.info(f"ВЕКТОР ✗ (порог >0.42) | лучший distance={best_dist:.4f} → \"{best_q}\" | "
                            f"user={user.id} ({display_name}) | запрос=\"{raw_text[:100]}{'...' if len(raw_text)>100 else ''}\" | "
                            f"топ-5: {' | '.join(top_log[:5])}")

            # Ключевой поиск, если вектор не нашёл
            if not best_answer:
                words = [w for w in clean_text.split() if len(w) > 3]
                all_meta = collection.get(include=["metadatas"])["metadatas"]
                for meta in all_meta:
                    if any(w in preprocess(meta["question"]) for w in words):
                        best_answer = meta["answer"]
                        source = "keyword"
                        stats["keyword"] += 1
                        keyword_q = meta["question"].split("\n")[0][:80]
                        logger.info(f"КЛЮЧЕВОЙ ПОИСК ✓ | user={user.id} ({display_name}) | "
                                    f"запрос=\"{raw_text[:100]}{'...' if len(raw_text)>100 else ''}\" | → \"{keyword_q}\"")
                        break

        except Exception as e:
            logger.error(f"Chroma ошибка: {e}", exc_info=True)

    # Если вообще ничего не нашли — молча выходим
    if not best_answer:
        return

    reply = best_answer

    # Улучшаем через Groq, если ответ короткий
    if source != "fallback" and len(best_answer) < 1000:
        prompt = f"""Используй текст полностью не сокращая и не удаляя ссылки в сообщении, текст должен быть локаничным и дружелюбным. Сохрани весь смысл.
Оригинал:
{best_answer}
Вопрос: {raw_text}
Ответ:"""
        async with GROQ_SEM:
            stats["groq"] += 1
            save_stats()
            try:
                resp = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=500,
                        temperature=0.2,
                    ),
                    timeout=20
                )
                new = resp.choices[0].message.content.strip()
                if 15 < len(new) < len(best_answer) * 2:
                    reply = new
            except Exception as e:
                logger.warning(f"Groq упал: {e}")

    response_cache[cache_key] = reply
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply)

# ====================== БЛОКИРОВКА ЛИЧКИ ======================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Пауза тоже работает здесь
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

    # ПРАВИЛЬНЫЙ ПОРЯДОК ХЕНДЛЕРОВ
   # app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
   # app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_message))

   # app.add_handler(MessageHandler(
   #     filters.ChatType.PRIVATE & ~filters.COMMAND & ~filters.User(user_id=ADMIN_IDS),
   #     block_private
   # ))

    # Блокировка лички для всех, кроме админов
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & ~filters.COMMAND & ~filters.User(user_id=ADMIN_IDS),
        block_private
    ))
    # Основная обработка — только группы + админы в личке
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
    #app.job_queue.run_repeating(lambda _: asyncio.create_task(update_vector_db()), interval=600, first=600)

    logger.info("Бот запущен — пауза работает, Alt+Enter поддерживается, всё идеально!")

    app.run_polling(drop_pending_updates=True)








