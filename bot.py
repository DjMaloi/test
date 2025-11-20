import os
import json
import logging
import asyncio
from functools import lru_cache
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
from telegram.error import BadRequest
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# Отключаем телеметрию ChromaDB
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# ============================ ЛОГИРОВАНИЕ ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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
        ADMIN_IDS = [int(uid.strip()) for uid in ADMIN_ID_STR.split(",") if uid.strip()]
        logger.info(f"Админы: {ADMIN_IDS}")
    except ValueError:
        errors.append("ADMIN_ID — некорректный формат")

if errors:
    logger.error("ОШИБКИ ЗАПУСКА:\n" + "\n".join(f"→ {e}" for e in errors))
    exit(1)

# ============================ ФАЙЛ-ФЛАГ ПАУЗЫ (РАБОТАЕТ В DOCKER/COOLIFY) ============================
PAUSE_FILE = "/app/paused.flag"   # путь внутри контейнера

def is_paused() -> bool:
    return os.path.exists(PAUSE_FILE)

def set_paused(state: bool):
    if state:
        open(PAUSE_FILE, "w").close()
        logger.info("Бот поставлен на паузу (файл-флаг создан)")
    else:
        try:
            os.remove(PAUSE_FILE)
            logger.info("Пауза снята (файл-флаг удалён)")
        except FileNotFoundError:
            pass

# Поддержка паузы при старте через переменную окружения (удобно в Coolify)
if os.getenv("BOT_PAUSED", "").lower() == "true":
    set_paused(True)
    logger.info("Бот запущен в режиме паузы (BOT_PAUSED=true)")

# ============================ GOOGLE SHEETS ============================
try:
    with open(GOOGLE_CREDENTIALS_PATH) as f:
        creds_info = json.load(f)
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    logger.info("Google Sheets подключён")
except Exception as e:
    logger.error(f"Ошибка Google Auth: {e}")
    exit(1)

# ============================ GROQ ============================
client = Groq(api_key=GROQ_API_KEY)

# ============================ ВЕКТОРНАЯ БАЗА ============================
chroma_client = chromadb.Client()
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ============================ ЗАГРУЗКА БАЗЫ ЗНАНИЙ ============================
@lru_cache(maxsize=1)
def get_knowledge_base() -> str:
    try:
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Support!A:B").execute()
        values = result.get("values", [])
        if not values:
            return ""
        rows = values[1:] if len(values) > 0 and ("проблема" in str(values[0][0]).lower() or "keyword" in str(values[0][0]).lower()) else values
        entries = []
        for row in rows:
            if len(row) >= 2:
                problem = row[0].strip()
                solution = row[1].strip()
                if problem:
                    entries.append(f"Проблема: {problem}\nРешение: {solution}")
        kb = "\n\n".join(entries)
        logger.info(f"Загружено {len(entries)} записей из Google Sheets")
        return kb
    except Exception as e:
        logger.error(f"Ошибка чтения таблицы: {e}")
        return ""

# ============================ ОБНОВЛЕНИЕ ВЕКТОРНОЙ БАЗЫ ============================
async def update_vector_db():
    kb_text = get_knowledge_base()
    if not kb_text:
        logger.warning("База знаний пуста")
        return
    blocks = [b.strip() for b in kb_text.split("\n\n") if b.strip()]
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
    except Exception:
        pass
    if docs:
        collection = chroma_client.get_or_create_collection(name="support_kb")
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.info(f"Векторная база обновлена: {len(docs)} записей")

# ============================ ОБРАБОТКА СООБЩЕНИЙ ============================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Проверка паузы — работает во всех воркерах и после рестарта
    if is_paused() and update.effective_user.id not in ADMIN_IDS:
        return

    text = (update.message.text or update.message.caption or "").strip()
    if not text or text.startswith("/") or len(text) > 1500:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    logger.info(f"Запрос от {user_id} в чате {chat_id}: {text[:80]}...")

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    await asyncio.sleep(2 + len(text.split()) / 10)

    try:
        collection = chroma_client.get_collection(name="support_kb")
        query_emb = embedder.encode(text).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=6, include=["metadatas"])
        relevant = results["metadatas"][0] if results["metadatas"] else []

        if not relevant:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Точного решения пока нет в базе знаний.\nОпишите подробнее — передам специалисту."
            )
            return

        context_str = "\n\n".join([f"Проблема: {m['problem']}\nРешение: {m['solution']}" for m in relevant])

        prompt = f"""Ты — опытный русскоязычный специалист техподдержки.
Отвечай ТОЛЬКО на основе информации ниже. Ничего не придумывай.
База знаний:
{context_str}
Правила:
- Кратко, по-русски, по делу, используй списки.
- Пиши как живой человек.
Вопрос: {text}
Ответ:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        reply = response.choices[0].message.content.strip()

        if any(w in reply.lower() for w in ["не знаю", "не уверен", "не могу найти"]):
            reply = "Точного решения пока нет в базе знаний.\nПередам ваш запрос — ответим скоро."

        if len(reply) > 900:
            for part in reply.split("\n\n"):
                if part.strip():
                    await context.bot.send_message(chat_id=chat_id, text=part.strip())
                    await asyncio.sleep(0.7)
            await asyncio.sleep(1)
            await context.bot.send_message(chat_id=chat_id, text="Если не помогло — уточните, что происходит сейчас.")
        else:
            await context.bot.send_message(chat_id=chat_id, text=reply)
            if hash(text) % 10 < 7:
                await asyncio.sleep(1.5)
                await context.bot.send_message(chat_id=chat_id, text="Если проблема осталась — напишите подробнее, посмотрю ещё раз.")

    except Exception as e:
        logger.error(f"Ошибка в handle_message (чат {chat_id}): {e}", exc_info=True)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Сервис временно недоступен. Попробуйте через пару минут."
            )
        except Exception as te:
            logger.error(f"Не удалось отправить fallback: {te}")

# ============================ БЛОКИРОВКА ЛИЧКИ ============================
async def block_non_admin_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private":
        return
    if update.effective_user.id in ADMIN_IDS:
        return
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
    await update.message.reply_text(
        "Писать боту в личку могут только администраторы.\nНужна помощь — нажми кнопку ниже:",
        reply_markup=keyboard
    )

# ============================ АДМИН-КОМАНДЫ ============================
async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    get_knowledge_base.cache_clear()
    await update_vector_db()
    await update.message.reply_text("База знаний перезагружена!")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    set_paused(True)
    await update.message.reply_text("Бот поставлен на паузу\nОбычные пользователи больше не получают автоответы")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    set_paused(False)
    await update.message.reply_text("Пауза снята — бот снова работает для всех")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    status = "На паузе" if is_paused() else "Работает нормально"
    await update.message.reply_text(f"Статус бота: {status}")

async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("PAUSED" if is_paused() else "OK")

# ============================ ЗАПУСК ============================
if __name__ == "__main__":
    app = Application.builder() \
        .token(TELEGRAM_TOKEN) \
        .request(HTTPXRequest(connection_pool_size=100)) \
        .concurrent_updates(False) \   # Один процесс — пауза срабатывает мгновенно и надёжно
        .build()

    # Порядок важен!
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_non_admin_private))

    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("health", health_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.CAPTION & ~filters.COMMAND, handle_message))

    # Обновление базы знаний
    app.job_queue.run_once(lambda _: asyncio.create_task(update_vector_db()), when=3)
    app.job_queue.run_repeating(lambda _: asyncio.create_task(update_vector_db()), interval=600, first=600)

    logger.info("Бот запущен и готов к работе!")
    app.run_polling(drop_pending_updates=True)