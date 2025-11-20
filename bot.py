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
PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

errors = []
for var, name in [
    (TELEGRAM_TOKEN, "TELEGRAM_TOKEN"),
    (GROQ_API_KEY, "GROQ_API_KEY"),
    (SHEET_ID, "SHEET_ID"),
]:
    if not var or not var.strip():
        errors.append(f"{name} не задан")

if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    errors.append(f"Файл credentials не найден: {GOOGLE_CREDENTIALS_PATH}")

if not ADMIN_ID_STR.strip():
    errors.append("ADMIN_ID не задан")
else:
    try:
        ADMIN_IDS = [int(uid.strip()) for uid in ADMIN_ID_STR.split(",") if uid.strip()]
    except ValueError:
        errors.append("ADMIN_ID — некорректный формат")

if errors:
    logger.error("ОШИБКИ ЗАПУСКА:\n" + "\n".join(f"→ {e}" for e in errors))
    exit(1)

logger.info("Все переменные загружены успешно")

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
            logger.warning("Таблица пуста")
            return ""

        # Пропускаем заголовок
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
        logger.warning("База знаний пуста — пропускаем обновление векторной базы")
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

    # УДАЛЯЕМ ВСЮ КОЛЛЕКЦИЮ ЦЕЛИКОМ
    try:
        chroma_client.delete_collection("support_kb")
        logger.info("Старая коллекция удалена")
    except Exception as e:
        logger.info(f"Коллекция не существовала (нормально при первом запуске): {e}")

    if docs:
        collection = chroma_client.get_or_create_collection(name="support_kb")
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.info(f"Векторная база успешно обновлена: {len(docs)} записей")
    else:
        logger.warning("Нет записей для добавления в векторную базу")

# ============================ ОБРАБОТКА СООБЩЕНИЙ ============================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if PAUSED and update.effective_user.id not in ADMIN_IDS:
        return

    text = (update.message.text or update.message.caption or "").strip()
    if not text or text.startswith("/") or len(text) > 1500:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    logger.info(f"Запрос от {user_id} в чате {chat_id}: {text[:80]}...")

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    await asyncio.sleep(2 + len(text.split()) / 10)  # 2–8 сек задержка

    # Семантический поиск
    try:
        collection = chroma_client.get_collection(name="support_kb")
        query_emb = embedder.encode(text).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=6, include=["metadatas"])

        relevant = results["metadatas"][0] if results["metadatas"] else []
        if not relevant:
            await context.bot.send_message(chat_id=chat_id, text="Точного решения по вашему вопросу пока нет в базе знаний.\nОпишите проблему подробнее — передам специалисту.")
            return

        context_blocks = [f"Проблема: {m['problem']}\nРешение: {m['solution']}" for m in relevant]
        context = "\n\n".join(context_blocks)

        prompt = f"""Ты — опытный русскоязычный специалист техподдержки.
Отвечай ТОЛЬКО на основе информации ниже. Ничего не придумывай.

База знаний:
{context}

Правила:
- Если точного ответа нет — скажи: «Точного решения пока нет в базе знаний. Передам коллеге.»
- Кратко, по-русски, по делу, используй списки.
- Пиши как живой человек, без смайликов и лишних восклицаний.

Вопрос: {text}
Ответ:"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ← ФИКС: Новая модель вместо устаревшей
            messages=[{"role": "system", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        reply = response.choices[0].message.content.strip()

        # Защита от редких галлюцинаций
        if any(w in reply.lower() for w in ["не знаю", "не уверен", "не могу найти"]):
            reply = "Точного решения пока нет в базе знаний.\nПередам ваш запрос — ответим скоро."

        if len(reply) > 900:
            for part in reply.split("\n\n"):
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
        logger.error(f"Ошибка в handle_message (чат {chat_id}, юзер {user_id}): {e}")
        try:
            # Fallback: используем send_message вместо reply_text (реже падает на правах)
            await context.bot.send_message(
                chat_id=chat_id,
                text="Сервис временно недоступен. Попробуйте через пару минут."
            )
        except BadRequest as te:
            logger.error(f"Telegram BadRequest в чате {chat_id}: {te} — проверьте права бота в чате!")
        except Exception as te:
            logger.error(f"Дополнительная ошибка Telegram: {te}")

# ============================ БЛОКИРОВКА ЛИЧКИ И АДМИНКИ ============================
async def block_non_admin_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private" or update.effective_user.id in ADMIN_IDS:
        return
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]])
    try:
        await update.message.reply_text(
            "Писать боту в личку могут только администраторы.\nНужна помощь — нажми кнопку:",
            reply_markup=keyboard
        )
    except BadRequest:
        logger.warning("Не удалось ответить в приватке — права?")

async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != "private" or update.effective_user.id not in ADMIN_IDS:
        return
    get_knowledge_base.cache_clear()
    await update_vector_db()
    await update.message.reply_text("База знаний и векторная база перезагружены!")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    global PAUSED
    PAUSED = True
    await update.message.reply_text("Бот на паузе")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS: return
    global PAUSED
    PAUSED = False
    await update.message.reply_text("Бот снова работает")

# ============================ ЗАПУСК ============================
if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).request(HTTPXRequest(connection_pool_size=100)).build()

    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, block_non_admin_private))
    app.add_handler(MessageHandler((filters.TEXT | filters.CAPTION) & ~filters.COMMAND, handle_message))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))

    # Первая загрузка через 3 сек после старта + каждые 10 минут
    app.job_queue.run_once(lambda _: asyncio.create_task(update_vector_db()), when=3)
    app.job_queue.run_repeating(lambda _: asyncio.create_task(update_vector_db()), interval=600, first=600)

    logger.info("Бот запущен и готов к работе!")
    app.run_polling(drop_pending_updates=True)