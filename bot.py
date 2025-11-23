import os
import re
import json
import logging
import asyncio
# Ограничитель параллельных запросов к Groq
GROQ_SEM = asyncio.Semaphore(2)
VECTOR_THRESHOLD = 0.7   # порог для векторного поиска
# MAX_LEN = 4000           # лимит длины сообщения для Telegram
from hashlib import md5
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
import chromadb
from groq import AsyncGroq

# ====================== LOGGING ======================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# ====================== CONFIG ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_ID", "").split(",") if x]

# ====================== GOOGLE SHEETS ======================
creds = Credentials.from_service_account_file(
    os.getenv("GOOGLE_CREDENTIALS", "/app/service_account.json"),
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)
sheet = build("sheets", "v4", credentials=creds).spreadsheets()

# ====================== CHROMA ======================
CHROMA_DIR = "/app/chroma"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

collection_general = chroma_client.get_or_create_collection("general_kb")
collection_technical = chroma_client.get_or_create_collection("technical_kb")

# ====================== EMBEDDERS ======================
# Используем кэш моделей в persistent volume
import os
os.environ['TRANSFORMERS_CACHE'] = '/app/models_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/models_cache'

#embedder_general = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embedder_general = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
embedder_technical = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ====================== GROQ ======================
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# ====================== PAUSE & STATS ======================
PAUSE_FILE = "/app/data/paused.flag"
STATS_FILE = "/app/data/stats.json"
ADMINLIST_FILE = "/app/data/adminlist.json"

def is_paused() -> bool:
    return os.path.exists(PAUSE_FILE)

def set_paused(state: bool):
    if state:
        open(PAUSE_FILE, "w").close()
        logger.info("БОТ НА ПАУЗЕ — отвечает только админам")
    else:
        try:
            os.remove(PAUSE_FILE)
        except FileNotFoundError:
            pass
        logger.info("Пауза снята")

# ====================== ADMIN LIST ======================
adminlist = set()

def load_adminlist():
    global adminlist
    try:
        if os.path.exists(ADMINLIST_FILE):
            with open(ADMINLIST_FILE, "r") as f:
                adminlist = set(json.load(f))
                logger.info(f"Список администраторов загружен: {len(adminlist)} пользователей")
        else:
            adminlist = set()
    except Exception as e:
        logger.error(f"Ошибка загрузки списка администраторов: {e}")
        adminlist = set()

def save_adminlist():
    try:
        with open(ADMINLIST_FILE, "w") as f:
            json.dump(list(adminlist), f)
    except Exception as e:
        logger.error(f"Ошибка сохранения списка администраторов: {e}")

def is_admin_special(user_id: int) -> bool:
    return user_id in adminlist

def add_admin(user_id: int):
    adminlist.add(user_id)
    save_adminlist()
    logger.info(f"Пользователь {user_id} добавлен в список администраторов")

def remove_admin(user_id: int):
    adminlist.discard(user_id)
    save_adminlist()
    logger.info(f"Пользователь {user_id} удалён из списка администраторов")

stats = {"total": 0, "cached": 0, "groq": 0, "vector": 0, "keyword": 0}

def save_stats():
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"Ошибка сохранения статистики: {e}")

# ====================== CACHE ======================
response_cache = TTLCache(maxsize=1000, ttl=3600)

def preprocess(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^а-яa-z0-9\s]', ' ', text.lower())).strip()

async def safe_typing(bot, chat_id):
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except:
        pass
# ====================== ОБНОВЛЕНИЕ БАЗЫ ======================
async def update_vector_db(context: ContextTypes.DEFAULT_TYPE = None):
    global collection_general, collection_technical
    try:
        logger.info("Обновление базы знаний из Google Sheets...")

        # читаем данные из таблицы
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="General!A:B").execute()
        general_rows = result.get("values", [])
        logger.info(f"General rows загружено: {len(general_rows)}")
        if general_rows:
            logger.info(f"Пример General: {general_rows[0]}")
        
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Technical!A:B").execute()
        technical_rows = result.get("values", [])
        logger.info(f"Technical rows загружено: {len(technical_rows)}")
        if technical_rows:
            logger.info(f"Пример Technical: {technical_rows[0]}")
        
        # пересоздаём коллекции
        try:
            chroma_client.delete_collection("general_kb")
        except:
            pass
        try:
            chroma_client.delete_collection("technical_kb")
        except:
            pass

        collection_general = chroma_client.create_collection("general_kb")
        collection_technical = chroma_client.create_collection("technical_kb")

        # добавляем данные с ключами и ответами
        if general_rows:
            keys = [row[0] for row in general_rows if len(row) > 0]
            answers = [row[1] for row in general_rows if len(row) > 1]
            collection_general.add(
                ids=[f"general_{i}" for i in range(len(keys))],
                documents=keys,
                metadatas=[{"answer": ans} for ans in answers],
                embeddings=embedder_general.encode(keys).tolist()
            )

        if technical_rows:
            keys = [row[0] for row in technical_rows if len(row) > 0]
            answers = [row[1] for row in technical_rows if len(row) > 1]
            collection_technical.add(
                ids=[f"technical_{i}" for i in range(len(keys))],
                documents=keys,
                metadatas=[{"answer": ans} for ans in answers],
                embeddings=embedder_technical.encode(keys).tolist()
            )

        logger.info(f"База обновлена: общая={len(general_rows)}, тех={len(technical_rows)}")

    except Exception as e:
        logger.error(f"Ошибка загрузки базы: {e}", exc_info=True)


# ====================== MESSAGE HANDLER ======================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # проверяем список администраторов
    if is_admin_special(user_id):
        logger.info(f"Пользователь {user_id} в списке администраторов — игнорируем")
        return
    
    if is_paused() and user_id not in ADMIN_IDS:
        return

    raw_text = (update.message.text or update.message.caption or "").strip()
    if not raw_text or raw_text.startswith("/") or len(raw_text) > 1500:
        return

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

    # === Google Sheets поиск ===
    try:
        values = []
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="General!A:B").execute()
        values += result.get("values", [])
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Technical!A:B").execute()
        values += result.get("values", [])

        for row in values:
            if len(row) >= 2:
                keyword, answer = row[0].strip().lower(), row[1].strip()
                if keyword in clean_text or clean_text in keyword:
                    best_answer = answer
                    source = "keyword"
                    stats["keyword"] += 1
                    break
    except Exception as e:
        logger.error(f"Ошибка Google Sheets: {e}", exc_info=True)

    # === Векторный поиск (general) ===
    if not best_answer and collection_general and collection_general.count() > 0:
        try:
            emb = embedder_general.encode(clean_text).tolist()
            results = collection_general.query(
                query_embeddings=[emb],
                n_results=10,
                include=["metadatas", "distances"]
            )

            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            top_log = []
            for d, m in zip(distances, metadatas):
                preview = (m.get("answer") or "").replace("\n", " ")[:60]
                top_log.append(f"{d:.3f}→{preview}")

            selected_dist = None
            selected_preview = None

            for d, m in zip(distances, metadatas):
                if d < VECTOR_THRESHOLD and best_answer is None:
                    best_answer = m.get("answer")
                    source = "vector"
                    stats["vector"] += 1
                    selected_dist = d
                    selected_preview = (best_answer or "").replace("\n", " ")[:280]

            if best_answer:
                logger.info(
                    f"ВЕКТОР ✓ | distance={selected_dist:.4f} | user={user.id} ({display_name}) | "
                    f"запрос=\"{raw_text[:100]}{'...' if len(raw_text)>100 else ''}\" | "
                    f"→ \"{selected_preview}\" | топ-3: {' | '.join(top_log[:3])}"
                )
            else:
                best_dist = distances[0] if distances else 1.0
                best_q = (metadatas[0].get('answer') or '—').split("\n")[0][:280] if metadatas else "—"
                logger.info(
                    f"ВЕКТОР ✗ (порог >0.7) | лучший distance={best_dist:.4f} → \"{best_q}\" | "
                    f"user={user.id} ({display_name}) | запрос=\"{raw_text[:100]}{'...' if len(raw_text)>100 else ''}\" | "
                    f"топ-5: {' | '.join(top_log[:5])}"
                )
        except Exception as e:
            logger.error(f"Chroma ошибка: {e}", exc_info=True)

    # === Векторный поиск (technical) ===
    if not best_answer and collection_technical and collection_technical.count() > 0:
        try:
            emb = embedder_technical.encode(clean_text).tolist()
            results = collection_technical.query(
                query_embeddings=[emb],
                n_results=10,
                include=["metadatas", "distances"]
            )

            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            top_log = []
            for d, m in zip(distances, metadatas):
                preview = (m.get("answer") or "").replace("\n", " ")[:60]
                top_log.append(f"{d:.3f}→{preview}")

            selected_dist = None
            selected_preview = None

            for d, m in zip(distances, metadatas):
                if d < VECTOR_THRESHOLD and best_answer is None:
                    best_answer = m.get("answer")
                    source = "vector"
                    stats["vector"] += 1
                    selected_dist = d
                    selected_preview = (best_answer or "").replace("\n", " ")[:280]

            if best_answer:
                logger.info(
                    f"ВЕКТОР (TECH) ✓ | distance={selected_dist:.4f} | user={user.id} ({display_name}) | "
                    f"запрос=\"{raw_text[:100]}{'...' if len(raw_text)>100 else ''}\" | "
                    f"→ \"{selected_preview}\" | топ-3: {' | '.join(top_log[:3])}"
                )
            else:
                best_dist = distances[0] if distances else 1.0
                best_q = (metadatas[0].get('answer') or '—').split("\n")[0][:280] if metadatas else "—"
                logger.info(
                    f"ВЕКТОР (TECH) ✗ (порог >0.7) | лучший distance={best_dist:.4f} → \"{best_q}\" | "
                    f"user={user.id} ({display_name}) | запрос=\"{raw_text[:100]}{'...' if len(raw_text)>100 else ''}\" | "
                    f"топ-5: {' | '.join(top_log[:5])}"
                )
        except Exception as e:
            logger.error(f"Chroma ошибка (technical): {e}", exc_info=True)

    # === Fallback через Groq (с промтом, молчание при отсутствии данных) ===
    if not best_answer:
        try:
            system_prompt = (
                "Ты помощник службы поддержки. Отвечай коротко, по делу и только по фактам. Ответ должен быть понятным для технически не подкованным читателям\n\n"
                "Правила:\n"
                "1) Не придумывай. Если недостаточно данных — не отвечай.\n"
                "2) Сохраняй все ссылки и технические обозначения как есть.\n"
                "3) Не добавляй предположений, историй, аналогий и лишних пояснений.\n"
                "4) Формат: либо до 3 кратких предложений, либо до 5 маркеров.\n"
                "5) Длина: не длиннее исходного ответа и не более 800 символов."
            )

            user_prompt = f"Вопрос: {raw_text}\n\nОтвет:"

            stats["groq"] += 1
            save_stats()
            completion = await groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.0,
                top_p=0.1,
            )
            candidate = completion.choices[0].message.content.strip()

            # Если модель вернула пусто или "не знаю" → молчим
            if not candidate or candidate.lower().startswith("не знаю"):
                best_answer = None
                logger.info(
                    f"Groq fallback ✗ | user={user.id} ({display_name}) | модель промолчала"
                )
            else:
                best_answer = candidate
                source = "groq"
                logger.info(
                    f"Groq fallback ✓ | user={user.id} ({display_name}) | "
                    f"ответ={len(best_answer)} симв."
                )
        except Exception as e:
            logger.error(f"Groq ошибка: {e}", exc_info=True)
            # резервный ответ — ближайший из базы (даже если выше порога)
            if 'metadatas' in locals() and metadatas:
                best_answer = metadatas[0].get("answer")
                reply = f"⚠️ Groq недоступен. Похожий ответ из базы:\n\n{best_answer}"
                source = "vector-fallback"
                logger.info(
                    f"Groq недоступен → используем vector-fallback | user={user.id} ({display_name}) | "
                    f"ответ={len(best_answer)} симв."
                )
            else:
                best_answer = None
                reply = "Извините, я сейчас не могу найти ответ. Попробуйте переформулировать вопрос или обратитесь в поддержку."
                source = "none"

    # === Улучшаем через Groq, если ответ короткий ===
    reply = best_answer
    if source != "fallback" and best_answer and len(best_answer) < 1200:
        system_prompt = (
            "Ты помощник службы поддержки. Отвечай точно как в базе сильно не сокращая и не удаляя ссылки,но по делу и только по фактам.\n\n"
            "Правила:\n"
            "1) Не придумывай. Если недостаточно данных — ответ: \"Не знаю\".\n"
            "2) Сохраняй все ссылки и технические обозначения как есть.\n"
            "3) Не добавляй предположений, историй, аналогий и лишних пояснений.\n"
            "4) Формат: либо до 3 кратких предложений, либо до 5 маркеров.\n"
            "5) Длина: не длиннее исходного ответа и не более 800 символов.\n"
            "6) Если вопрос не относится к базе — \"Не знаю\".\n\n"
            "Твоя задача: переформулировать исходный ответ, сделав его более понятнее для технически не подкованных читателей и точнее без потери смысла."
        )

        prompt_user = f"Оригинал:\n{best_answer}\n\nВопрос: {raw_text}\n\nОтвет:"

        async with GROQ_SEM:
            stats["groq"] += 1
            save_stats()
            try:
                resp = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_user},
                        ],
                        max_tokens=400,
                        temperature=0.0,
                        top_p=0.1,
                    ),
                    timeout=20
                )
                new = resp.choices[0].message.content.strip()
                if 30 < len(new) <= 800 and len(new) <= len(best_answer):
                    reply = new
                    logger.info(
                        f"Groq улучшил ответ | user={user.id} ({display_name}) | "
                        f"старый={len(best_answer)} симв. → новый={len(new)} симв."
                    )
            except Exception as e:
                logger.warning(f"Groq упал: {e}")

    # === Финальная отправка ответа ===
    reply = reply or best_answer or "Извините, я не смог найти ответ."
    response_cache[cache_key] = reply
    logger.info(f"ОТПРАВКА → user={user.id} ({display_name}) | {reply[:100]}{'...' if len(reply)>100 else ''}")
    try:
        MAX_LEN = 4000
        for i in range(0, len(reply), MAX_LEN):
            await context.bot.send_message(chat_id=update.effective_chat.id, text=reply[i:i+MAX_LEN])
    except telegram.error.TimedOut:
        logger.warning("Отправка превысила таймаут, пробуем ещё раз...")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=reply[:4000])
    except Exception as e:
        logger.error(f"Ошибка отправки: {e}", exc_info=True)
        
# ====================== BLOCK PRIVATE ======================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Эта функция срабатывает только для неадминов в личных чатах
    if is_paused():
        return

    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Связаться с поддержкой", url="https://t.me/alexeymaloi")]]
    )
    await update.message.reply_text(
        "Писать боту в личку могут только администраторы.\nНужна помощь — нажми ниже:",
        reply_markup=keyboard
    )

# ====================== АДМИН-КОМАНДЫ ======================
async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    await update_vector_db()
    await update.message.reply_text("База перезагружена!")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    set_paused(True)
    await update.message.reply_text("Бот на паузе — обычные пользователи не получают ответы")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    set_paused(False)
    await update.message.reply_text("Бот снова работает")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    paused = "Пауза" if is_paused() else "Работает"
    count_general = collection_general.count() if collection_general else 0
    count_technical = collection_technical.count() if collection_technical else 0
    await update.message.reply_text(
        f"Статус: {paused}\n"
        f"Записей: общая={count_general}, тех={count_technical}\n"
        f"Запросов: {stats['total']} (кэш: {stats['cached']})\n"
        f"Вектор: {stats['vector']} | Ключи: {stats['keyword']} | Groq: {stats['groq']}"
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Глобальная ошибка: {context.error}", exc_info=True)

async def clear_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    response_cache.clear()
    stats["cached"] = 0  # можно обнулить счётчик кэша
    save_stats()
    await update.message.reply_text("Кэш очищен!")

async def add_admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /addadmin <user_id>")
        return
    
    user_id = int(context.args[0])
    add_admin(user_id)
    await update.message.reply_text(f"✅ Пользователь {user_id} добавлен в список администраторов")

async def remove_admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /removeadmin <user_id>")
        return
    
    user_id = int(context.args[0])
    remove_admin(user_id)
    await update.message.reply_text(f"✅ Пользователь {user_id} удален из списка администраторов")

async def adminlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    if not adminlist:
        await update.message.reply_text("Список администраторов пуст")
        return
    
    admin_users = "\n".join([str(uid) for uid in sorted(adminlist)])
    await update.message.reply_text(f"👨‍💼 Администраторы ({len(adminlist)}):\n{admin_users}")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        return
    commands_text = (
        "📌 Доступные команды для админов:\n\n"
        "/reload – перезагрузить базу\n"
        "/pause – поставить бота на паузу\n"
        "/resume – возобновить работу бота\n"
        "/status – показать статус и статистику\n"
        "/clearcache – очистить кэш ответов\n"
        "/addadmin <user_id> – добавить администратора\n"
        "/removeadmin <user_id> – удалить администратора\n"
        "/adminlist – показать список администраторов\n"
        "/help – показать это меню\n"
    )
    await update.message.reply_text(commands_text)



# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    app = Application.builder()\
        .token(TELEGRAM_TOKEN)\
        .concurrent_updates(False)\
        .build()

    # блокируем личные чаты для не-админов
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & ~filters.COMMAND & ~filters.User(user_id=ADMIN_IDS),
        block_private
    ))

    # обработка сообщений в группах и от админов (включая личные чаты админов)
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND &
        ((filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP | filters.ChatType.PRIVATE) & filters.User(user_id=ADMIN_IDS)) |
        (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP),
        handle_message
    ))
    app.add_handler(MessageHandler(
        filters.CAPTION & ~filters.COMMAND &
        ((filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP | filters.ChatType.PRIVATE) & filters.User(user_id=ADMIN_IDS)) |
        (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP),
        handle_message
    ))

    # команды админов
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("clearcache", clear_cache))
    app.add_handler(CommandHandler("addadmin", add_admin_cmd))
    app.add_handler(CommandHandler("removeadmin", remove_admin_cmd))
    app.add_handler(CommandHandler("adminlist", adminlist_cmd))
    app.add_handler(CommandHandler("help", help_cmd))


    app.add_error_handler(error_handler)

    # загрузка списка администраторов
    load_adminlist()

    # первая загрузка базы через 15 секунд после старта
    app.job_queue.run_once(update_vector_db, when=15)

    logger.info("4.1 Добавлены Админы Групп, Админы бота продолжают получать сообщения в ЛС бота")

    app.run_polling(drop_pending_updates=True)
