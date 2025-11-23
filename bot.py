import os
import re
import json
import logging
import asyncio
from hashlib import md5
from typing import Optional, Tuple, List
from contextlib import asynccontextmanager

# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.error import TimedOut, NetworkError, RetryAfter

# Google Sheets
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# ML/AI
from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
import chromadb
from groq import AsyncGroq

# ====================== КОНСТАНТЫ ======================
GROQ_SEM = asyncio.Semaphore(3)  # Увеличено с 2 до 3
VECTOR_THRESHOLD = 0.65  # Понижен с 0.7 до 0.65 для лучшего покрытия
MAX_MESSAGE_LENGTH = 4000
CACHE_SIZE = 2000  # Увеличен с 1000
CACHE_TTL = 7200  # Увеличен с 3600 (2 часа)

CRITICAL_MISMATCHES = {
    "касса": ["киоск", "КСО", "сканер", "принтер чеков", "терминал самообслуживания"],
    "киоск": ["касса", "онлайн-касса", "фискальный регистратор", "терминал оплаты"],
}

def is_mismatch(question: str, answer: str) -> bool:
    """
    Проверяет, не противоречит ли ответ вопросу (например, вопрос про кассу, а ответ — про киоск)
    """
    question_lower = question.lower()
    answer_lower = answer.lower()

    for expected, forbidden in CRITICAL_MISMATCHES.items():
        if expected in question_lower:
            for word in forbidden:
                if word in answer_lower:
                    return True
    return False
# ====================== LOGGING ======================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Уменьшаем шум от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# ====================== CONFIG ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_ID", "").split(",") if x]

# Валидация конфигурации
if not all([TELEGRAM_TOKEN, GROQ_API_KEY, SHEET_ID]):
    raise ValueError("Отсутствуют обязательные переменные окружения!")

# ====================== GOOGLE SHEETS ======================
creds = Credentials.from_service_account_file(
    os.getenv("GOOGLE_CREDENTIALS", "/app/service_account.json"),
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)
sheet = build("sheets", "v4", credentials=creds).spreadsheets()

# ====================== CHROMA ======================
CHROMA_DIR = "/app/chroma"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Блокировка для безопасного обновления коллекций
collection_lock = asyncio.Lock()
collection_general = None
collection_technical = None

# ====================== EMBEDDERS ======================
os.environ['TRANSFORMERS_CACHE'] = '/app/models_cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/models_cache'

logger.info("Загрузка моделей эмбеддингов...")
embedder_general = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
embedder_technical = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("✓ Модели загружены")

# ====================== GROQ ======================
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# ====================== ФАЙЛЫ СОСТОЯНИЯ ======================
PAUSE_FILE = "/app/data/paused.flag"
STATS_FILE = "/app/data/stats.json"
ADMINLIST_FILE = "/app/data/adminlist.json"
ALARM_FILE = "/app/data/alarm.txt"


# ====================== ФУНКЦИИ ПАУЗЫ ======================
def is_paused() -> bool:
    """Проверяет, находится ли бот на паузе"""
    return os.path.exists(PAUSE_FILE)

def set_paused(state: bool):
    """Устанавливает состояние паузы"""
    if state:
        open(PAUSE_FILE, "w").close()
        logger.info("🔴 БОТ НА ПАУЗЕ — отвечает только админам")
    else:
        try:
            os.remove(PAUSE_FILE)
            logger.info("🟢 Пауза снята, бот работает в обычном режиме")
        except FileNotFoundError:
            pass

# ====================== УПРАВЛЕНИЕ АДМИНАМИ ======================
current_alarm: Optional[str] = None  # Новое: глобальная переменная для хранения текущего alarm

adminlist = set()

def load_adminlist() -> set:
    """Загружает список админов из файла"""
    global adminlist
    try:
        logger.info(f"🔍 Ищу adminlist.json по пути: {ADMINLIST_FILE}")
        
        os.makedirs(os.path.dirname(ADMINLIST_FILE), exist_ok=True)
        
        with open(ADMINLIST_FILE, "r") as f:
            data = json.load(f)
            logger.info(f"📄 Прочитан файл: {data}")
        
        # ИЗМЕНЕНИЕ №1: поддержка формата {"admins": [...]}
        adminlist = {int(x) for x in data.get("admins", [])}
        logger.info(f"✅ Загружено {len(adminlist)} админов: {adminlist}")
        return adminlist
    
    except FileNotFoundError:
        logger.error(f"❌ Файл не найден: {ADMINLIST_FILE}")
        adminlist = set()
        save_adminlist()  # Создаём пустой файл
        return adminlist
    
    except json.JSONDecodeError as e:
        logger.error(f"❌ Ошибка парсинга JSON: {e}")
        adminlist = set()
        return adminlist
    
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}", exc_info=True)
        adminlist = set()
        return adminlist

def save_adminlist():
    """Сохраняет список администраторов в файл"""
    global adminlist
    try:
        os.makedirs(os.path.dirname(ADMINLIST_FILE), exist_ok=True)
        with open(ADMINLIST_FILE, "w") as f:
            # ИЗМЕНЕНИЕ №2: сохраняем в формате {"admins": [...]}
            json.dump({"admins": list(adminlist)}, f, indent=2)
        logger.info(f"💾 Сохранено {len(adminlist)} админов")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения adminlist: {e}")

def is_admin_special(user_id: int) -> bool:
    """Проверяет, является ли пользователь специальным администратором"""
    return user_id in adminlist

def add_admin(user_id: int):
    """Добавляет пользователя в список администраторов"""
    global adminlist
    adminlist.add(user_id)
    save_adminlist()
    logger.info(f"➕ Пользователь {user_id} добавлен в adminlist")

def remove_admin(user_id: int):
    """Удаляет пользователя из списка администраторов"""
    global adminlist
    adminlist.discard(user_id)
    save_adminlist()
    logger.info(f"➖ Пользователь {user_id} удалён из adminlist")

# ====================== ALARM СИСТЕМА ======================

def load_alarm() -> Optional[str]:
    """Загружает текст alarm из файла"""
    try:
        if os.path.exists(ALARM_FILE):
            with open(ALARM_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    logger.info(f"🔊 Загружен alarm: {content[:100]}{'...' if len(content) > 100 else ''}")
                    return content
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки alarm: {e}")
    return None

def save_alarm(text: str):
    """Сохраняет текст alarm в файл"""
    try:
        os.makedirs(os.path.dirname(ALARM_FILE), exist_ok=True)
        with open(ALARM_FILE, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"📢 Alarm сохранён: {text[:100]}{'...' if len(text) > 100 else ''}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения alarm: {e}")

def clear_alarm():
    """Удаляет файл alarm"""
    try:
        os.remove(ALARM_FILE)
        logger.info("🔇 Alarm удалён")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error(f"❌ Ошибка удаления alarm: {e}")


# ====================== СТАТИСТИКА ======================
stats = {
    "total": 0,
    "cached": 0,
    "groq": 0,
    "vector": 0,
    "keyword": 0,
    "errors": 0
}

def load_stats():
    """Загружает статистику из файла"""
    global stats
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                loaded = json.load(f)
                stats.update(loaded)
                logger.info(f"✓ Статистика загружена: {stats['total']} запросов")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки статистики: {e}")

def save_stats():
    """Сохраняет статистику в файл"""
    try:
        os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения статистики: {e}")

# ====================== КЭШИРОВАНИЕ ======================
response_cache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

def preprocess(text: str) -> str:
    """Нормализует текст для поиска и кэширования"""
    text = text.lower()
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def safe_typing(bot, chat_id):
    """Безопасно отправляет индикатор "печатает" """
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass  # Игнорируем ошибки индикатора

# ====================== ВЕКТОРНЫЙ ПОИСК ======================
async def search_in_collection(
    collection,
    embedder: SentenceTransformer,
    query: str,
    threshold: float = VECTOR_THRESHOLD,
    n_results: int = 10
) -> Tuple[Optional[str], float, List[str]]:
    """
    Универсальная функция векторного поиска
    
    Возвращает: (лучший_ответ, расстояние, топ_результаты_для_логов)
    """
    if not collection or collection.count() == 0:
        return None, 1.0, []
    
    try:
        # Генерируем эмбеддинг запроса
        emb = embedder.encode(query).tolist()
        
        # Выполняем поиск
        results = collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["metadatas", "distances"]
        )
        
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        
        # Формируем лог для отладки
        top_log = []
        for d, m in zip(distances, metadatas):
            preview = (m.get("answer") or "").replace("\n", " ")[:60]
            top_log.append(f"{d:.3f}→{preview}")
        
        # Ищем лучший результат ниже порога
        best_answer = None
        best_distance = 1.0
        
        if distances and distances[0] < threshold:
            best_answer = metadatas[0].get("answer")
            best_distance = distances[0]
        
        return best_answer, best_distance, top_log
        
    except Exception as e:
        logger.error(f"❌ Ошибка векторного поиска: {e}", exc_info=True)
        return None, 1.0, []

# ====================== GROQ API ======================
@asynccontextmanager
async def groq_with_timeout(timeout: int = 20):
    """Контекстный менеджер для запросов к Groq с таймаутом"""
    async with GROQ_SEM:
        stats["groq"] += 1
        save_stats()
        try:
            yield
        except asyncio.TimeoutError:
            logger.warning("⏱️ Groq API превысил таймаут")
            raise

async def improve_with_groq(original_answer: str, question: str) -> Optional[str]:
    """
    Улучшает ответ через Groq, делая его более понятным
    
    Возвращает улучшенный ответ или None при ошибке
    """
    system_prompt = (
        "Ты — помощник техподдержки. Твоя задача — упростить и переформулировать "
        "уже существующий ответ так, чтобы он был понятен обычному пользователю.\n\n"
        
        "ИНСТРУКЦИЯ:\n"
        "1. Упрощай язык, но НЕ теряй точность и технические детали.\n"
        "2. НИКОГДА не добавляй информацию, которой нет в исходном ответе.\n"
        "3. Сохраняй ВСЕ ссылки, ID, артикулы, коды и термины без изменений.\n"
        "4. Не заменяй термины: 'касса' ≠ 'киоск', 'КСО' ≠ 'терминал оплаты' — это разные вещи.\n"
        "5. Если не понимаешь — верни оригинальный ответ без изменений.\n"
        "6. Максимум 800 символов, не длиннее оригинала более чем на 20%.\n"
        "7. НЕ используй списки, markdown или форматирование.\n"
        "8. НЕ начинай с 'Конечно', 'Вот улучшенный ответ' и т.п. — только с сути.\n\n"
        
        "ОЧЕНЬ ВАЖНО: ТОЧНОСТЬ ТЕРМИНОЛОГИИ\n"
        "- 'Касса' — это терминал для приёма оплаты (онлайн-касса).\n"
        "- 'Киоск' — это устройство самообслуживания (КСО), может включать кассу, но не то же самое.\n"
        "- Никогда не подставляй одно вместо другого.\n\n"
        
        "ФОРМАТ ВЫВОДА:\n"
        "Один связный абзац, без вступлений — только улучшенный ответ."
    )
    
    user_prompt = f"Исходный ответ:\n{original_answer}\n\nВопрос: {question}\n\nУлучшенный ответ:"
    
    try:
        async with groq_with_timeout():
            resp = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=500,
                    temperature=0.0,
                    top_p=0.1,
                ),
                timeout=20
            )
            
            improved = resp.choices[0].message.content.strip()
            
            # Проверяем качество улучшения
            if 30 < len(improved) <= 800 and len(improved) <= len(original_answer) * 1.2:
                return improved
            
            return None
            
    except Exception as e:
        logger.warning(f"⚠️ Groq улучшение не удалось: {e}")
        return None

async def fallback_groq(question: str) -> Optional[str]:
    """
    Запрос к Groq когда ничего не найдено в базе
    
    Возвращает ответ или None если модель не знает ответа
    """
    system_prompt = (
        "Ты — помощник техподдержки. Отвечай ТОЛЬКО если уверен в ответе.\n\n"
        
        "СТРОГИЕ ПРАВИЛА:\n"
        "1. Если недостаточно данных — ответь: 'НЕТ ДАННЫХ'.\n"
        "2. НЕ придумывай, не угадывай, не интерпретируй.\n"
        "3. Сохраняй ВСЕ ссылки, ID и коды без изменений.\n"
        "4. Ответ — максимум 3 коротких предложения или 5 пунктов.\n"
        "5. Длина — до 800 символов.\n"
        "6. НЕ используй markdown, списки или форматирование.\n"
        "7. НЕ начинай с 'Конечно', 'Вот ответ' и т.п.\n\n"
        
        "ОЧЕНЬ ВАЖНО: ТОЧНОСТЬ ТЕРМИНОЛОГИИ\n"
        "- 'Касса' — это терминал для приёма оплаты (онлайн-касса, фискальный регистратор).\n"
        "- 'Киоск' — это устройство самообслуживания (КСО), может включать кассу, сканер, экран.\n"
        "- Эти понятия НЕ взаимозаменяемы. НЕ подставляй одно вместо другого.\n"
        "- Если вопрос про кассу — не отвечай про киоск, и наоборот.\n\n"
        
        "ФОРМАТ ВЫВОДА:\n"
        "Один абзац или краткий список — только суть."
    )


    
    user_prompt = f"Вопрос: {question}\n\nОтвет:"
    
    try:
        async with groq_with_timeout():
            completion = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.0,
                    top_p=0.1,
                ),
                timeout=15
            )
            
            answer = completion.choices[0].message.content.strip()
            
            # Проверяем, что модель не отказалась отвечать
            if not answer or answer.upper().startswith("НЕТ ДАННЫХ") or \
               answer.lower().startswith("не знаю") or len(answer) < 10:
                return None
            
            return answer
            
    except Exception as e:
        logger.error(f"❌ Groq fallback ошибка: {e}")
        return None

# ====================== ОБНОВЛЕНИЕ БАЗЫ ======================
async def update_vector_db(context: ContextTypes.DEFAULT_TYPE = None):
    """Обновляет векторную базу из Google Sheets"""
    global collection_general, collection_technical
    
    async with collection_lock:  # Защита от race condition
        try:
            logger.info("🔄 Начало обновления базы знаний...")
            
            # Читаем данные из Google Sheets
            result_general = sheet.values().get(
                spreadsheetId=SHEET_ID, 
                range="General!A:B"
            ).execute()
            general_rows = result_general.get("values", [])
            
            result_technical = sheet.values().get(
                spreadsheetId=SHEET_ID, 
                range="Technical!A:B"
            ).execute()
            technical_rows = result_technical.get("values", [])
            
            logger.info(f"📥 Загружено: General={len(general_rows)}, Technical={len(technical_rows)}")
            
            # Пересоздаём коллекции
            for name in ["general_kb", "technical_kb"]:
                try:
                    chroma_client.delete_collection(name)
                except Exception:
                    pass
            
            collection_general = chroma_client.create_collection("general_kb")
            collection_technical = chroma_client.create_collection("technical_kb")
            
            # Заполняем коллекцию General
            if general_rows:
                keys = [row[0] for row in general_rows if len(row) > 0]
                answers = [row[1] if len(row) > 1 else "" for row in general_rows]
                
                embeddings = embedder_general.encode(keys).tolist()
                
                collection_general.add(
                    ids=[f"general_{i}" for i in range(len(keys))],
                    documents=keys,
                    metadatas=[{"answer": ans} for ans in answers],
                    embeddings=embeddings
                )
            
            # Заполняем коллекцию Technical
            if technical_rows:
                keys = [row[0] for row in technical_rows if len(row) > 0]
                answers = [row[1] if len(row) > 1 else "" for row in technical_rows]
                
                embeddings = embedder_technical.encode(keys).tolist()
                
                collection_technical.add(
                    ids=[f"technical_{i}" for i in range(len(keys))],
                    documents=keys,
                    metadatas=[{"answer": ans} for ans in answers],
                    embeddings=embeddings
                )
            
            logger.info(f"✅ База обновлена успешно!")
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка обновления базы: {e}", exc_info=True)
            stats["errors"] += 1
            save_stats()

# ====================== ОТПРАВКА СООБЩЕНИЙ ======================
async def send_long_message(bot, chat_id: int, text: str, max_retries: int = 3):
    """
    Безопасно отправляет длинное сообщение с разбивкой и повторами
    """
    for attempt in range(max_retries):
        try:
            # Разбиваем на части если нужно
            for i in range(0, len(text), MAX_MESSAGE_LENGTH):
                chunk = text[i:i + MAX_MESSAGE_LENGTH]
                await bot.send_message(chat_id=chat_id, text=chunk)
            return True
            
        except RetryAfter as e:
            # Telegram просит подождать
            wait_time = e.retry_after + 1
            logger.warning(f"⏸️ Rate limit, ждём {wait_time}с...")
            await asyncio.sleep(wait_time)
            
        except TimedOut:
            logger.warning(f"⏱️ Таймаут отправки (попытка {attempt + 1}/{max_retries})")
            await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка
            
        except NetworkError as e:
            logger.error(f"🌐 Сетевая ошибка: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(3)
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки: {e}", exc_info=True)
            return False
    
    return False



# ====================== ОСНОВНОЙ ОБРАБОТЧИК ======================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главная функция обработки сообщений"""
    user_id = update.effective_user.id
    chat_type = update.effective_chat.type
    
    # 🔧 ТЕСТОВЫЙ ЛОГ
    logger.info(f"🧪 adminlist = {adminlist}")
    logger.info(f"🧪 user_id = {user_id}, in adminlist? {user_id in adminlist}")
    
    # ============ ЛОГИКА ДОСТУПА ============
    
    # В ГРУППЕ: игнорируем админов из adminlist
    if chat_type in ["group", "supergroup"]:
        if is_admin_special(user_id):
            logger.debug(f"⏭️ Игнор admin {user_id} в группе (из adminlist.json)")
            return
        logger.info(f"✅ Обработаю обычного пользователя {user_id} в группе")
    
    # В ЛС (private): отвечаем ТОЛЬКО админам из ADMIN_IDS
    elif chat_type == "private":
        if user_id not in ADMIN_IDS:
            logger.info(f"🚫 БЛОКИРУЮ ЛС от {user_id} (не админ)")
            return
        logger.info(f"✅ Отвечу админу {user_id} в ЛС")
    
    # Проверка паузы (кроме главных админов из env)
    if is_paused() and user_id not in ADMIN_IDS:
        return
    
    # Валидация сообщения
    raw_text = (update.message.text or update.message.caption or "").strip()
    if not raw_text or raw_text.startswith("/") or len(raw_text) > 1500:
        return
    
    # Информация о пользователе для логов
    user = update.effective_user
    username = f"@{user.username}" if user.username else ""
    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    display_name = f"{name} {username}".strip() or "Без имени"
    
    logger.info(
        f"📨 ЗАПРОС | user={user.id} | {display_name} | "
        f"chat_type={chat_type} | \"{raw_text[:100]}{'...' if len(raw_text) > 100 else ''}\""
    )
    
    stats["total"] += 1
    save_stats()
    
    # Проверка кэша — отвечаем мгновенно, без "печатает"
    clean_text = preprocess(raw_text)
    cache_key = md5(clean_text.encode()).hexdigest()
    
    if cache_key in response_cache:
        stats["cached"] += 1
        save_stats()
        logger.info(f"💾 КЭШИРОВАННЫЙ ОТВЕТ для user={user.id}")
        await send_long_message(context.bot, update.effective_chat.id, response_cache[cache_key])
        return

    
    # ============ ALARM: отправка системного сообщения ============
    if current_alarm and chat_type in ["group", "supergroup"]:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"🔔 {current_alarm}",
                disable_notification=True  # Чтобы не будить всех
            )
        except Exception as e:
            logger.error(f"❌ Не удалось отправить alarm: {e}")


    
    # Показываем "печатает", только если ответ НЕ из кэша
    await safe_typing(context.bot, update.effective_chat.id)

    best_answer = None
    source = "none"
    distance = 1.0
    
    # ============ ЭТАП 1: Поиск по ключевым словам в Google Sheets ============
    try:
        all_rows = []
        
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="General!A:B").execute()
        all_rows.extend(result.get("values", []))
        
        result = sheet.values().get(spreadsheetId=SHEET_ID, range="Technical!A:B").execute()
        all_rows.extend(result.get("values", []))
        
        for row in all_rows:
            if len(row) >= 2:
                keyword = row[0].strip().lower()
                answer = row[1].strip()
                
                # Простое вхождение подстроки
                if keyword in clean_text or clean_text in keyword:
                    best_answer = answer
                    source = "keyword"
                    stats["keyword"] += 1
                    save_stats()
                    logger.info(f"🔑 KEYWORD MATCH | keyword=\"{keyword[:50]}\"")
                    break
                    
    except Exception as e:
        logger.error(f"❌ Ошибка Google Sheets: {e}", exc_info=True)
    
    # ============ ЭТАП 2: Векторный поиск (General) ============
    if not best_answer:
        answer, dist, top_log = await search_in_collection(
            collection_general,
            embedder_general,
            clean_text
        )
        
        if answer:
            # 🔍 Проверяем, не противоречит ли ответ вопросу
            if is_mismatch(raw_text, answer):
                mismatch_words = [w for w in CRITICAL_MISMATCHES.get("касса", []) if w in answer.lower()]
                if not mismatch_words:
                    mismatch_words = [w for w in CRITICAL_MISMATCHES.get("киоск", []) if w in answer.lower()]
                word = mismatch_words[0] if mismatch_words else "неизвестный термин"
                logger.warning(f"⚠️ НЕСООТВЕТСТВИЕ: вопрос про '{raw_text}' → но найден ответ с '{word}'")
                answer = None  # игнорируем
            else:
                best_answer = answer
                distance = dist
                source = "vector_general"
                stats["vector"] += 1
                save_stats()
            
                preview = (answer or "").replace("\n", " ")[:200]
                logger.info(
                    f"🎯 VECTOR (General) ✓ | dist={dist:.4f} | user={user.id} | "
                    f"→ \"{preview}\" | топ-3: {' | '.join(top_log[:3])}"
                )
        else:
            best_dist = top_log[0].split("→")[0] if top_log else "N/A"
            logger.info(
                f"❌ VECTOR (General) ✗ | лучший dist={best_dist} | "
                f"user={user.id} | топ-5: {' | '.join(top_log[:5])}"
            )
    
    # ============ ЭТАП 3: Векторный поиск (Technical) ============
    if not best_answer:
        answer, dist, top_log = await search_in_collection(
            collection_technical,
            embedder_technical,
            clean_text
        )
        
        if answer:
            best_answer = answer
            distance = dist
            source = "vector_technical"
            stats["vector"] += 1
            save_stats()
            
            preview = (answer or "").replace("\n", " ")[:200]
            logger.info(
                f"🎯 VECTOR (Technical) ✓ | dist={dist:.4f} | user={user.id} | "
                f"→ \"{preview}\" | топ-3: {' | '.join(top_log[:3])}"
            )
        else:
            best_dist = top_log[0].split("→")[0] if top_log else "N/A"
            logger.info(
                f"❌ VECTOR (Technical) ✗ | лучший dist={best_dist} | "
                f"user={user.id} | топ-5: {' | '.join(top_log[:5])}"
            )
    
    # ============ ЭТАП 4: Fallback через Groq ============
    if not best_answer:
        answer = await fallback_groq(raw_text)
        
        if answer:
            best_answer = answer
            source = "groq_fallback"
            logger.info(f"🤖 GROQ FALLBACK ✓ | len={len(answer)} | user={user.id}")
        else:
            logger.info(f"🤷 НЕТ ОТВЕТА | user={user.id} | запрос=\"{raw_text[:100]}\"")
    
    # ============ ЭТАП 5: Улучшение ответа через Groq ============
    final_reply = best_answer
    
    if best_answer and source in ["vector_general", "vector_technical", "keyword"] and len(best_answer) < 1200:
        improved = await improve_with_groq(best_answer, raw_text)
        
        if improved:
            final_reply = improved
            logger.info(
                f"✨ GROQ УЛУЧШИЛ | user={user.id} | "
                f"было={len(best_answer)} → стало={len(improved)}"
            )
    
    # ============ ЭТАП 6: Отправка ответа ============
    if not final_reply:
       # final_reply = (
       #     "Извините, я не смог найти точный ответ на ваш вопрос. "
       #     "Попробуйте переформулировать или обратитесь в поддержку."
       # )
        return 
        source = "default_fallback"
    
    # Сохраняем в кэш
    response_cache[cache_key] = final_reply
    
    logger.info(
        f"📤 ОТПРАВКА | source={source} | dist={distance:.3f} | "
        f"len={len(final_reply)} | user={user.id} | "
        f"\"{final_reply[:100]}{'...' if len(final_reply) > 100 else ''}\""
    )
    
    success = await send_long_message(context.bot, update.effective_chat.id, final_reply)
    
    if not success:
        stats["errors"] += 1
        save_stats()

# ====================== БЛОКИРОВКА ЛИЧНЫХ ЧАТОВ ======================
async def block_private(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Блокирует личные сообщения от не-админов"""
    if is_paused():
        return
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📞 Связаться с поддержкой", url="https://t.me/alexeymaloi")]
    ])
    
    await update.message.reply_text(
        "⚠️ Бот не отвечает в личных сообщениях.\n"
        "Используйте бота в группе или обратитесь напрямую:",
        reply_markup=keyboard
    )

# ====================== АДМИН-КОМАНДЫ ======================
async def reload_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Перезагрузка базы знаний"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    await update.message.reply_text("🔄 Начинаю перезагрузку базы...")
    await update_vector_db()
    await update.message.reply_text("✅ База знаний обновлена!")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ставит бота на паузу"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    set_paused(True)
    await update.message.reply_text(
        "⏸️ Бот на паузе\n"
        "Обычные пользователи не получают ответы"
    )

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Снимает бота с паузы"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    set_paused(False)
    await update.message.reply_text("▶️ Бот возобновил работу!")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает статус и статистику бота"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    paused = "⏸️ На паузе" if is_paused() else "▶️ Работает"
    
    count_general = collection_general.count() if collection_general else 0
    count_technical = collection_technical.count() if collection_technical else 0
    
    cache_usage = f"{len(response_cache)}/{CACHE_SIZE}"
    
    total = stats['total']
    cached_pct = (stats['cached'] / total * 100) if total > 0 else 0
    vector_pct = (stats['vector'] / total * 100) if total > 0 else 0
    
    text = (
        f"📊 СТАТУС БОТА\n\n"
        f"Состояние: {paused}\n"
        f"Записей в базе:\n"
        f"  • General: {count_general}\n"
        f"  • Technical: {count_technical}\n\n"
        f"Статистика запросов:\n"
        f"Всего: {stats['total']}\n"
        f"  • Из кэша: {stats['cached']} ({cached_pct:.1f}%)\n"
        f"  • Векторный поиск: {stats['vector']} ({vector_pct:.1f}%)\n"
        f"  • Ключевые слова: {stats['keyword']}\n"
        f"  • Groq API: {stats['groq']}\n"
        f"  • Ошибки: {stats['errors']}\n\n"
        f"Кэш: {cache_usage} записей\n"
        f"Порог вектора: {VECTOR_THRESHOLD}\n"
        f"\n"
        f"Alarm-уведомление:\n"
        f"  {'✅ Активно: ' + current_alarm[:50] + '...' if current_alarm and len(current_alarm) > 50 else current_alarm if current_alarm else '❌ Не установлено'}\n"
    )

    
    await update.message.reply_text(text)

async def clear_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очищает кэш ответов"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    old_size = len(response_cache)
    response_cache.clear()
    
    await update.message.reply_text(f"🗑️ Кэш очищен! Удалено {old_size} записей")

async def add_admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Добавляет администратора в adminlist"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "❌ Использование: /addadmin <user_id>\n"
            "Пример: /addadmin 123456789"
        )
        return
    
    user_id = int(context.args[0])
    add_admin(user_id)
    await update.message.reply_text(
        f"✅ Пользователь {user_id} добавлен в список администраторов\n"
        f"Теперь он игнорируется ботом в группах"
    )

async def remove_admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Удаляет администратора из adminlist"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "❌ Использование: /removeadmin <user_id>\n"
            "Пример: /removeadmin 123456789"
        )
        return
    
    user_id = int(context.args[0])
    
    if user_id not in adminlist:
        await update.message.reply_text(f"⚠️ Пользователь {user_id} не в списке")
        return
    
    remove_admin(user_id)
    await update.message.reply_text(f"✅ Пользователь {user_id} удалён из списка администраторов")

async def adminlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает список администраторов с никнеймами"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    if not adminlist:
        await update.message.reply_text("📋 Список администраторов пуст")
        return
    
    try:
        admin_info = []
        
        # ✅ Гарантируй int и сортируй
        for user_id in sorted([int(uid) for uid in adminlist]):
            try:
                user = await context.bot.get_chat(user_id)
                
                # Приоритет: @username > Full Name
                if user.username:
                    display = f"@{user.username}"
                else:
                    display = user.first_name or "Unknown"
                    if user.last_name:
                        display += f" {user.last_name}"
                
                admin_info.append(f"  • {user_id} ({display})")
                
            except Exception as e:
                logger.warning(f"⚠️ Не удалось получить юзера {user_id}: {e}")
                admin_info.append(f"  • {user_id} (⚠️ Ошибка)")
        
        message = f"👨‍💼 АДМИНИСТРАТОРЫ ({len(adminlist)}):\n\n" + "\n".join(admin_info)
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"❌ adminlist_cmd error: {e}")
        await update.message.reply_text(f"⚠️ Системная ошибка: {str(e)}")

async def addalarm_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает alarm-сообщение, которое бот будет выводить при каждом сообщении"""
    if update.effective_user.id not in ADMIN_IDS:
        return

    if not context.args:
        await update.message.reply_text('❌ Использование: /addalarm "Текст сообщения"')
        return

    # Собираем аргументы, учитывая кавычки
    raw_text = " ".join(context.args)
    # Пытаемся извлечь текст в кавычках
    import re
    match = re.search(r'"([^"]+)"', raw_text)
    if match:
        text = match.group(1)
    else:
        text = raw_text  # Если кавычек нет — берём всё

    if not text.strip():
        await update.message.reply_text("❌ Текст сообщения пуст!")
        return

    global current_alarm
    current_alarm = text.strip()
    save_alarm(current_alarm)

    await update.message.reply_text(
        f"📢 Alarm установлен:\n\n{current_alarm}\n\n"
        "✅ Бот будет показывать это при каждом сообщении."
    )

async def delalarm_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Удаляет текущий alarm"""
    if update.effective_user.id not in ADMIN_IDS:
        return

    global current_alarm
    if current_alarm is None:
        await update.message.reply_text("🔇 Нет активного alarm для удаления.")
        return

    clear_alarm()
    current_alarm = None

    await update.message.reply_text("✅ Alarm удалён.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает список команд"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    text = (
        "📌 КОМАНДЫ АДМИНИСТРАТОРА\n\n"
        "Управление ботом:\n"
        "/pause — поставить бота на паузу\n"
        "/resume — возобновить работу\n"
        "/status — показать статус и статистику\n"
        "/reload — перезагрузить базу знаний\n\n"
        "Управление кэшем:\n"
        "/clearcache — очистить кэш ответов\n\n"
        "Управление уведомлениями:\n"
        "/addalarm \"текст\" — установить уведомление при каждом сообщении\n"
        "/delalarm — удалить уведомление\n\n"
        "Управление администраторами:\n"
        "/addadmin [user_id] — добавить в adminlist\n"
        "/removeadmin <user_id> — удалить из adminlist\n"
        "/adminlist — показать список\n\n"
        "/help — показать это меню\n\n"
        "💡 Админы из adminlist.json игнорируются ботом в группах"
    )
    
    await update.message.reply_text(text)

async def set_threshold_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Изменяет порог векторного поиска (для экспериментов)"""
    global VECTOR_THRESHOLD
    
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    if not context.args or not context.args[0].replace(".", "").isdigit():
        await update.message.reply_text(
            f"❌ Использование: /threshold <значение>\n"
            f"Текущий порог: {VECTOR_THRESHOLD}\n"
            f"Рекомендуемый диапазон: 0.5-0.8"
        )
        return
    
    try:
        new_threshold = float(context.args[0])
        
        if not 0.0 <= new_threshold <= 1.0:
            await update.message.reply_text("❌ Порог должен быть от 0.0 до 1.0")
            return
        
        old_threshold = VECTOR_THRESHOLD
        VECTOR_THRESHOLD = new_threshold
        
        await update.message.reply_text(
            f"✅ Порог изменён: {old_threshold} → {new_threshold}\n\n"
            f"⚠️ Это изменение временное (до перезапуска бота)"
        )
        
        logger.info(f"🎚️ Порог изменён: {old_threshold} → {new_threshold}")
        
    except ValueError:
        await update.message.reply_text("❌ Неверный формат числа")

# ====================== ОБРАБОТЧИК ОШИБОК ======================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    logger.error(f"❌ Необработанная ошибка: {context.error}", exc_info=context.error)
    
    stats["errors"] += 1
    save_stats()
    
    # Пытаемся уведомить пользователя если возможно
    if update and isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Произошла внутренняя ошибка. Попробуйте позже или обратитесь к администратору."
            )
        except Exception:
            pass

# ====================== GRACEFUL SHUTDOWN ======================
async def shutdown(application: Application):
    """Корректное завершение работы бота"""
    logger.info("🛑 Начало корректного завершения работы...")
    
    # Сохраняем все данные
    save_stats()
    save_adminlist()
    
    logger.info("💾 Все данные сохранены")
    logger.info("👋 Бот остановлен")

# ====================== ЗАПУСК БОТА ======================

if __name__ == "__main__":
    logger.info("🚀 Запуск бота...")
    
    # Загружаем сохранённые данные
    adminlist = load_adminlist()
    logger.info(f"📋 Текущих админов в списке: {len(adminlist)}")
    load_stats()
    
    # Загружаем alarm
    current_alarm = load_alarm()

    # Создаём приложение
    app = Application.builder()\
        .token(TELEGRAM_TOKEN)\
        .concurrent_updates(False)\
        .build()
    
    # ============ ФИЛЬТРЫ СООБЩЕНИЙ ============
    
    # Блокировка личных чатов для не-админов
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & 
        ~filters.COMMAND & 
        ~filters.User(user_id=ADMIN_IDS),
        block_private
    ))
    
    # Обработка текстовых сообщений
    # В группах: от всех кроме adminlist
    # В личке: только от ADMIN_IDS
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & (
            # Личные чаты админов
            (filters.ChatType.PRIVATE & filters.User(user_id=ADMIN_IDS)) |
            # Все группы
            (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP)
        ),
        handle_message
    ))
    
    # Обработка сообщений с подписями (caption)
    app.add_handler(MessageHandler(
        filters.CAPTION & ~filters.COMMAND & (
            (filters.ChatType.PRIVATE & filters.User(user_id=ADMIN_IDS)) |
            (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP)
        ),
        handle_message
    ))
    
    # ============ КОМАНДЫ АДМИНИСТРАТОРА ============
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("clearcache", clear_cache))
    app.add_handler(CommandHandler("addadmin", add_admin_cmd))
    app.add_handler(CommandHandler("removeadmin", remove_admin_cmd))
    app.add_handler(CommandHandler("adminlist", adminlist_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("threshold", set_threshold_cmd))
    app.add_handler(CommandHandler("addalarm", addalarm_cmd))
    app.add_handler(CommandHandler("delalarm", delalarm_cmd))

    
    # ============ ОБРАБОТЧИК ОШИБОК ============
    app.add_error_handler(error_handler)
    
    # ============ ОТЛОЖЕННЫЕ ЗАДАЧИ ============
    # Загружаем базу через 15 секунд после старта
    app.job_queue.run_once(update_vector_db, when=15)
    
    # Опционально: Автоматическое обновление базы каждые 6 часов
    # app.job_queue.run_repeating(update_vector_db, interval=21600, first=15)
    
    # ============ ЗАПУСК ============
    logger.info("=" * 60)
    logger.info("✅ БОТ ГОТОВ К РАБОТЕ")
    logger.info(f"📊 Порог вектора: {VECTOR_THRESHOLD}")
    logger.info(f"👥 Главных админов: {len(ADMIN_IDS)}")
    logger.info(f"👨‍💼 Админов в списке: {len(adminlist)}")
    logger.info(f"📈 Всего запросов: {stats['total']}")
    logger.info("=" * 60)
    
    try:
        app.run_polling(
            drop_pending_updates=True,
            close_loop=False
        )
    except KeyboardInterrupt:
        logger.info("⌨️ Получен сигнал остановки (Ctrl+C)")
    finally:
        # Корректное завершение
        import asyncio
        asyncio.run(shutdown(app))
