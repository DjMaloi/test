import os
import re
import json
import logging
import asyncio
from hashlib import md5
from typing import Optional, Tuple, List
from contextlib import asynccontextmanager
from functools import lru_cache

# Telegram imports
import asyncio
import logging
import os
import hashlib
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
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

# ====================== КЛАССЫ ИСКЛЮЧЕНИЙ ======================
class BotError(Exception):
    """Базовый класс для ошибок бота"""
    pass

class DatabaseError(BotError):
    """Ошибки базы данных (ChromaDB, Google Sheets)"""
    pass

class AIServiceError(BotError):
    """Ошибки AI сервисов (Groq, эмбеддинги)"""
    pass

class TelegramError(BotError):
    """Ошибки Telegram API"""
    pass

class ConfigurationError(BotError):
    """Ошибки конфигурации"""
    pass

CRITICAL_MISMATCHES = {
    "касса": ["киоск", "КСО", "сканер", "принтер чеков", "терминал самообслуживания"],
    "киоск": ["касса", "онлайн-касса", "фискальный регистратор", "терминал оплаты"],
}

def is_mismatch(question: str, answer: str) -> bool:
    """
    Проверяет, не противоречит ли ответ вопросу
    """
    question_lower = question.lower()
    answer_lower = answer.lower()

    # Правило 1: вопрос про кассу → ответ не должен содержать "киоск", "КСО"
    if "касса" in question_lower:
        forbidden = ["киоск", "КСО", "самообслуживания", "самообслуживани", "kiosk"]
        for word in forbidden:
            if word.lower() in answer_lower:
                return True

    # Правило 2: вопрос про киоск → ответ не должен содержать "касса", "онлайн-касса"
    if "киоск" in question_lower or "КСО" in question_lower or "самообслуживани" in question_lower:
        forbidden = ["касса", "онлайн-касса", "фискальный", "регистратор", "терминал оплаты"]
        for word in forbidden:
            if word in answer_lower:
                return True

    return False

# ====================== LOGGING ======================
LOG_FILE = "/app/data/bot.log"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Уменьшаем шум от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

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
LOG_FILE = "/app/data/bot.log"



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
        #logger.info(f"🔍 Ищу adminlist.json по пути: {ADMINLIST_FILE}")
        
        os.makedirs(os.path.dirname(ADMINLIST_FILE), exist_ok=True)
        
        with open(ADMINLIST_FILE, "r") as f:
            data = json.load(f)
           # logger.info(f"📄 Прочитан файл: {data}")
        
        # ИЗМЕНЕНИЕ №1: поддержка формата {"admins": [...]}
        adminlist = {int(x) for x in data.get("admins", [])}
        #logger.info(f"✅ Загружено {len(adminlist)} админов: {adminlist}")
        return adminlist
    
    except FileNotFoundError:
        #logger.error(f"❌ Файл не найден: {ADMINLIST_FILE}")
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
                #logger.info(f"✓ Статистика загружена: {stats['total']} запросов")
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

# Кэширование эмбеддингов для ускорения
@lru_cache(maxsize=1000)
def get_embedding_general(text: str) -> List[float]:
    """Кэшированное получение эмбеддинга для General модели"""
    try:
        return embedder_general.encode(text).tolist()
    except Exception as e:
        logger.error(f"❌ Ошибка эмбеддинга General: {e}")
        raise AIServiceError(f"General embedding error: {e}")

@lru_cache(maxsize=1000)
def get_embedding_technical(text: str) -> List[float]:
    """Кэшированное получение эмбеддинга для Technical модели"""
    try:
        return embedder_technical.encode(text).tolist()
    except Exception as e:
        logger.error(f"❌ Ошибка эмбеддинга Technical: {e}")
        raise AIServiceError(f"Technical embedding error: {e}")

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
    embedder_type: str,
    query: str,
    threshold: float = VECTOR_THRESHOLD,
    n_results: int = 10
) -> Tuple[Optional[str], float, List[str]]:
    """
    Универсальная функция векторного поиска с кэшированием эмбеддингов
    
    Возвращает: (лучший_ответ, расстояние, топ_результаты_для_логов)
    """
    if not collection or collection.count() == 0:
        return None, 1.0, []
    
    try:
        # Используем кэшированные эмбеддинги
        if embedder_type == "general":
            emb = get_embedding_general(query)
        elif embedder_type == "technical":
            emb = get_embedding_technical(query)
        else:
            raise AIServiceError(f"Unknown embedder type: {embedder_type}")
        
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
        
    except chromadb.errors.DuplicateIDException as e:
        logger.warning(f"⚠️ Дубликат ID в векторном поиске: {e}")
        return None, 1.0, []
    except chromadb.errors.InvalidDimensionException as e:
        logger.error(f"❌ Неверная размерность вектора: {e}")
        return None, 1.0, []
    except Exception as e:
        logger.error(f"❌ Ошибка векторного поиска: {e}", exc_info=True)
        raise DatabaseError(f"Vector search error: {e}")

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
    Улучшает ответ через Groq с учетом типа запроса
    
    Возвращает улучшенный ответ или None при ошибке
    """
    # Определяем тип запроса
    query_type = classify_query_type(question)
    system_prompt = get_contextual_prompt(query_type, is_fallback=False)
    
    user_prompt = f"Исходный ответ:\n{original_answer}\n\nВопрос: {question}\n\nУлучшенный ответ:"
    
    # 🔒 Запрет улучшения ложных ответов
    if "касса" in question.lower() and "киоск" in original_answer.lower():
        logger.warning("⚠️ Запрет улучшения: вопрос про 'кассу', но ответ содержит 'киоск'")
        return None

    if "киоск" in question.lower() and "касса" in original_answer.lower():
        logger.warning("⚠️ Запрет улучшения: вопрос про 'киоск', но ответ содержит 'кассу'")
        return None

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
                logger.info(f"✨ GROQ УЛУЧШИЛ ({query_type}) | было={len(original_answer)} → стало={len(improved)}")
                return improved
            
            return None
            
    except Exception as e:
        logger.warning(f"⚠️ Groq улучшение не удалось ({query_type}): {e}")
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
    """Обновляет векторную базу из Google Sheets с сохранением query в метаданных"""
    global collection_general, collection_technical
    
    async with collection_lock:
        try:
            logger.info("🔄 Начало обновления базы знаний из Google Sheets...")
            
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
            
            # Удаляем старые коллекции
            for name in ["general_kb", "technical_kb"]:
                try:
                    chroma_client.delete_collection(name)
                    logger.debug(f"🗑️ Удалена коллекция: {name}")
                except Exception as e:
                    logger.debug(f"🔍 Коллекция {name} не найдена или уже удалена: {e}")
            
            # Создаём новые коллекции
            collection_general = chroma_client.create_collection("general_kb")
            collection_technical = chroma_client.create_collection("technical_kb")
            
            # === Заполняем General ===
            if general_rows:
                # Фильтруем пустые строки
                valid_rows = [row for row in general_rows if len(row) >= 2 and row[0].strip()]
                
                keys = [row[0].strip() for row in valid_rows]
                answers = [row[1].strip() for row in valid_rows]
                
                # Используем кэшированные эмбеддинги
                embeddings = [get_embedding_general(key) for key in keys]
                
                # Сохраняем query + answer в метаданных
                collection_general.add(
                    ids=[f"general_{i}" for i in range(len(valid_rows))],
                    documents=keys,  # для векторного поиска
                    metadatas=[
                        {"query": keys[i], "answer": answers[i]} 
                        for i in range(len(valid_rows))
                    ],
                    embeddings=embeddings
                )
                
                logger.info(f"✅ General: добавлено {len(valid_rows)} пар (вопрос/ответ)")
            else:
                logger.info("🟡 General: нет данных для загрузки")
            
            # === Заполняем Technical ===
            if technical_rows:
                valid_rows = [row for row in technical_rows if len(row) >= 2 and row[0].strip()]
                
                keys = [row[0].strip() for row in valid_rows]
                answers = [row[1].strip() for row in valid_rows]
                
                # Используем кэшированные эмбеддинги
                embeddings = [get_embedding_technical(key) for key in keys]
                
                collection_technical.add(
                    ids=[f"technical_{i}" for i in range(len(valid_rows))],
                    documents=keys,
                    metadatas=[
                        {"query": keys[i], "answer": answers[i]} 
                        for i in range(len(valid_rows))
                    ],
                    embeddings=embeddings
                )
                
                logger.info(f"✅ Technical: добавлено {len(valid_rows)} пар (вопрос/ответ)")
            else:
                logger.info("🟡 Technical: нет данных для загрузки")
            
            logger.info("🟢 Обновление векторной базы завершено успешно!")
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка обновления базы: {e}", exc_info=True)
            stats["errors"] += 1
            save_stats()



def get_source_emoji(source: str) -> str:
    """Возвращает смайлик в зависимости от источника ответа"""
    emoji_map = {
        "cached": "💾",           # Из кэша
        "keyword": "🔑",          # Ключевые слова
        "vector_general": "🎯",   # Векторный поиск (General)
        "vector_technical": "⚙️", # Векторный поиск (Technical)
        "groq_fallback": "🤖",    # Ответ от AI
        "default_fallback": "❓"  # Не найдено
    }
    return emoji_map.get(source, "")


async def run_startup_test(context: ContextTypes.DEFAULT_TYPE):
    """Запускает автопроверку ключевого поиска при старте"""
    logger.info("🧪 Запуск автопроверки ключевого поиска...")

    # Тестовый ключ, который ДОЛЖЕН быть в базе
    test_query = "как дела"  # ← ЗАМЕНИ НА ЛЮБОЙ РЕАЛЬНЫЙ, ЕСТЬ В ТАБЛИЦЕ
    clean_test = preprocess(test_query)

    try:
        # Проверяем General
        results = collection_general.get(
            where={"query": {"$eq": clean_test}},
            include=["metadatas"]
        )

        if results["metadatas"]:
            answer = results["metadatas"][0]["answer"]
            logger.info(f"✅ УСПЕШНЫЙ ТЕСТ: найдено в General → '{answer}'")
        else:
            # Проверяем Technical
            results = collection_technical.get(
                where={"query": {"$eq": clean_test}},
                include=["metadatas"]
            )
            if results["metadatas"]:
                answer = results["metadatas"][0]["answer"]
                logger.info(f"✅ УСПЕШНЫЙ ТЕСТ: найдено в Technical → '{answer}'")
            else:
                logger.warning(f"❌ НЕ НАЙДЕНО: ключевой запрос '{test_query}' не найден ни в одной базе!")
                logger.warning("🔧 Проверь: 1) Есть ли он в Google Sheets? 2) Выполнен ли /reload? 3) Правильно ли сохраняется query в metadatas?")
    except Exception as e:
        logger.error(f"❌ ОШИБКА при автопроверке: {e}", exc_info=True)


# ====================== КЛАССИФИКАЦИЯ ЗАПРОСОВ ======================
def classify_query_type(query: str) -> str:
    """
    Определяет тип запроса для выбора лучшей стратегии ответа
    
    Returns:
        'technical' - технический вопрос (касса, киоск, оборудование)
        'general' - общий вопрос (работа, доступ, поддержка)
        'mixed' - смешанный тип
    """
    query_lower = query.lower()
    
    # Технические ключевые слова
    technical_keywords = [
        'касса', 'киоск', 'ксо', 'терминал', 'оборудование', 
        'принтер', 'сканер', 'фискальный', 'чек', 'оплата',
        'самообслуживание', 'интеграция', 'настройка', 'ошибка',
        'сбой', 'ремонт', 'установка', 'подключение'
    ]
    
    # Общие ключевые слова
    general_keywords = [
        'работа', 'часы', 'график', 'доступ', 'поддержка',
        'контакты', 'адрес', 'телефон', 'email', 'помощь',
        'вопрос', 'ответ', 'стоимость', 'цена', 'оплата'
    ]
    
    technical_count = sum(1 for word in technical_keywords if word in query_lower)
    general_count = sum(1 for word in general_keywords if word in query_lower)
    
    if technical_count > general_count and technical_count > 0:
        return 'technical'
    elif general_count > 0:
        return 'general'
    else:
        return 'mixed'

def get_contextual_prompt(query_type: str, is_fallback: bool = False) -> str:
    """
    Возвращает контекстный промпт в зависимости от типа запроса
    
    Args:
        query_type: 'technical', 'general', 'mixed'
        is_fallback: True для fallback запросов, False для улучшения ответов
    """
    
    if is_fallback:
        # Промпты для fallback (когда ничего не найдено в базе)
        prompts = {
            'technical': (
                "Ты — технический специалист поддержки. Отвечай ТОЛЬКО на технические вопросы.\n\n"
                "СТРОГИЕ ПРАВИЛА:\n"
                "1. Если вопрос не технический или данных недостаточно — ответь: 'НЕТ ДАННЫХ'.\n"
                "2. Отвечай только про кассы, киоски, терминалы, оборудование.\n"
                "3. НЕ выдумывай спецификации, модели, цены.\n"
                "4. Сохраняй точную терминологию: 'касса' ≠ 'киоск'.\n"
                "5. Ответ — до 600 символов, без форматирования.\n\n"
                "ОБЛАСТЬ КОМПЕТЕНЦИИ:\n"
                "- Оборудование: кассы, киоски, принтеры чеков, сканеры\n"
                "- Программное обеспечение: настройка, интеграция\n"
                "- Технические проблемы: ошибки, сбои, ремонт\n\n"
                "ФОРМАТ: Краткий технический ответ."
            ),
            'general': (
                "Ты — консультант поддержки. Отвечай на общие вопросы о работе компании.\n\n"
                "СТРОГИЕ ПРАВИЛА:\n"
                "1. Если вопрос технический — ответь: 'НЕТ ДАННЫХ'.\n"
                "2. Отвечай только про работу, контакты, услуги.\n"
                "3. НЕ давай технических консультаций.\n"
                "4. Ответ — до 600 символов, дружелюбный тон.\n\n"
                "ОБЛАСТЬ КОМПЕТЕНЦИИ:\n"
                "- Режим работы, часы, контакты\n"
                "- Услуги, стоимость, условия\n"
                "- Общая информация о компании\n\n"
                "ФОРМАТ: Дружелюбный краткий ответ."
            ),
            'mixed': (
                "Ты — универсальный консультант. Определи тип вопроса и отвечай соответственно.\n\n"
                "СТРОГИЕ ПРАВИЛА:\n"
                "1. Если unsure — ответь: 'НЕТ ДАННЫХ'.\n"
                "2. Технические вопросы: кратко и по делу.\n"
                "3. Общие вопросы: дружелюбно и понятно.\n"
                "4. НЕ смешивай технические и общие темы.\n"
                "5. Ответ — до 600 символов.\n\n"
                "ФОРМАТ: Адаптивный ответ под тип вопроса."
            )
        }
    else:
        # Промпты для улучшения существующих ответов
        prompts = {
            'technical': (
                "Ты — технический редактор. Улучши технический ответ, сохранив точность.\n\n"
                "ПРАВИЛА УЛУЧШЕНИЯ:\n"
                "1. Упрости сложные термины, но НЕ меняй их.\n"
                "2. Добавь структуру, если поможет понять.\n"
                "3. Сохраняй все технические детали и параметры.\n"
                "4. НЕ заменяй 'касса' ↔ 'киоск'.\n"
                "5. Длина — до 800 символов.\n\n"
                "ЦЕЛЬ: Сделать технический ответ понятнее без потери точности."
            ),
            'general': (
                "Ты — редактор поддержки. Улучши общий ответ, сделав его дружелюбнее.\n\n"
                "ПРАВИЛА УЛУЧШЕНИЯ:\n"
                "1. Добавь дружелюбный тон и эмпатию.\n"
                "2. Структурируй информацию для лучшего понимания.\n"
                "3. Упрости формулировки без потери смысла.\n"
                "4. Длина — до 800 символов.\n\n"
                "ЦЕЛЬ: Сделать ответ более helpful и понятным."
            ),
            'mixed': (
                "Ты — универсальный редактор. Адаптируй ответ под контекст вопроса.\n\n"
                "ПРАВИЛА УЛУЧШЕНИЯ:\n"
                "1. Определи тип вопроса и адаптируй стиль.\n"
                "2. Технические детали — точными, общие — понятными.\n"
                "3. Сохраняй баланс между детализацией и простотой.\n"
                "4. Длина — до 800 символов.\n\n"
                "ЦЕЛЬ: Идеальный баланс техничности и понятности."
            )
        }
    
    return prompts.get(query_type, prompts['mixed'])

# ====================== GRACEFUL DEGRADATION ======================
async def robust_search(query: str, raw_text: str) -> Tuple[Optional[str], str, float]:
    """
    Надежный поиск с плавным снижением качества при проблемах
    
    Порядок попыток:
    1. Кэш ответов
    2. Поиск по ключевым словам  
    3. Векторный поиск (General + Technical)
    4. Groq fallback
    5. Сообщение об ошибке
    
    Returns:
        (answer, source, distance)
    """
    clean_text = preprocess(query)
    
    # Попытка 1: Кэш ответов
    try:
        cache_key = md5(clean_text.encode()).hexdigest()
        if cache_key in response_cache:
            stats["cached"] += 1
            save_stats()
            logger.info(f"💾 КЭШИРОВАННЫЙ ОТВЕТ (robust)")
            return response_cache[cache_key], "cached", 0.0
    except Exception as e:
        logger.warning(f"⚠️ Ошибка кэша: {e}")
    
    # Попытка 2: Поиск по ключевым словам
    try:
        keyword_answer = await unified_keyword_search(clean_text)
        if keyword_answer:
            logger.info(f"🔑 КЛЮЧЕВОЙ ПОИСК (robust)")
            return keyword_answer, "keyword", 0.0
    except Exception as e:
        logger.warning(f"⚠️ Ошибка поиска по ключевым словам: {e}")
    
    # Попытка 3: Векторный поиск General
    try:
        answer, dist, _ = await search_in_collection(collection_general, "general", clean_text)
        if answer and dist < VECTOR_THRESHOLD:
            # Проверка на несоответствие
            if not is_mismatch(raw_text, answer):
                stats["vector"] += 1
                save_stats()
                logger.info(f"🎯 ВЕКТОРНЫЙ ПОИСК General (robust) | dist={dist:.4f}")
                return answer, "vector_general", dist
            else:
                logger.warning(f"⚠️ НЕСООТВЕТСТВИЕ в General, пробуем Technical")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка векторного поиска General: {e}")
    
    # Попытка 4: Векторный поиск Technical
    try:
        answer, dist, _ = await search_in_collection(collection_technical, "technical", clean_text)
        if answer and dist < VECTOR_THRESHOLD:
            stats["vector"] += 1
            save_stats()
            logger.info(f"🎯 ВЕКТОРНЫЙ ПОИСК Technical (robust) | dist={dist:.4f}")
            return answer, "vector_technical", dist
    except Exception as e:
        logger.warning(f"⚠️ Ошибка векторного поиска Technical: {e}")
    
    # Попытка 5: Groq fallback
    try:
        groq_answer = await fallback_groq(raw_text)
        if groq_answer:
            logger.info(f"🤖 GROQ FALLBACK (robust)")
            return groq_answer, "groq_fallback", 1.0
    except Exception as e:
        logger.warning(f"⚠️ Ошибка Groq fallback: {e}")
    
    # Попытка 6: Ultimate fallback
    logger.error(f"🚨 ВСЕ МЕТОДЫ ПОИСКА ПРОВАЛИЛИСЬ для запроса: '{query[:50]}...'")
    stats["errors"] += 1
    save_stats()
    
    return None, "error", 1.0

async def notify_admins_about_problems(context: ContextTypes.DEFAULT_TYPE, problem_type: str, error_msg: str):
    """Уведомляет админов о проблемах с сервисами"""
    if not ADMIN_IDS:
        return
    
    message = f"🚨 ПРОБЛЕМА С СЕРВИСАМИ\n\nТип: {problem_type}\nОшибка: {error_msg}\n\nВремя: {datetime.now().strftime('%H:%M:%S')}"
    
    for admin_id in ADMIN_IDS:
        try:
            await context.bot.send_message(
                chat_id=admin_id,
                text=message
            )
        except Exception as e:
            logger.warning(f"⚠️ Не удалось уведомить админа {admin_id}: {e}")

# ====================== HEALTH CHECKS ======================
async def check_google_sheets_health() -> Dict[str, Any]:
    """Проверка доступности Google Sheets"""
    try:
        result = sheet.values().get(
            spreadsheetId=SHEET_ID, 
            range="General!A1:A1"
        ).execute()
        return {
            "status": "✅ OK",
            "response_time": "fast",
            "error": None
        }
    except googleapiclient.errors.HttpError as e:
        return {
            "status": "❌ HTTP Error", 
            "response_time": "N/A",
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "❌ Error",
            "response_time": "N/A", 
            "error": str(e)
        }

async def check_groq_health() -> Dict[str, Any]:
    """Проверка доступности Groq API"""
    try:
        start_time = time.time()
        async with groq_with_timeout():
            resp = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    temperature=0.0,
                ),
                timeout=5
            )
        response_time = f"{(time.time() - start_time)*1000:.0f}ms"
        return {
            "status": "✅ OK",
            "response_time": response_time,
            "error": None
        }
    except Exception as e:
        return {
            "status": "❌ Error",
            "response_time": "N/A",
            "error": str(e)
        }

def check_chromadb_health() -> Dict[str, Any]:
    """Проверка состояния ChromaDB"""
    try:
        general_count = collection_general.count() if collection_general else 0
        technical_count = collection_technical.count() if collection_technical else 0
        
        return {
            "status": "✅ OK",
            "general_records": general_count,
            "technical_records": technical_count,
            "error": None
        }
    except Exception as e:
        return {
            "status": "❌ Error",
            "general_records": 0,
            "technical_records": 0,
            "error": str(e)
        }

def check_embedding_models_health() -> Dict[str, Any]:
    """Проверка состояния моделей эмбеддингов"""
    try:
        # Тестовое эмбеддингирование
        test_text = "тест"
        general_emb = get_embedding_general(test_text)
        technical_emb = get_embedding_technical(test_text)
        
        general_cache = get_embedding_general.cache_info()
        technical_cache = get_embedding_technical.cache_info()
        
        return {
            "status": "✅ OK",
            "general_cache": f"{general_cache.currsize}/{general_cache.maxsize}",
            "technical_cache": f"{technical_cache.currsize}/{technical_cache.maxsize}",
            "error": None
        }
    except Exception as e:
        return {
            "status": "❌ Error",
            "general_cache": "N/A",
            "technical_cache": "N/A", 
            "error": str(e)
        }

async def run_health_checks() -> Dict[str, Any]:
    """Запуск всех проверок здоровья"""
    logger.info("🔍 Запуск health checks...")
    
    # Параллельное выполнение проверок
    sheets_task = asyncio.create_task(check_google_sheets_health())
    groq_task = asyncio.create_task(check_groq_health())
    
    sheets_result = await sheets_task
    groq_result = await groq_task
    
    chromadb_result = check_chromadb_health()
    embedding_result = check_embedding_models_health()
    
    # Общий статус
    all_ok = all([
        sheets_result["status"] == "✅ OK",
        groq_result["status"] == "✅ OK", 
        chromadb_result["status"] == "✅ OK",
        embedding_result["status"] == "✅ OK"
    ])
    
    overall_status = "🟢 Все системы работают" if all_ok else "🟡 Есть проблемы"
    
    return {
        "overall": overall_status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "google_sheets": sheets_result,
        "groq_api": groq_result,
        "chromadb": chromadb_result,
        "embedding_models": embedding_result
    }

async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда проверки здоровья системы"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    await update.message.reply_text("🔍 Проверяю состояние системы...")
    
    health_report = await run_health_checks()
    
    text = (
        f"🏥 HEALTH CHECK\n\n"
        f"Общий статус: {health_report['overall']}\n"
        f"Время проверки: {health_report['timestamp']}\n\n"
        f"📊 Google Sheets:\n"
        f"  Статус: {health_report['google_sheets']['status']}\n"
        f"  Время ответа: {health_report['google_sheets']['response_time']}\n"
        f"  Ошибка: {health_report['google_sheets']['error'] or 'Нет'}\n\n"
        f"🤖 Groq API:\n"
        f"  Статус: {health_report['groq_api']['status']}\n"
        f"  Время ответа: {health_report['groq_api']['response_time']}\n"
        f"  Ошибка: {health_report['groq_api']['error'] or 'Нет'}\n\n"
        f"🗄️ ChromaDB:\n"
        f"  Статус: {health_report['chromadb']['status']}\n"
        f"  General записей: {health_report['chromadb']['general_records']}\n"
        f"  Technical записей: {health_report['chromadb']['technical_records']}\n"
        f"  Ошибка: {health_report['chromadb']['error'] or 'Нет'}\n\n"
        f"🧠 Модели эмбеддингов:\n"
        f"  Статус: {health_report['embedding_models']['status']}\n"
        f"  General кэш: {health_report['embedding_models']['general_cache']}\n"
        f"  Technical кэш: {health_report['embedding_models']['technical_cache']}\n"
        f"  Ошибка: {health_report['embedding_models']['error'] or 'Нет'}"
    )
    
    await update.message.reply_text(text)

# ====================== ОТПРАВКА СООБЩЕНИЙ ======================
async def send_long_message(bot, chat_id: int, text: str, max_retries: int = 3, reply_to_message_id: int = None):

    """
    Безопасно отправляет длинное сообщение с разбивкой и повторами
    """
    for attempt in range(max_retries):
        try:
            # Разбиваем на части если нужно
            chunks = [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]
            for idx, chunk in enumerate(chunks):
                # Цитируем только первое сообщение
                reply_id = reply_to_message_id if idx == 0 else None
                await bot.send_message(
                    chat_id=chat_id, 
                    text=chunk,
                    reply_to_message_id=reply_id
                )

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



# ====================== УНИФИЦИРОВАННЫЙ ПОИСК ======================
async def unified_keyword_search(clean_text: str) -> Optional[str]:
    """
    Единая функция поиска по ключевым словам
    
    Приоритет:
    1. Поиск в метаданных ChromaDB (быстро)
    2. Если ничего не найдено - поиск в Google Sheets (медленно)
    """
    # Этап 1: Быстрый поиск в метаданных ChromaDB
    try:
        # Поиск в General
        results = collection_general.get(
            where={"query": {"$eq": clean_text}},
            include=["metadatas"]
        )
        if results["metadatas"]:
            answer = results["metadatas"][0].get("answer")
            if answer:
                stats["keyword"] += 1
                save_stats()
                logger.info(f"🔑 KEYWORD MATCH (General) | query='{clean_text}'")
                return answer
    except chromadb.errors.DuplicateIDException as e:
        logger.warning(f"⚠️ Дубликат ID в ChromaDB General: {e}")
    except chromadb.errors.InvalidDimensionException as e:
        logger.error(f"❌ Неверная размерность вектора в General: {e}")
    except Exception as e:
        logger.error(f"❌ Ошибка поиска в метаданных General: {e}", exc_info=True)
        raise DatabaseError(f"ChromaDB General error: {e}")

    try:
        # Поиск в Technical
        results = collection_technical.get(
            where={"query": {"$eq": clean_text}},
            include=["metadatas"]
        )
        if results["metadatas"]:
            answer = results["metadatas"][0].get("answer")
            if answer:
                stats["keyword"] += 1
                save_stats()
                logger.info(f"🔑 KEYWORD MATCH (Technical) | query='{clean_text}'")
                return answer
    except chromadb.errors.DuplicateIDException as e:
        logger.warning(f"⚠️ Дубликат ID в ChromaDB Technical: {e}")
    except chromadb.errors.InvalidDimensionException as e:
        logger.error(f"❌ Неверная размерность вектора в Technical: {e}")
    except Exception as e:
        logger.error(f"❌ Ошибка поиска в метаданных Technical: {e}", exc_info=True)
        raise DatabaseError(f"ChromaDB Technical error: {e}")

    # Этап 2: Поиск в Google Sheets (только если ничего не найдено)
    try:
        result_general = sheet.values().get(spreadsheetId=SHEET_ID, range="General!A:B").execute()
        general_rows = result_general.get("values", [])
        
        result_technical = sheet.values().get(spreadsheetId=SHEET_ID, range="Technical!A:B").execute()
        technical_rows = result_technical.get("values", [])
        
        all_rows = general_rows + technical_rows
        
        for row in all_rows:
            if len(row) >= 2:
                keyword = row[0].strip().lower()
                answer = row[1].strip()
                
                # Простое вхождение подстроки
                if keyword in clean_text or clean_text in keyword:
                    stats["keyword"] += 1
                    save_stats()
                    logger.info(f"🔑 KEYWORD MATCH (Sheets) | keyword=\"{keyword[:50]}\"")
                    return answer
                    
    except googleapiclient.errors.HttpError as e:
        logger.error(f"❌ HTTP ошибка Google Sheets: {e}")
        raise DatabaseError(f"Google Sheets HTTP error: {e}")
    except googleapiclient.errors.Error as e:
        logger.error(f"❌ Ошибка API Google Sheets: {e}")
        raise DatabaseError(f"Google Sheets API error: {e}")
    except Exception as e:
        logger.error(f"❌ Неизвестная ошибка Google Sheets: {e}", exc_info=True)
        raise DatabaseError(f"Google Sheets unknown error: {e}")
    
    return None

# ====================== ОСНОВНОЙ ОБРАБОТЧИК ======================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главная функция обработки сообщений"""
    user_id = update.effective_user.id
    chat_type = update.effective_chat.type

    # 🔥 ОБЯЗАТЕЛЬНО: объявляем переменные
    best_answer = None
    source = "none"
    distance = 1.0
    
    # 🔧 ТЕСТОВЫЙ ЛОГ
    #logger.info(f"🧪 adminlist = {adminlist}")
    #logger.info(f"🧪 user_id = {user_id}, in adminlist? {user_id in adminlist}")
    
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
            #logger.info(f"🚫 БЛОКИРУЮ ЛС от {user_id} (не админ)")
            return
        #logger.info(f"✅ Отвечу админу {user_id} в ЛС")
    
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
    
        cached_answer = response_cache[cache_key]
        emoji = get_source_emoji("cached")
        final_text = f"{cached_answer}\n\n{emoji}"
    
        await send_long_message(
            context.bot, 
            update.effective_chat.id, 
            final_text,
            reply_to_message_id=update.message.message_id
        )
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
    
    # ============ ОСНОВНОЙ ПОИСК С GRACEFUL DEGRADATION ============
    best_answer, source, distance = await robust_search(raw_text, clean_text)
    
    # Если все методы провалились, уведомляем админов
    if source == "error":
        await notify_admins_about_problems(
            context, 
            "Поиск ответов", 
            f"Все методы поиска провалились для запроса: '{raw_text[:50]}...'"
        )
        return
    
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
    
    # Сохраняем в кэш (БЕЗ смайлика)
    response_cache[cache_key] = final_reply

    # Добавляем смайлик только для отправки
    emoji = get_source_emoji(source)
    final_text_with_emoji = f"{final_reply}\n\n{emoji}"

    logger.info(
        f"📤 ОТПРАВКА | source={source} | dist={distance:.3f} | "
        f"len={len(final_reply)} | user={user.id} | "
        f"\"{final_reply[:100]}{'...' if len(final_reply) > 100 else ''}\""
    )

    success = await send_long_message(
        context.bot, 
        update.effective_chat.id, 
        final_text_with_emoji,
        reply_to_message_id=update.message.message_id
    )



    
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
    
    # Получаем статистику кэша эмбеддингов
    try:
        from functools import lru_cache
        general_cache_info = get_embedding_general.cache_info()
        technical_cache_info = get_embedding_technical.cache_info()
        
        embedding_cache = (
            f"General: {general_cache_info.hits}/{general_cache_info.hits + general_cache_info.misses} "
            f"({general_cache_info.currsize}/{general_cache_info.maxsize})\n"
            f"  • Technical: {technical_cache_info.hits}/{technical_cache_info.hits + technical_cache_info.misses} "
            f"({technical_cache_info.currsize}/{technical_cache_info.maxsize})"
        )
    except Exception:
        embedding_cache = "❌ Недоступно"
    
    total = stats['total']
    cached_pct = (stats['cached'] / total * 100) if total > 0 else 0
    vector_pct = (stats['vector'] / total * 100) if total > 0 else 0
    keyword_pct = (stats['keyword'] / total * 100) if total > 0 else 0
    
    # Эффективность бота (сколько запросов обработано без AI)
    efficiency = ((stats['cached'] + stats['keyword']) / total * 100) if total > 0 else 0
    
    text = (
        f"📊 СТАТУС БОТА\n\n"
        f"Состояние: {paused}\n"
        f"Записей в базе:\n"
        f"  • General: {count_general}\n"
        f"  • Technical: {count_technical}\n\n"
        f"📈 Статистика запросов:\n"
        f"Всего: {stats['total']}\n"
        f"  • Из кэша ответов: {stats['cached']} ({cached_pct:.1f}%)\n"
        f"  • Ключевые слова: {stats['keyword']} ({keyword_pct:.1f}%)\n"
        f"  • Векторный поиск: {stats['vector']} ({vector_pct:.1f}%)\n"
        f"  • Groq API: {stats['groq']}\n"
        f"  • Ошибки: {stats['errors']}\n\n"
        f"🚀 Производительность:\n"
        f"  • Эффективность: {efficiency:.1f}% (без AI)\n"
        f"  • Порог вектора: {VECTOR_THRESHOLD}\n\n"
        f"💾 Кэширование:\n"
        f"  • Ответы: {cache_usage} записей\n"
        f"  • Эмбеддинги:\n"
        f"    {embedding_cache}\n\n"
        f"🔔 Alarm-уведомление:\n"
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

async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет последние 200 строк лога"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    try:
        if not os.path.exists(LOG_FILE):
            await update.message.reply_text("❌ Лог-файл не найден")
            return
        
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Берём последние 200 строк
        last_lines = lines[-200:]
        log_text = "".join(last_lines)
        
        # Ограничиваем длину для Telegram
        if len(log_text) > 4000:
            log_text = "...\n" + log_text[-3900:]
        
        await update.message.reply_text(
            f"📋 ПОСЛЕДНИЕ {len(last_lines)} СТРОК ЛОГА:\n\n{log_text}"
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка чтения логов: {e}")
        await update.message.reply_text(f"⚠️ Ошибка: {e}")

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
        "/health — проверка здоровья системы\n"
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
        "Диагностика:\n"
        "/logs — последние 200 строк лога\n\n"
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
    app.add_handler(CommandHandler("health", health_cmd))
    app.add_handler(CommandHandler("clearcache", clear_cache))
    app.add_handler(CommandHandler("addadmin", add_admin_cmd))
    app.add_handler(CommandHandler("removeadmin", remove_admin_cmd))
    app.add_handler(CommandHandler("adminlist", adminlist_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("threshold", set_threshold_cmd))
    app.add_handler(CommandHandler("addalarm", addalarm_cmd))
    app.add_handler(CommandHandler("delalarm", delalarm_cmd))

    
    # ============ ОБРАБОТЧИК ОШИБОК ============
    app.add_error_handler(error_handler)
    
    # ============ ОТЛОЖЕННЫЕ ЗАДАЧИ ============
    async def update_and_test(context: ContextTypes.DEFAULT_TYPE):
        await update_vector_db(context)
        await run_startup_test(context)

    app.job_queue.run_once(update_and_test, when=15)


    
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
