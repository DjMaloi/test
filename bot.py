import os
import re
import json
import logging
import asyncio
import time
import gc
from hashlib import md5
from typing import Optional, Tuple, List, Dict, Any
from contextlib import asynccontextmanager
from threading import RLock
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
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
GROQ_SEM = asyncio.Semaphore(3)
VECTOR_THRESHOLD = 0.65  # Значение по умолчанию, будет перезаписано при загрузке

MAX_MESSAGE_LENGTH = 4000
CACHE_SIZE = 2000
CACHE_TTL = 7200

CRITICAL_MISMATCHES = {
    "касса": ["киоск", "КСО", "сканер", "принтер чеков", "терминал самообслуживания"],
    "киоск": ["касса", "онлайн-касса", "фискальный регистратор", "терминал оплаты"],
}


def is_mismatch(question: str, answer: str) -> bool:
    """Проверяет, не противоречит ли ответ вопросу используя CRITICAL_MISMATCHES"""
    question_lower = question.lower()
    answer_lower = answer.lower()

    # Проверяем каждую ключевую категорию из CRITICAL_MISMATCHES
    for category, forbidden_terms in CRITICAL_MISMATCHES.items():
        if category.lower() in question_lower:
            for forbidden in forbidden_terms:
                if forbidden.lower() in answer_lower:
                    logger.warning(
                        f"⚠️ НЕСООТВЕТСТВИЕ: вопрос про '{category}', "
                        f"но ответ содержит '{forbidden}'"
                    )
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

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# ====================== CONFIG ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_ID", "").split(",") if x]

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
THRESHOLD_FILE = "/app/data/threshold.json"

# ====================== УПРАВЛЕНИЕ ПОРОГОМ ======================
def load_threshold() -> float:
    """Загружает порог векторного поиска из файла"""
    try:
        if os.path.exists(THRESHOLD_FILE):
            with open(THRESHOLD_FILE, "r") as f:
                data = json.load(f)
                threshold = data.get("threshold", 0.65)
                if 0.0 <= threshold <= 1.0:
                    logger.info(f"🎚️ Загружен порог вектора: {threshold}")
                    return threshold
                else:
                    logger.warning(f"⚠️ Некорректный порог в файле: {threshold}, используем 0.65")
        else:
            logger.info("🎚️ Файл порога не найден, используем значение по умолчанию: 0.65")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки порога: {e}")
    
    return 0.65  # значение по умолчанию

def save_threshold(threshold: float):
    """Сохраняет порог векторного поиска в файл"""
    try:
        os.makedirs(os.path.dirname(THRESHOLD_FILE), exist_ok=True)
        with open(THRESHOLD_FILE, "w") as f:
            json.dump({"threshold": threshold}, f, indent=2)
        logger.info(f"🎚️ Порог сохранён: {threshold}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения порога: {e}")

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
current_alarm: Optional[str] = None
adminlist = set()

def load_adminlist() -> set:
    """Загружает список админов из файла"""
    global adminlist
    try:
        os.makedirs(os.path.dirname(ADMINLIST_FILE), exist_ok=True)
        with open(ADMINLIST_FILE, "r") as f:
            data = json.load(f)
        adminlist = {int(x) for x in data.get("admins", [])}
        return adminlist
    except FileNotFoundError:
        adminlist = set()
        save_adminlist()
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
    "errors": 0,
    "no_answer": 0,
    "quality_good": 0,
    "quality_bad": 0,
    "response_times": [],  # Список времен ответа для расчета среднего
    "last_error_alert": 0,  # Время последнего алерта об ошибках
    "ssl_errors": 0,  # Счетчик SSL ошибок Google Sheets
    "typing_timeouts": 0  # Счетчик таймаутов индикатора "печатает"
}

# Константы для алертов
ERROR_ALERT_THRESHOLD = 0.1  # 10% ошибок от общего числа запросов
ERROR_ALERT_MIN_REQUESTS = 20  # Минимум запросов для проверки
ERROR_ALERT_COOLDOWN = 3600  # 1 час между алертами

# Батчинг для оптимизации сохранения статистики
_stats_dirty = False
_stats_last_save = time.time()
STATS_SAVE_INTERVAL = 30  # Сохранять минимум раз в 30 секунд
STATS_SAVE_THRESHOLD = 10  # Или после 10 изменений

def load_stats():
    """Загружает статистику из файла"""
    global stats
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                loaded = json.load(f)
                stats.update(loaded)
                # Инициализируем отсутствующие поля
                if "response_times" not in stats:
                    stats["response_times"] = []
                if "last_error_alert" not in stats:
                    stats["last_error_alert"] = 0
                if "ssl_errors" not in stats:
                    stats["ssl_errors"] = 0
                if "typing_timeouts" not in stats:
                    stats["typing_timeouts"] = 0
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки статистики: {e}")

def save_stats(force: bool = False):
    """Сохраняет статистику в файл с батчингом"""
    global _stats_dirty, _stats_last_save
    
    if not force:
        _stats_dirty = True
        now = time.time()
        # Сохраняем только если прошло достаточно времени или много изменений
        if (now - _stats_last_save < STATS_SAVE_INTERVAL and 
            stats.get("_change_count", 0) < STATS_SAVE_THRESHOLD):
            stats["_change_count"] = stats.get("_change_count", 0) + 1
            return
    
    try:
        os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
        # Удаляем служебные поля и ограничиваем response_times перед сохранением
        stats_to_save = {k: v for k, v in stats.items() if not k.startswith("_")}
        # Сохраняем только последние 100 времен ответа для экономии места
        if "response_times" in stats_to_save and len(stats_to_save["response_times"]) > 100:
            stats_to_save["response_times"] = stats_to_save["response_times"][-100:]
        with open(STATS_FILE, "w") as f:
            json.dump(stats_to_save, f, indent=2)
        _stats_dirty = False
        _stats_last_save = time.time()
        stats["_change_count"] = 0
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения статистики: {e}")

def track_quality(distance: float, source: str):
    """Отслеживает качество ответов"""
    if source in ["vector_general", "vector_technical"]:
        if distance < 0.5:
            stats["quality_good"] += 1
            logger.info(f"🟢 Отличное совпадение: {distance:.3f}")
        elif distance > 0.8:
            stats["quality_bad"] += 1
            logger.warning(f"🔴 Плохое совпадение: {distance:.3f}")
        else:
            logger.info(f"🟡 Среднее совпадение: {distance:.3f}")

def get_quality_metrics() -> dict:
    """Возвращает метрики качества"""
    total = stats['total']
    if total == 0:
        return {}
    
    success_rate = (stats['cached'] + stats['vector'] + stats['keyword']) / total
    no_answer_rate = stats.get('no_answer', 0) / total
    
    vector_total = stats['vector']
    if vector_total > 0:
        good_rate = stats.get('quality_good', 0) / vector_total
        bad_rate = stats.get('quality_bad', 0) / vector_total
    else:
        good_rate = bad_rate = 0
    
    return {
        'success_rate': success_rate,
        'no_answer_rate': no_answer_rate,
        'vector_good_rate': good_rate,
        'vector_bad_rate': bad_rate,
        'total_requests': total
    }

# ====================== ОПТИМИЗИРОВАННОЕ КЭШИРОВАНИЕ ======================

# Улучшенный LRU кэш с метриками
class AdvancedLRUCache:
    """Продвинутый LRU кэш с метриками и автоматической очисткой"""
    def __init__(self, maxsize: int = 1000, cleanup_ratio: float = 0.8):
        self.maxsize = maxsize
        self.cleanup_ratio = cleanup_ratio
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = RLock()
        
    def get(self, key):
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value  # Перемещаем в конец
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                self._cleanup()
            
            self.cache[key] = value
    
    def _cleanup(self):
        """Удаляет старые элементы до cleanup_ratio от лимита"""
        cleanup_size = int(self.maxsize * (1 - self.cleanup_ratio))
        while len(self.cache) > cleanup_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self):
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%"
            }

# Глобальные кэши с метриками
embedding_cache_general = AdvancedLRUCache(maxsize=2000, cleanup_ratio=0.8)
embedding_cache_technical = AdvancedLRUCache(maxsize=2000, cleanup_ratio=0.8)

# Кэш ответов с метриками
class ResponseCache:
    """Кэш ответов с TTL и метриками"""
    def __init__(self, maxsize: int = 2000, ttl: int = 7200):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
        self.lock = RLock()
    
    def get(self, key):
        with self.lock:
            current_time = time.time()
            
            if key in self.timestamps:
                if current_time - self.timestamps[key] > self.ttl:
                    self._remove(key)
                    self.misses += 1
                    return None
            
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key, value):
        with self.lock:
            current_time = time.time()
            
            if len(self.cache) >= self.maxsize:
                self._cleanup()
            
            self.cache[key] = value
            self.timestamps[key] = current_time
    
    def _cleanup(self):
        """Удаляет просроченные и самые старые записи"""
        current_time = time.time()
        
        expired_keys = [
            key for key, ts in self.timestamps.items()
            if current_time - ts > self.ttl
        ]
        for key in expired_keys:
            self._remove(key)
        
        if len(self.cache) >= self.maxsize:
            sorted_items = sorted(self.timestamps.items(), key=lambda x: x[1])
            cleanup_count = int(self.maxsize * 0.25)
            for key, _ in sorted_items[:cleanup_count]:
                self._remove(key)
    
    def _remove(self, key):
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self):
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "ttl": self.ttl
            }

# Заменяем старые кэши
response_cache = ResponseCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

# Оптимизированные функции эмбеддингов
def get_embedding_general(text: str) -> List[float]:
    """Оптимизированное получение эмбеддинга для General модели"""
    cache_key = f"general_{text}"
    
    cached = embedding_cache_general.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        embedding = embedder_general.encode(text).tolist()
        embedding_cache_general.put(cache_key, embedding)
        return embedding
    except Exception as e:
        logger.error(f"❌ Ошибка эмбеддинга General: {e}")
        raise Exception(f"General embedding error: {e}")

def get_embedding_technical(text: str) -> List[float]:
    """Оптимизированное получение эмбеддинга для Technical модели"""
    cache_key = f"technical_{text}"
    
    cached = embedding_cache_technical.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        embedding = embedder_technical.encode(text).tolist()
        embedding_cache_technical.put(cache_key, embedding)
        return embedding
    except Exception as e:
        logger.error(f"❌ Ошибка эмбеддинга Technical: {e}")
        raise Exception(f"Technical embedding error: {e}")

def get_cache_stats() -> Dict[str, Any]:
    """Возвращает статистику всех кэшей"""
    return {
        "response_cache": response_cache.get_stats(),
        "embedding_general": embedding_cache_general.get_stats(),
        "embedding_technical": embedding_cache_technical.get_stats()
    }

def cleanup_caches():
    """Очищает все кэши и вызывает garbage collector"""
    logger.info("🧹 Начало очистки кэшей...")
    
    response_cache.clear()
    embedding_cache_general.clear()
    embedding_cache_technical.clear()
    
    collected = gc.collect()
    
    logger.info(f"🧹 Очистка завершена. Собрано объектов: {collected}")
    return collected

# ====================== КЭШИРОВАНИЕ ======================

def preprocess(text: str) -> str:
    """Нормализует текст для поиска и кэширования, заменяя синонимы"""
    # Приводим к нижнему регистру
    text = text.lower()

    # Удаление приветствий и вводных
    greetings = [
        "здравствуйте", "привет", "добрый день", "добрый вечер", 
        "доброе утро", "приветствую", "хай", "hello"
    ]
    for g in greetings:
        text = re.sub(rf"\b{g}\b", "", text)

    # ЗАМЕНА СИНОНИМОВ — ключевое обновление
    # Словарь: синоним → основной термин
    synonyms = {
        r'\bкд\b': 'касса доставки',                        # КД → касса доставки
        #r'\bкасса доставки\b': 'кд',           # касса доставки → кд
        #r'\bкасса ресторана\b': 'кр',      # касса ресторана → кр
        r'\bкр\b': 'касса ресторана',                    # КР → касса ресторана
        #r'\bксо\b': 'касса',                      # КСО → касса
        #r'\bсамообслуживани[еяю]\b': 'касса',     # самообслуживание → касса
        #r'\bтерминал самообслуживания\b': 'касса',
        r'\bфискальный регистратор\b': 'фн',
        r'\bонлайн-касса\b': 'касса',
    }

    for pattern, replacement in synonyms.items():
        text = re.sub(pattern, replacement, text)

    # Удаление лишних символов
    text = re.sub(r'[^а-яa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


async def safe_typing(bot, chat_id, max_retries: int = 2):
    """Безопасно отправляет индикатор "печатает" с retry и таймаутом"""
    for attempt in range(max_retries):
        try:
            await asyncio.wait_for(
                bot.send_chat_action(chat_id=chat_id, action="typing"),
                timeout=3.0  # Таймаут 3 секунды
            )
            return  # Успешно отправлено
        except TimedOut:
            if attempt < max_retries - 1:
                logger.debug(f"⏱️ Таймаут индикатора 'печатает' (попытка {attempt + 1}/{max_retries})")
                await asyncio.sleep(0.5)  # Короткая задержка перед повтором
            else:
                stats["typing_timeouts"] = stats.get("typing_timeouts", 0) + 1
                logger.warning(f"⚠️ Не удалось отправить индикатор 'печатает' после {max_retries} попыток")
        except (NetworkError, RetryAfter) as e:
            if attempt < max_retries - 1:
                wait_time = getattr(e, 'retry_after', 1) + 0.5
                logger.debug(f"🌐 Сетевая ошибка индикатора, ждём {wait_time:.1f}с")
                await asyncio.sleep(wait_time)
            else:
                logger.warning(f"⚠️ Сетевая ошибка индикатора 'печатает': {e}")
        except Exception as e:
            # Для других ошибок не повторяем
            logger.debug(f"⚠️ Ошибка индикатора 'печатает': {e}")
            return

# ====================== ОПТИМИЗАЦИЯ ПАРАЛЛЕЛИЗМА ======================
# Пул потоков для CPU-intensive операций
thread_pool = ThreadPoolExecutor(max_workers=4)

# Оптимизированная работа с Google Sheets
class GoogleSheetsPool:
    """Пул подключений к Google Sheets с кэшированием"""
    def __init__(self, max_connections: int = 3):
        self.max_connections = max_connections
        self.semaphore = asyncio.Semaphore(max_connections)
        self._cache = {}
        self._cache_ttl = 300  # 5 минут кэширования
        
    async def get_range(self, range_name: str) -> List[List[str]]:
        """Получает данные из диапазона с кэшированием"""
        cache_key = f"range_{range_name}"
        current_time = time.time()
        
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if current_time - cached_time < self._cache_ttl:
                logger.debug(f"📋 Используем кэш Google Sheets: {range_name}")
                return cached_data
            else:
                logger.warning(f"⚠️ Используем УСТАРЕВШИЙ кэш для {range_name} (просрочен на {(current_time - cached_time):.0f}с)")
                return cached_data  # ✅ Возвращаем даже если просрочен

        async with self.semaphore:
            # Retry для SSL и сетевых ошибок
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    loop = asyncio.get_event_loop()
                    
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            thread_pool,
                            lambda: sheet.values().get(
                                spreadsheetId=SHEET_ID,
                                range=range_name
                            ).execute()
                        ),
                        timeout=15.0  # Таймаут 15 секунд
                    )
                    
                    data = result.get("values", [])
                    
                    self._cache[cache_key] = (data, current_time)
                    
                    if len(self._cache) > 20:
                        self._cleanup_cache()
                    
                    logger.debug(f"📋 Загружено из Google Sheets: {range_name} ({len(data)} строк)")
                    return data
                    
                except asyncio.TimeoutError:
                    last_error = "Timeout"
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Экспоненциальная задержка: 2, 4, 6 сек
                        logger.warning(f"⏱️ Таймаут Google Sheets ({range_name}), попытка {attempt + 1}/{max_retries}, ждём {wait_time}с")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"❌ Таймаут Google Sheets ({range_name}) после {max_retries} попыток")
                except Exception as e:
                    error_str = str(e)
                    last_error = error_str
                    
                    # Проверяем тип ошибки
                    is_ssl_error = "SSL" in error_str or "ssl" in error_str.lower() or "_ssl.c" in error_str
                    is_network_error = "network" in error_str.lower() or "connection" in error_str.lower()
                    
                    if (is_ssl_error or is_network_error) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Экспоненциальная задержка
                        error_type = "SSL" if is_ssl_error else "Network"
                        if is_ssl_error:
                            stats["ssl_errors"] = stats.get("ssl_errors", 0) + 1
                        logger.warning(
                            f"🌐 {error_type} ошибка Google Sheets ({range_name}), "
                            f"попытка {attempt + 1}/{max_retries}, ждём {wait_time}с: {error_str[:100]}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        # Для других ошибок или последней попытки - логируем и пробрасываем
                        logger.error(f"❌ Ошибка Google Sheets ({range_name}): {error_str}")
                        if attempt == max_retries - 1:
                            # На последней попытке возвращаем кэш если есть
                            if cache_key in self._cache:
                                cached_data, _ = self._cache[cache_key]
                                logger.warning(f"⚠️ Используем кэш из-за ошибки: {range_name}")
                                return cached_data
                        raise Exception(f"Google Sheets error: {error_str}")
            
            # Если все попытки исчерпаны и нет кэша
            if cache_key in self._cache:
                cached_data, _ = self._cache[cache_key]
                logger.warning(f"⚠️ Используем устаревший кэш после всех попыток: {range_name}")
                return cached_data
            
            raise Exception(f"Google Sheets error after {max_retries} attempts: {last_error}")
    
    def _cleanup_cache(self):
        """Чистит старые записи в кэше"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, cached_time) in self._cache.items()
            if current_time - cached_time > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def clear_cache(self):
        """Очищает весь кэш"""
        self._cache.clear()

# Глобальный пул для Google Sheets
sheets_pool = GoogleSheetsPool(max_connections=3)

# Оптимизированная функция поиска по ключевым словам
async def optimized_keyword_search(clean_text: str) -> Optional[str]:
    """Оптимизированный поиск по ключевым словам с независимыми источниками"""
    
    # === 1. Поиск в метаданных ChromaDB ===
    tasks = []
    
    async def search_in_metadata(collection, collection_name):
        try:
            if not collection:
                return None
            results = collection.get(
                where={"query": {"$eq": clean_text}},
                include=["metadatas"]
            )
            if results["metadatas"]:
                answer = results["metadatas"][0].get("answer")
                if answer:
                    stats["keyword"] += 1
                    save_stats()
                    logger.info(f"🔑 KEYWORD MATCH (ChromaDB) | query='{clean_text}'")
                    return answer
        except Exception as e:
            logger.warning(f"⚠️ Ошибка поиска в метаданных {collection_name}: {e}")
        return None
    
    if collection_general:
        tasks.append(search_in_metadata(collection_general, "General"))
    if collection_technical:
        tasks.append(search_in_metadata(collection_technical, "Technical"))
    
    # Запускаем без ожидания Google Sheets
    metadata_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Проверяем: найден ли ответ в ChromaDB?
    for result in metadata_results:
        if isinstance(result, Exception):
            logger.warning(f"⚠️ Ошибка в поиске по метаданным: {result}")
        elif result is not None:
            return result  # ✅ Нашли — возвращаем сразу
    
    # === 2. Только если в ChromaDB не нашли — пытаемся Google Sheets (с fallback на кэш) ===
    try:
        general_task = sheets_pool.get_range("General!A:B")
        technical_task = sheets_pool.get_range("Technical!A:B")
        
        general_rows, technical_rows = await asyncio.gather(
            general_task, technical_task,
            return_exceptions=True  # ✅ Не падаем при ошибке
        )
        
        all_rows = []
        
        if isinstance(general_rows, list):
            all_rows.extend(general_rows)
        elif isinstance(general_rows, Exception):
            logger.warning(f"⚠️ Google Sheets (General) недоступны: {general_rows}")
        
        if isinstance(technical_rows, list):
            all_rows.extend(technical_rows)
        elif isinstance(technical_rows, Exception):
            logger.warning(f"⚠️ Google Sheets (Technical) недоступны: {technical_rows}")
        
        for row in all_rows:
            if len(row) >= 2:
                keyword = row[0].strip().lower()
                answer = row[1].strip()
                
                if keyword in clean_text or clean_text in keyword:
                    stats["keyword"] += 1
                    save_stats()
                    logger.info(f"🔑 KEYWORD MATCH (Sheets) | keyword=\"{keyword[:50]}\"")
                    return answer
                        
    except Exception as e:
        logger.error(f"❌ Фатальная ошибка поиска в Google Sheets: {e}")
    
    return None


# ====================== ВЕКТОРНЫЙ ПОИСК ======================
async def search_in_collection(
    collection,
    embedder_type: str,
    query: str,
    threshold: float = None,
    n_results: int = 15
) -> Tuple[Optional[str], float, List[str]]:
    """Универсальная функция векторного поиска с возвратом top_log"""
    if threshold is None:
        threshold = VECTOR_THRESHOLD
    
    if not collection or collection.count() == 0:
        return None, 1.0, []
    
    try:
        embedder_func = get_embedding_general if embedder_type == "general" else get_embedding_technical
        emb = embedder_func(query)
        
        results = collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["metadatas", "distances"]
        )
        
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        
        # Формируем top_log
        top_log = []
        for d, m in zip(distances, metadatas):
            preview = (m.get("answer") or "").replace("\n", " ")[:60]
            top_log.append(f"{d:.3f} → {preview}")
        
        # Логируем
        logger.info(f"🔍 ВЕКТОРНЫЙ ПОИСК: top-3 для '{query[:30]}...'")
        for item in top_log[:3]:
            logger.info(f"   → {item}")
        
        # Лучший результат
        best_answer = None
        best_distance = 1.0
        if distances and distances[0] < threshold:
            best_answer = metadatas[0].get("answer")
            best_distance = distances[0]
        
        return best_answer, best_distance, top_log  # ✅ Возвращаем top_log
        
    except Exception as e:
        logger.error(f"❌ Ошибка векторного поиска: {e}", exc_info=True)
        return None, 1.0, []


# Оптимизированный поиск с параллельными запросами
async def parallel_vector_search(query: str, threshold: float = None) -> Tuple[Optional[str], str, float, List[str]]:
    """Параллельный векторный поиск с возвратом top_log"""
    if threshold is None:
        threshold = VECTOR_THRESHOLD

    tasks = []
    if collection_general and collection_general.count() > 0:
        tasks.append(("vector_general", asyncio.create_task(
            search_in_collection(collection_general, "general", query, threshold)
        )))
    if collection_technical and collection_technical.count() > 0:
        tasks.append(("vector_technical", asyncio.create_task(
            search_in_collection(collection_technical, "technical", query, threshold)
        )))
    
    if not tasks:
        return None, "none", 1.0, []

    results = []
    all_top_logs = []  # Собираем все top_log

    for source_type, task in tasks:
        try:
            answer, distance, top_log = await asyncio.wait_for(task, timeout=10)
            all_top_logs.extend([(source_type, item) for item in top_log])
            if answer and distance < threshold:
                results.append((answer, source_type, distance))
        except Exception as e:
            logger.warning(f"⚠️ Ошибка векторного поиска в {source_type}: {e}")

    if results:
        results.sort(key=lambda x: x[2])
        best_answer, best_source, best_distance = results[0]
        logger.info(f"🎯 ПАРАЛЛЕЛЬНЫЙ ПОИСК: {best_source} | dist={best_distance:.4f}")
        return best_answer, best_source, best_distance, all_top_logs
    
    return None, "none", 1.0, all_top_logs


# ====================== RATE LIMITING ======================
user_requests = defaultdict(deque)
RATE_LIMIT = 10
RATE_WINDOW = 60

def is_rate_limited(user_id: int) -> bool:
    """Проверяет, не превышает ли пользователь лимит запросов"""
    now = time.time()
    user_times = user_requests[user_id]
    
    while user_times and user_times[0] < now - RATE_WINDOW:
        user_times.popleft()
    
    if len(user_times) >= RATE_LIMIT:
        return True
    
    user_times.append(now)
    return False

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
    """Улучшает ответ через Groq, делая его более понятным"""
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
        
        "ЗАПРЕЩЕНО:\n"
        "- НИКОГДА не заменяй 'киоск' на 'кассу' и наоборот.\n"
        "- Не адаптируй термины под вопрос — передавай ответ В ТОЧНОСТИ как есть.\n"
        "- Если в исходном ответе 'киоск' — не меняй на 'кассу', даже если вопрос про кассу.\n\n"
        
        "ФОРМАТ ВЫВОДА:\n"
        "Один связный абзац, без вступлений — только улучшенный ответ."
    )

    user_prompt = f"Исходный ответ:\n{original_answer}\n\nВопрос: {question}\n\nУлучшенный ответ:"
    
    # Используем единую функцию проверки несоответствий
    if is_mismatch(question, original_answer):
        logger.warning("⚠️ Запрет улучшения: обнаружено несоответствие терминов")
        return None
    
    logger.debug(f"✅ Улучшение разрешено: вопрос='{question[:50]}...', ответ='{original_answer[:50]}...'")

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
            
            if 30 < len(improved) <= 800 and len(improved) <= len(original_answer) * 1.2:
                return improved
            
            return None
            
    except Exception as e:
        logger.warning(f"⚠️ Groq улучшение не удалось: {e}")
        return None

async def fallback_groq(question: str) -> Optional[str]:
    """Запрос к Groq когда ничего не найдено в базе"""
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
            
            if not answer or answer.upper().startswith("НЕТ ДАННЫХ") or \
               answer.lower().startswith("не знаю") or len(answer) < 10:
               logger.debug(f"❌ Groq отказался отвечать: '{answer[:100]}'")
               return None
            
            return answer
            
    except Exception as e:
        logger.error(f"❌ Groq fallback ошибка: {e}")
        return None
    
# ====================== ОБНОВЛЕНИЕ БАЗЫ ======================
async def update_vector_db(context: ContextTypes.DEFAULT_TYPE = None):
    """Обновляет векторную базу из Google Sheets с применением preprocess к вопросам"""
    global collection_general, collection_technical
    
    async with collection_lock:
        try:
            logger.info("🔄 Начинаю обновление базы знаний из Google Sheets...")

            # Загрузка данных
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
                    logger.debug(f"🔍 Коллекция {name} не найдена: {e}")

            # Создаём новые коллекции
            collection_general = chroma_client.create_collection("general_kb")
            collection_technical = chroma_client.create_collection("technical_kb")
            
            # Обработка General
            if general_rows:
                valid_rows = [
                    row for row in general_rows 
                    if len(row) >= 2 and row[0].strip()
                ]
                if not valid_rows:
                    logger.warning("🟡 General: нет валидных строк после фильтрации")
                else:
                    original_keys = [row[0].strip() for row in valid_rows]
                    answers = [row[1].strip() for row in valid_rows]
                    processed_keys = [preprocess(key) for key in original_keys]
                    
                    embeddings = embedder_general.encode(original_keys).tolist()
                    
                    collection_general.add(
                        ids=[f"general_{i}" for i in range(len(valid_rows))],
                        documents=original_keys,
                        metadatas=[
                            {"query": processed_keys[i], "answer": answers[i]} 
                            for i in range(len(valid_rows))
                        ],
                        embeddings=embeddings
                    )
                    
                    # Логируем пример
                    logger.info(f"✅ General: добавлено {len(valid_rows)} пар")
                    logger.debug(f"📄 Пример: '{original_keys[0]}' → '{processed_keys[0]}'")
            else:
                logger.info("🟡 General: нет данных для загрузки")

            # Обработка Technical
            if technical_rows:
                valid_rows = [
                    row for row in technical_rows 
                    if len(row) >= 2 and row[0].strip()
                ]
                if not valid_rows:
                    logger.warning("🟡 Technical: нет валидных строк после фильтрации")
                else:
                    original_keys = [row[0].strip() for row in valid_rows]
                    answers = [row[1].strip() for row in valid_rows]
                    processed_keys = [preprocess(key) for key in original_keys]
                    
                    embeddings = embedder_technical.encode(original_keys).tolist()
                    
                    collection_technical.add(
                        ids=[f"technical_{i}" for i in range(len(valid_rows))],
                        documents=original_keys,
                        metadatas=[
                            {"query": processed_keys[i], "answer": answers[i]} 
                            for i in range(len(valid_rows))
                        ],
                        embeddings=embeddings
                    )
                    
                    logger.info(f"✅ Technical: добавлено {len(valid_rows)} пар")
                    logger.debug(f"📄 Пример: '{original_keys[0]}' → '{processed_keys[0]}'")
            else:
                logger.info("🟡 Technical: нет данных для загрузки")

            logger.info("🟢 Обновление векторной базы завершено успешно!")
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при обновлении базы: {e}", exc_info=True)
            stats["errors"] += 1
            save_stats()

def get_source_emoji(source: str) -> str:
    """Возвращает смайлик в зависимости от источника ответа"""
    emoji_map = {
        "cached": "💾",
        "keyword": "🔑",
        "vector_general": "🎯",
        "vector_technical": "⚙️",
        "groq_fallback": "🤖",
        "default_fallback": "❓"
    }
    return emoji_map.get(source, "")

async def run_startup_test(context: ContextTypes.DEFAULT_TYPE):
    """Запускает автопроверку ключевого поиска при старте"""
    logger.info("🧪 Запуск автопроверки ключевого поиска...")

    test_query = "как дела"
    clean_test = preprocess(test_query)

    try:
        results = collection_general.get(
            where={"query": {"$eq": clean_test}},
            include=["metadatas"]
        )

        if results["metadatas"]:
            answer = results["metadatas"][0]["answer"]
            logger.info(f"✅ УСПЕШНЫЙ ТЕСТ: найдено в General → '{answer}'")
        else:
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

# ====================== ОТПРАВКА СООБЩЕНИЙ ======================
async def send_long_message(bot, chat_id: int, text: str, max_retries: int = 3, reply_to_message_id: int = None):
    """Безопасно отправляет длинное сообщение с разбивкой и повторами"""
    for attempt in range(max_retries):
        try:
            chunks = [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]
            for idx, chunk in enumerate(chunks):
                reply_id = reply_to_message_id if idx == 0 else None
                await bot.send_message(
                    chat_id=chat_id, 
                    text=chunk,
                    reply_to_message_id=reply_id
                )

            return True
            
        except RetryAfter as e:
            wait_time = e.retry_after + 1
            logger.warning(f"⏸️ Rate limit, ждём {wait_time}с...")
            await asyncio.sleep(wait_time)
            
        except TimedOut:
            logger.warning(f"⏱️ Таймаут отправки (попытка {attempt + 1}/{max_retries})")
            await asyncio.sleep(2 ** attempt)
            
        except NetworkError as e:
            logger.error(f"🌐 Сетевая ошибка: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(3)
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки: {e}", exc_info=True)
            return False
    
    return False

# ====================== ОПТИМИЗИРОВАННЫЙ ПОИСК ======================
async def optimized_robust_search(query: str, raw_text: str) -> Tuple[Optional[str], str, float]:
    """Оптимизированный надежный поиск с параллельными запросами"""
    search_start = time.time()
    search_timing = {}
    
    clean_text = preprocess(query)
    
    # Попытка 1: Кэш ответов
    t0 = time.time()
    try:
        cache_key = md5(clean_text.encode()).hexdigest()
        cached_answer = response_cache.get(cache_key)
        search_timing["cache"] = time.time() - t0
        if cached_answer:
            stats["cached"] += 1
            save_stats()
            logger.info(f"💾 ОПТИМИЗИРОВАННЫЙ КЭШИРОВАННЫЙ ОТВЕТ")
            return cached_answer, "cached", 0.0
    except Exception as e:
        search_timing["cache"] = time.time() - t0
        logger.warning(f"⚠️ Ошибка кэша: {e}")
    
    # Попытка 2: Оптимизированный параллельный поиск по ключевым словам
    t0 = time.time()
    try:
        keyword_answer = await optimized_keyword_search(clean_text)
        search_timing["keyword"] = time.time() - t0
        if keyword_answer:
            logger.info(f"🔑 ОПТИМИЗИРОВАННЫЙ КЛЮЧЕВОЙ ПОИСК")
            return keyword_answer, "keyword", 0.0
    except Exception as e:
        search_timing["keyword"] = time.time() - t0
        logger.warning(f"⚠️ Ошибка оптимизированного поиска по ключевым словам: {e}")
    
    # Попытка 3: Параллельный векторный поиск
    t0 = time.time()
    try:
        answer, source, distance, _ = await parallel_vector_search(clean_text)  # ✅ Добавлено _
        search_timing["vector"] = time.time() - t0
        if answer and distance < VECTOR_THRESHOLD:
            if not is_mismatch(raw_text, answer):
                stats["vector"] += 1
                save_stats()
                logger.info(f"🎯 ПАРАЛЛЕЛЬНЫЙ ВЕКТОРНЫЙ ПОИСК | dist={distance:.4f}")
                return answer, source, distance
            else:
                logger.warning(f"⚠️ НЕСООТВЕТСТВИЕ в векторном поиске")
    except Exception as e:
        search_timing["vector"] = time.time() - t0
        logger.warning(f"⚠️ Ошибка параллельного векторного поиска: {e}")
        logger.exception(e)  # 🔁 Добавь полный traceback для отладки

    
    # Попытка 4: Groq fallback
    t0 = time.time()
    try:
        groq_answer = await fallback_groq(raw_text)
        search_timing["groq_fallback"] = time.time() - t0
        if groq_answer:
            logger.info(f"🤖 GROQ FALLBACK")
            return groq_answer, "groq_fallback", 1.0
    except Exception as e:
        search_timing["groq_fallback"] = time.time() - t0
        logger.warning(f"⚠️ Ошибка Groq fallback: {e}")
    
    total_search_time = time.time() - search_start
    search_breakdown = " | ".join([
        f"{k}={v:.2f}s" for k, v in sorted(search_timing.items(), key=lambda x: x[1], reverse=True)
        if v > 0.1
    ])
    
    logger.error(
        f"🚨 ВСЕ МЕТОДЫ ПОИСКА ПРОВАЛИЛИСЬ | "
        f"total={total_search_time:.2f}s | "
        f"Breakdown: {search_breakdown if search_breakdown else 'N/A'} | "
        f"запрос: '{query[:50]}...'"
    )
    stats["errors"] += 1
    save_stats()
    
    return None, "error", 1.0

# ====================== UX УЛУЧШЕНИЯ ======================
def get_quick_access_keyboard(chat_type: str = "group") -> InlineKeyboardMarkup:
    """Возвращает клавиатуру быстрого доступа в зависимости от типа чата"""
    if chat_type == "private":
        keyboard = [
            [
                InlineKeyboardButton("🔧 Технические вопросы", callback_data="quick_tech"),
                InlineKeyboardButton("📞 Общие вопросы", callback_data="quick_general")
            ],
            [
                InlineKeyboardButton("⚙️ Настройка кассы", callback_data="quick_cash_setup"),
                InlineKeyboardButton("🖥️ Настройка киоска", callback_data="quick_kiosk_setup")
            ],
            [
                InlineKeyboardButton("💳 Оплата и чеки", callback_data="quick_payment"),
                InlineKeyboardButton("🛠️ Ошибки и сбой", callback_data="quick_errors")
            ],
            [
                InlineKeyboardButton("📊 Статистика бота", callback_data="quick_status"),
                InlineKeyboardButton("🏥 Проверка систем", callback_data="quick_health")
            ]
        ]
    else:
        keyboard = [
            [
                InlineKeyboardButton("🔧 Техническая поддержка", callback_data="quick_tech"),
                InlineKeyboardButton("📞 Общая информация", callback_data="quick_general")
            ],
            [
                InlineKeyboardButton("💳 Вопросы оплаты", callback_data="quick_payment"),
                InlineKeyboardButton("🛠️ Проблемы с оборудованием", callback_data="quick_errors")
            ]
        ]
    
    return InlineKeyboardMarkup(keyboard)

def get_suggested_questions(query_type: str) -> List[str]:
    """Возвращает список предложенных вопросов в зависимости от типа запроса"""
    suggestions = {
        'technical': [
            "Как настроить фискальный регистратор?",
            "Киоск не печатает чеки, что делать?",
            "Ошибка подключения к серверу",
            "Как обновить ПО на кассе?",
            "Принтер чеков не работает"
        ],
        'general': [
            "Время работы поддержки",
            "Контакты технической службы",
            "Стоимость обслуживания",
            "Как заказать оборудование?",
            "График работы компании"
        ],
        'mixed': [
            "Проблемы с оплатой картой",
            "Настройка нового терминала",
            "Обслуживание кассового аппарата",
            "Интеграция с учетной системой",
            "Гарантия и ремонт оборудования"
        ]
    }
    
    return suggestions.get(query_type, suggestions['mixed'])

def get_adaptive_context_message(chat_type: str, user_name: str = "") -> str:
    """Возвращает адаптивное приветствие в зависимости от типа чата"""
    if chat_type == "private":
        if user_name:
            return f"👋 {user_name}, я ваш персональный ассистент поддержки!\n\nВыберите интересующую тему:"
        else:
            return "👋 Я ваш персональный ассистент поддержки!\n\nВыберите интересующую тему:"
    else:
        return "🤖 Бот поддержки готов помочь!\n\nВыберите категорию вопроса или напишите свой вопрос:"

async def handle_quick_access_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает нажатия на кнопки быстрого доступа"""
    query = update.callback_query
    await query.answer()
    
    chat_type = update.effective_chat.type
    user = update.effective_user
    user_name = user.first_name or ""
    
    responses = {
        "quick_tech": (
            "🔧 **Техническая поддержка**\n\n"
            "Задайте ваш вопрос про оборудование:\n"
            "• Настройка касс и киосков\n"
            "• Ошибки и сбои\n"
            "• Подключение устройств\n"
            "• Обновление ПО\n\n"
            "Напишите конкретный вопрос для получения помощи."
        ),
        "quick_general": (
            "📞 **Общая информация**\n\n"
            "Могу ответить на вопросы:\n"
            "• Режим работы компании\n"
            "• Контакты и адреса\n"
            "• Условия обслуживания\n"
            "• Стоимость услуг\n\n"
            "Что вас интересует?"
        ),
        "quick_cash_setup": (
            "⚙️ **Настройка кассы**\n\n"
            "Популярные вопросы:\n"
            "• Как подключить фискальный регистратор?\n"
            "• Настройка реквизитов компании\n"
            "• Подключение принтера чеков\n"
            "• Настройка способов оплаты\n\n"
            "Опишите вашу проблему подробно."
        ),
        "quick_kiosk_setup": (
            "🖥️ **Настройка киоска**\n\n"
            "Помогу с вопросами:\n"
            "• Установка ПО киоска\n"
            "• Настройка сенсорного экрана\n"
            "• Подключение сканера и принтера\n"
            "• Интеграция с платежной системой\n\n"
            "Какой у вас вопрос?"
        ),
        "quick_payment": (
            "💳 **Оплата и чеки**\n\n"
            "Могу помочь с:\n"
            "• Настройкой эквайринга\n"
            "• Проблемами с печатью чеков\n"
            "• Подключением банковских терминалов\n"
            "• Настройкой безналичной оплаты\n\n"
            "Опишите вашу ситуацию."
        ),
        "quick_errors": (
            "🛠️ **Ошибки и сбои**\n\n"
            "Распространенные проблемы:\n"
            "• Касса не включается\n"
            "• Киоск зависает\n"
            "• Ошибка связи с сервером\n"
            "• Принтер не печатает\n\n"
            "Напишите текст ошибки или опишите проблему."
        ),
        "quick_status": "📊 Статистику бота может показать только администратор. Используйте команду /status",
        "quick_health": "🏥 Проверку систем может выполнить только администратор. Используйте команду /health"
    }
    
    response_text = responses.get(query.data, "❓ Неизвестная команда")
    
    if query.data in ["quick_tech", "quick_general", "quick_cash_setup", "quick_kiosk_setup", "quick_payment", "quick_errors"]:
        query_type = "technical" if "tech" in query.data or "cash" in query.data or "kiosk" in query.data or "payment" in query.data or "errors" in query.data else "general"
        suggestions = get_suggested_questions(query_type)
        
        if suggestions:
            response_text += "\n\n**Популярные вопросы:**\n"
            for i, suggestion in enumerate(suggestions[:3], 1):
                response_text += f"{i}. {suggestion}\n"
            response_text += "\nИли напишите свой вопрос:"
    
    await query.edit_message_text(
        text=response_text,
        reply_markup=get_quick_access_keyboard(chat_type),
        parse_mode="Markdown"
    )

def classify_query_type(query: str) -> str:
    """Классифицирует тип запроса для улучшения ответов"""
    technical_keywords = [
        "касса", "киоск", "ксо", "принтер", "сканер", "терминал", 
        "фискальный", "эквайринг", "платеж", "чек", "ошибка", 
        "настройка", "подключение", "обновление", "по"
    ]
    
    general_keywords = [
        "время", "работа", "контакт", "адрес", "стоимость", 
        "цена", "оплата", "доставка", "гарантия", "сервис", 
        "поддержка", "компания", "офис"
    ]
    
    query_lower = query.lower()
    
    technical_score = sum(1 for keyword in technical_keywords if keyword in query_lower)
    general_score = sum(1 for keyword in general_keywords if keyword in query_lower)
    
    if technical_score > general_score:
        return "technical"
    elif general_score > technical_score:
        return "general"
    else:
        return "mixed"

def get_contextual_prompt(query_type: str) -> str:
    """Возвращает контекстный промпт в зависимости от типа запроса"""
    prompts = {
        'technical': (
            "Ты — технический специалист поддержки. Улучши технический ответ, "
            "сохраняя все точные детали.\n\n"
            "ПРАВИЛА УЛУЧШЕНИЯ:\n"
            "1. Сохраняй ВСЕ технические термины: 'касса', 'киоск', 'КСО', 'фискальный регистратор'.\n"
            "2. НЕ заменяй 'касса' ↔ 'киоск'.\n"
            "3. Структурируй техническую информацию чётко.\n"
            "4. Длина — до 800 символов.\n\n"
            "ЦЕЛЬ: Сделать технический ответ понятнее без потери точности."
        ),
        'general': (
            "Ты — редактор поддержки. Улучши общий ответ, делая его дружелюбнее.\n\n"
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

# ====================== ОСНОВНОЙ ОБРАБОТЧИК ======================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главная функция обработки сообщений"""
    user_id = update.effective_user.id
    chat_type = update.effective_chat.type

    best_answer = None
    source = "none"
    distance = 1.0
    
    # ============ ЛОГИКА ДОСТУПА ============
    
    if chat_type in ["group", "supergroup"]:
        if is_admin_special(user_id):
            logger.debug(f"⏭️ Игнор admin {user_id} в группе (из adminlist.json)")
            return
        logger.info(f"✅ Обработаю обычного пользователя {user_id} в группе")
    
    elif chat_type == "private":
        if user_id not in ADMIN_IDS:
            return
    
    if is_paused() and user_id not in ADMIN_IDS:
        return
    
    # Проверка rate limiting (только для обычных пользователей)
    if user_id not in ADMIN_IDS and is_rate_limited(user_id):
        logger.warning(f"⏸️ Rate limit для user={user_id}")
        try:
            await update.message.reply_text(
                "⏸️ Слишком много запросов. Пожалуйста, подождите немного."
            )
        except Exception:
            pass
        return
    
    raw_text = (update.message.text or update.message.caption or "").strip()
    if not raw_text or raw_text.startswith("/") or len(raw_text) > 1500:
        return
    
    user = update.effective_user
    username = f"@{user.username}" if user.username else ""
    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    display_name = f"{name} {username}".strip() or "Без имени"
    
    # Начало отсчета времени ответа
    start_time = time.time()
    timing_breakdown = {}  # Детальное логирование времени на каждом этапе
    
    logger.info(
        f"📨 ЗАПРОС | user={user.id} | {display_name} | "
        f"chat_type={chat_type} | \"{raw_text[:100]}{'...' if len(raw_text) > 100 else ''}\""
    )
    
    stats["total"] += 1
    save_stats()
    
    # Проверка кэша — отвечаем мгновенно, без "печатает"
    t0 = time.time()
    clean_text = preprocess(raw_text)
    cache_key = md5(clean_text.encode()).hexdigest()
    timing_breakdown["preprocess"] = time.time() - t0
    
    # Проверяем кэш через метод get()
    t0 = time.time()
    cached_answer = response_cache.get(cache_key)
    timing_breakdown["cache_check"] = time.time() - t0
    
    if cached_answer is not None:
        stats["cached"] += 1
        save_stats()
        logger.info(f"💾 КЭШИРОВАННЫЙ ОТВЕТ для user={user.id}")
        
        emoji = get_source_emoji("cached")
        final_text = f"{cached_answer}\n\n{emoji}"
        
        t0 = time.time()
        await send_long_message(
            context.bot, 
            update.effective_chat.id, 
            final_text,
            reply_to_message_id=update.message.message_id
        )
        timing_breakdown["send_message"] = time.time() - t0
        return

    # ============ ALARM: отправка системного сообщения ============
    if current_alarm and chat_type in ["group", "supergroup"]:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"🔔 {current_alarm}",
                disable_notification=True
            )
        except Exception as e:
            logger.error(f"❌ Не удалось отправить alarm: {e}")

    t0 = time.time()
    await safe_typing(context.bot, update.effective_chat.id)
    timing_breakdown["typing"] = time.time() - t0
    
    # ============ ОСНОВНОЙ ПОИСК С ОПТИМИЗАЦИЕЙ ============
    t0 = time.time()
    best_answer, source, distance = await optimized_robust_search(raw_text, clean_text)
    timing_breakdown["search"] = time.time() - t0
    
    if source == "error":
        await notify_admins_about_problems(
            context, 
            "Поиск ответов", 
            f"Все методы поиска провалились для запроса: '{raw_text[:50]}...'"
        )
        return
    
    # ============ ЭТАП 5: Улучшение ответа через Groq ============
    final_reply = best_answer
    timing_breakdown["groq_improve"] = 0.0
    
    if best_answer and source in ["vector_general", "vector_technical", "keyword"] and len(best_answer) < 1200:
        t0 = time.time()
        improved = await improve_with_groq(best_answer, raw_text)
        timing_breakdown["groq_improve"] = time.time() - t0
        
        if improved:
            final_reply = improved
            logger.info(
                f"✨ GROQ УЛУЧШИЛ | user={user.id} | "
                f"было={len(best_answer)} → стало={len(improved)}"
            )
    
    # ============ ЭТАП 6: Отправка ответа ============
    if not final_reply:
        query_type = classify_query_type(raw_text)
        suggestions = get_suggested_questions(query_type)
        
        fallback_text = (
            "🤔 К сожалению, я не нашел ответ на ваш вопрос.\n\n"
            "Попробуйте:\n"
            "• Переформулировать вопрос другими словами\n"
            "• Использовать более конкретные термины\n"
            "• Выбрать из популярных вопросов ниже\n\n"
        )
        
        if suggestions:
            fallback_text += "**Популярные вопросы:**\n"
            for i, suggestion in enumerate(suggestions[:3], 1):
                fallback_text += f"{i}. {suggestion}\n"
        
        fallback_text += f"\nИли используйте кнопки быстрого доступа:"
        
        await send_long_message(
            context.bot, 
            update.effective_chat.id, 
            fallback_text,
            reply_to_message_id=update.message.message_id
        )
        
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="🔍 Выберите категорию вопроса:",
            reply_markup=get_quick_access_keyboard(chat_type)
        )
        return
    
    # Сохраняем в кэш (БЕЗ смайлика)
    response_cache.put(cache_key, final_reply)

    emoji = get_source_emoji(source)
    final_text_with_emoji = f"{final_reply}\n\n{emoji}"

    # Отправка сообщения
    t0 = time.time()
    success = await send_long_message(
        context.bot, 
        update.effective_chat.id, 
        final_text_with_emoji,
        reply_to_message_id=update.message.message_id
    )
    timing_breakdown["send_message"] = time.time() - t0
    
    if not success:
        stats["errors"] += 1
        save_stats()
    
    # Общее время ответа
    response_time = time.time() - start_time
    
    # Формируем строку breakdown времени
    breakdown_str = " | ".join([
        f"{k}={v:.2f}s" for k, v in sorted(timing_breakdown.items(), key=lambda x: x[1], reverse=True)
        if v > 0.1  # Показываем только этапы >0.1 сек
    ])
    
    logger.info(
        f"📤 ОТПРАВКА | source={source} | dist={distance:.3f} | "
        f"len={len(final_reply)} | user={user.id} | "
        f"time={response_time:.2f}s"
    )
    
    if breakdown_str:
        logger.info(f"⏱️ TIMING BREAKDOWN: {breakdown_str}")
    
    # Сохраняем время ответа для метрик (храним последние 1000)
    if "response_times" not in stats:
        stats["response_times"] = []
    stats["response_times"].append(response_time)
    if len(stats["response_times"]) > 1000:
        stats["response_times"] = stats["response_times"][-1000:]
    
    # Логируем медленные ответы с детальным breakdown
    if response_time > 3.0:
        logger.warning(
            f"⚠️ Медленный ответ: {response_time:.2f}s для user={user.id} | "
            f"Breakdown: {breakdown_str if breakdown_str else 'N/A'}"
        )
    
    # Критически медленные ответы (>10 секунд) - отправляем алерт админам
    if response_time > 10.0:
        logger.error(
            f"🚨 КРИТИЧЕСКИ МЕДЛЕННЫЙ ОТВЕТ: {response_time:.2f}s для user={user.id} | "
            f"Breakdown: {breakdown_str if breakdown_str else 'N/A'}"
        )
        await notify_admins_about_problems(
            context,
            "Медленный ответ",
            f"Время ответа: {response_time:.2f}s\n"
            f"User: {user.id}\n"
            f"Запрос: {raw_text[:100]}\n"
            f"Breakdown: {breakdown_str if breakdown_str else 'N/A'}"
        )
    
    # Проверяем порог ошибок и отправляем алерт при необходимости
    await check_error_threshold(context)

async def check_error_threshold(context: ContextTypes.DEFAULT_TYPE):
    """Проверяет порог ошибок и отправляет алерт админам при превышении"""
    if not ADMIN_IDS:
        return
    
    total = stats.get("total", 0)
    errors = stats.get("errors", 0)
    
    # Проверяем только если достаточно запросов
    if total < ERROR_ALERT_MIN_REQUESTS:
        return
    
    error_rate = errors / total if total > 0 else 0
    
    # Проверяем порог и кулдаун
    current_time = time.time()
    last_alert = stats.get("last_error_alert", 0)
    
    if error_rate >= ERROR_ALERT_THRESHOLD and (current_time - last_alert) >= ERROR_ALERT_COOLDOWN:
        stats["last_error_alert"] = current_time
        save_stats(force=True)
        
        message = (
            f"🚨 ПРЕВЫШЕН ПОРОГ ОШИБОК\n\n"
            f"📊 Статистика:\n"
            f"• Всего запросов: {total}\n"
            f"• Ошибок: {errors}\n"
            f"• Процент ошибок: {error_rate * 100:.1f}%\n"
            f"• Порог: {ERROR_ALERT_THRESHOLD * 100:.1f}%\n\n"
            f"⏰ Время: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"⚠️ Рекомендуется проверить логи: /logs"
        )
        
        for admin_id in ADMIN_IDS:
            try:
                await context.bot.send_message(
                    chat_id=admin_id,
                    text=message
                )
                logger.warning(f"🚨 Отправлен алерт о превышении порога ошибок админу {admin_id}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось отправить алерт админу {admin_id}: {e}")

async def notify_admins_about_problems(context: ContextTypes.DEFAULT_TYPE, problem_type: str, error_msg: str):
    """Уведомляет админов о проблемах с сервисами"""
    if not ADMIN_IDS:
        return
    
    message = f"🚨 ПРОБЛЕМА С СЕРВИСАМИ\n\nТип: {problem_type}\nОшибка: {error_msg}\n\nВремя: {time.strftime('%H:%M:%S')}"
    
    for admin_id in ADMIN_IDS:
        try:
            await context.bot.send_message(
                chat_id=admin_id,
                text=message
            )
        except Exception as e:
            logger.warning(f"⚠️ Не удалось уведомить админа {admin_id}: {e}")

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
    
    # Получаем размер кэша ответов через метод get_stats()
    try:
        response_stats = response_cache.get_stats()
        cache_usage = f"{response_stats['size']}/{CACHE_SIZE}"
    except Exception:
        cache_usage = f"❌/{CACHE_SIZE}"
    
    try:
        cache_stats = get_cache_stats()
        response_stats = cache_stats["response_cache"]
        general_stats = cache_stats["embedding_general"]
        technical_stats = cache_stats["embedding_technical"]
        
        embedding_cache = (
            f"📊 Ответы: {response_stats['size']}/{response_stats['maxsize']} "
            f"(hit_rate={response_stats['hit_rate']})\n"
            f"  • General: {general_stats['size']}/{general_stats['maxsize']} "
            f"(hit_rate={general_stats['hit_rate']})\n"
            f"  • Technical: {technical_stats['size']}/{technical_stats['maxsize']} "
            f"(hit_rate={technical_stats['hit_rate']})"
        )
    except Exception:
        embedding_cache = "❌ Недоступно"
    
    total = stats['total']
    cached_pct = (stats['cached'] / total * 100) if total > 0 else 0
    vector_pct = (stats['vector'] / total * 100) if total > 0 else 0
    keyword_pct = (stats['keyword'] / total * 100) if total > 0 else 0
    
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
    """Очищает все кэши и оптимизирует память"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    await update.message.reply_text("🧹 Начинаю очистку кэшей...")
    
    # Получаем размеры кэшей через методы get_stats()
    try:
        response_stats = response_cache.get_stats()
        old_response_size = response_stats['size']
    except Exception:
        old_response_size = 0
    
    try:
        general_stats = embedding_cache_general.get_stats()
        old_general_size = general_stats['size']
    except Exception:
        old_general_size = 0
    
    try:
        technical_stats = embedding_cache_technical.get_stats()
        old_technical_size = technical_stats['size']
    except Exception:
        old_technical_size = 0
    
    response_cache.clear()
    embedding_cache_general.clear()
    embedding_cache_technical.clear()
    sheets_pool.clear_cache()
    
    collected = cleanup_caches()
    
    await update.message.reply_text(
        f"🗑️ Все кэши очищены!\n\n"
        f"📊 Удалено записей:\n"
        f"  • Ответы: {old_response_size}\n"
        f"  • General эмбеддинги: {old_general_size}\n"
        f"  • Technical эмбеддинги: {old_technical_size}\n"
        f"  • Google Sheets кэш: очищен\n\n"
        f"🧹 Garbage collector: {collected} объектов\n"
        f"✅ Память оптимизирована!"
    )

async def optimize_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Оптимизирует память и производительность бота"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    await update.message.reply_text("🧠 Начинаю оптимизацию памяти...")
    
    try:
        old_stats = get_cache_stats()
        collected = cleanup_caches()
        sheets_pool._cleanup_cache()
        new_stats = get_cache_stats()
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
        except ImportError:
            memory_mb = 0
        
        message = (
            f"🧠 Оптимизация памяти завершена!\n\n"
            f"📊 Статистика кэшей:\n"
            f"  • Ответы: {new_stats['response_cache']['size']}/{new_stats['response_cache']['maxsize']}\n"
            f"  • General: {new_stats['embedding_general']['size']}/{new_stats['embedding_general']['maxsize']}\n"
            f"  • Technical: {new_stats['embedding_technical']['size']}/{new_stats['embedding_technical']['maxsize']}\n\n"
            f"🧹 Garbage collector: {collected} объектов\n"
            f"💾 Использование памяти: {memory_mb:.1f} MB\n\n"
            f"✅ Производительность оптимизирована!"
        )
        
        await update.message.reply_text(message)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка оптимизации: {e}")

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
        
        for user_id in sorted([int(uid) for uid in adminlist]):
            try:
                user = await context.bot.get_chat(user_id)
                
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

    raw_text = " ".join(context.args)
    import re
    match = re.search(r'"([^"]+)"', raw_text)
    if match:
        text = match.group(1)
    else:
        text = raw_text

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
        
        last_lines = lines[-200:]
        log_text = "".join(last_lines)
        
        if len(log_text) > 4000:
            log_text = "...\n" + log_text[-3900:]
        
        await update.message.reply_text(
            f"📋 ПОСЛЕДНИЕ {len(last_lines)} СТРОК ЛОГА:\n\n{log_text}"
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка чтения логов: {e}")
        await update.message.reply_text(f"⚠️ Ошибка: {e}")

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start - показывает приветствие и кнопки быстрого доступа"""
    chat_type = update.effective_chat.type
    user = update.effective_user
    user_name = user.first_name or ""
    
    welcome_text = get_adaptive_context_message(chat_type, user_name)
    
    await update.message.reply_text(
        text=welcome_text,
        reply_markup=get_quick_access_keyboard(chat_type),
        parse_mode="Markdown"
    )

async def testquery_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Тест векторного поиска: показывает, что находит бот по запросу"""
    if update.effective_user.id not in ADMIN_IDS:
        return

    if not context.args:
        await update.message.reply_text("❌ Использование: /testquery <вопрос>")
        return

    query = " ".join(context.args)
    clean = preprocess(query)

    logger.info(f"🔍 ТЕСТ: запрос='{query}', clean='{clean}'")

    try:
        answer, source, distance, top_log = await parallel_vector_search(clean)
    except Exception as e:
        logger.error(f"❌ Ошибка в parallel_vector_search: {e}")
        await update.message.reply_text(f"❌ Ошибка поиска: {e}")
        return

    # Формируем ответ
    result_text = (
        f"🔍 ТЕСТ ЗАПРОСА\n\n"
        f"📥 Исходный: '{query}'\n"
        f"🧹 Очищенный: '{clean}'\n\n"
        f"🎯 Ответ найден: {'Да' if answer else 'Нет'}\n"
        f"📊 Источник: {source}\n"
        f"📏 Расстояние: {distance:.4f}\n"
        f"🎚️ Порог: {VECTOR_THRESHOLD}"
    )

    if answer:
        result_text += f"\n\n💬 Ответ:\n{answer}"

    if top_log:
        top3 = sorted(top_log, key=lambda x: float(x[1].split()[0]))[:3]
        result_text += f"\n\n📌 ТОП-3 НАЙДЕННЫХ ОТВЕТОВ:"
        for _, item in top3:
            result_text += f"\n→ {item}"

    try:
        await update.message.reply_text(result_text)
    except Exception as e:
        logger.error(f"❌ Не удалось отправить ответ: {e}")
        await update.message.reply_text("❌ Ответ найден, но не удалось отправить (слишком длинный)")



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
        "/metrics — метрики производительности и времени ответа\n"
        "/reload — перезагрузить базу знаний\n\n"
        "/testquery вопрос - тест векторного поиска\n\n"
        "Управление кэшем:\n"
        "/clearcache — очистить кэш ответов\n"
        "/optimize — оптимизировать память\n\n"
        "Управление уведомлениями:\n"
        "/addalarm \"текст\" — установить уведомление при каждом сообщении\n"
        "/delalarm — удалить уведомление\n\n"
        "Управление администраторами:\n"
        "/addadmin [user_id] — добавить в adminlist\n"
        "/removeadmin <user_id> — удалить из adminlist\n"
        "/adminlist — показать список\n\n"
        "/help — показать это меню\n\n"
        "Диагностика:\n"
        "/logs — последние 200 строк лога\n"
        "/metrics — метрики производительности и времени ответа\n"
        "/threshold <число> — установить порог векторного поиска (0.0–1.0)\n\n"
        "🔔 **Алерты:**\n"
        "Бот автоматически уведомляет админов при превышении порога ошибок (10%)\n"
        "после минимум 20 запросов. Кулдаун между алертами: 1 час.\n\n"
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
        save_threshold(new_threshold)
        
        await update.message.reply_text(
            f"✅ Порог изменён: {old_threshold} → {new_threshold}\n\n"
            f"⚠️ Это изменение временное (до перезапуска бота)"
        )
        
        logger.info(f"🎚️ Порог изменён: {old_threshold} → {new_threshold}")
        
    except ValueError:
        await update.message.reply_text("❌ Неверный формат числа")

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
        test_text = "тест"
        general_emb = get_embedding_general(test_text)
        technical_emb = get_embedding_technical(test_text)
        
        general_cache = embedding_cache_general.get_stats()
        technical_cache = embedding_cache_technical.get_stats()
        
        return {
            "status": "✅ OK",
            "general_cache": f"{general_cache['size']}/{general_cache['maxsize']}",
            "technical_cache": f"{technical_cache['size']}/{technical_cache['maxsize']}",
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
    
    sheets_task = asyncio.create_task(check_google_sheets_health())
    groq_task = asyncio.create_task(check_groq_health())
    
    sheets_result = await sheets_task
    groq_result = await groq_task
    
    chromadb_result = check_chromadb_health()
    embedding_result = check_embedding_models_health()
    
    all_ok = all([
        sheets_result["status"] == "✅ OK",
        groq_result["status"] == "✅ OK", 
        chromadb_result["status"] == "✅ OK",
        embedding_result["status"] == "✅ OK"
    ])
    
    overall_status = "✅ Все системы работают" if all_ok else "⚠️ Есть проблемы"
    
    return {
        "overall": overall_status,
        "google_sheets": sheets_result,
        "groq": groq_result,
        "chromadb": chromadb_result,
        "embeddings": embedding_result
    }

async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда проверки здоровья системы"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    await update.message.reply_text("🔍 Проверяю состояние систем...")
    
    try:
        health_results = await run_health_checks()
        
        message = (
            f"🏥 **HEALTH CHECK**\n\n"
            f"📊 Общий статус: {health_results['overall']}\n\n"
            f"📋 **Google Sheets:**\n"
            f"Статус: {health_results['google_sheets']['status']}\n"
            f"Время ответа: {health_results['google_sheets']['response_time']}\n"
            f"Ошибка: {health_results['google_sheets']['error'] or 'Нет'}\n\n"
            f"🤖 **Groq API:**\n"
            f"Статус: {health_results['groq']['status']}\n"
            f"Время ответа: {health_results['groq']['response_time']}\n"
            f"Ошибка: {health_results['groq']['error'] or 'Нет'}\n\n"
            f"🗄️ **ChromaDB:**\n"
            f"Статус: {health_results['chromadb']['status']}\n"
            f"General записей: {health_results['chromadb']['general_records']}\n"
            f"Technical записей: {health_results['chromadb']['technical_records']}\n"
            f"Ошибка: {health_results['chromadb']['error'] or 'Нет'}\n\n"
            f"🧠 **Модели эмбеддингов:**\n"
            f"Статус: {health_results['embeddings']['status']}\n"
            f"General кэш: {health_results['embeddings']['general_cache']}\n"
            f"Technical кэш: {health_results['embeddings']['technical_cache']}\n"
            f"Ошибка: {health_results['embeddings']['error'] or 'Нет'}"
        )
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"❌ Ошибка health check: {e}")
        await update.message.reply_text(f"❌ Ошибка проверки здоровья: {e}")

async def metrics_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для просмотра метрик производительности"""
    if update.effective_user.id not in ADMIN_IDS:
        return
    
    total = stats.get("total", 0)
    errors = stats.get("errors", 0)
    response_times = stats.get("response_times", [])
    
    # Расчет метрик времени ответа
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        # Медиана
        sorted_times = sorted(response_times)
        mid = len(sorted_times) // 2
        median_time = sorted_times[mid] if len(sorted_times) % 2 else (sorted_times[mid-1] + sorted_times[mid]) / 2
        
        # Процентили
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        p95_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
        p99_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]
        
        # Медленные ответы (>3 сек)
        slow_responses = sum(1 for t in response_times if t > 3.0)
        slow_percent = (slow_responses / len(response_times)) * 100
    else:
        avg_time = min_time = max_time = median_time = p95_time = p99_time = 0.0
        slow_responses = 0
        slow_percent = 0.0
    
    # Процент ошибок
    error_rate = (errors / total * 100) if total > 0 else 0.0
    
    # Статус алертов
    alert_status = "🟢 Норма"
    if total >= ERROR_ALERT_MIN_REQUESTS:
        if error_rate >= ERROR_ALERT_THRESHOLD * 100:
            alert_status = "🔴 Превышен порог"
        elif error_rate >= ERROR_ALERT_THRESHOLD * 50:
            alert_status = "🟡 Близко к порогу"
    
    message = (
        f"📊 **МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ**\n\n"
        f"📈 **Общая статистика:**\n"
        f"• Всего запросов: {total}\n"
        f"• Ошибок: {errors}\n"
        f"• Процент ошибок: {error_rate:.2f}%\n"
        f"• Статус алертов: {alert_status}\n\n"
        f"⏱️ **Время ответа:**\n"
        f"• Среднее: {avg_time:.3f}s\n"
        f"• Медиана: {median_time:.3f}s\n"
        f"• Минимум: {min_time:.3f}s\n"
        f"• Максимум: {max_time:.3f}s\n"
        f"• 95-й процентиль: {p95_time:.3f}s\n"
        f"• 99-й процентиль: {p99_time:.3f}s\n"
        f"• Медленных (>3s): {slow_responses} ({slow_percent:.1f}%)\n\n"
        f"📋 **Настройки алертов:**\n"
        f"• Порог ошибок: {ERROR_ALERT_THRESHOLD * 100:.1f}%\n"
        f"• Минимум запросов: {ERROR_ALERT_MIN_REQUESTS}\n"
        f"• Кулдаун: {ERROR_ALERT_COOLDOWN // 60} мин\n\n"
        f"💡 Используйте /status для детальной статистики"
    )
    
    await update.message.reply_text(message, parse_mode="Markdown")

# ====================== ОБРАБОТЧИК ОШИБОК ======================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    logger.error(f"❌ Необработанная ошибка: {context.error}", exc_info=context.error)
    
    stats["errors"] += 1
    save_stats()
    
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
    
    save_stats(force=True)  # Принудительно сохраняем статистику
    save_adminlist()
    
    logger.info("💾 Все данные сохранены")
    logger.info("👋 Бот остановлен")

# ====================== ЗАПУСК БОТА ======================
if __name__ == "__main__":
    logger.info("🚀 Запуск бота...")
    
    logger.info(f"🧪 ТЕСТ preprocess('кд'): '{preprocess('кд')}'")
    logger.info(f"🧪 ТЕСТ preprocess('касса доставки'): '{preprocess('касса доставки')}'")

    VECTOR_THRESHOLD = load_threshold()
    
    adminlist = load_adminlist()
    logger.info(f"📋 Текущих админов в списке: {len(adminlist)}")
    load_stats()
    
    current_alarm = load_alarm()

    app = Application.builder()\
        .token(TELEGRAM_TOKEN)\
        .concurrent_updates(False)\
        .build()
    
    # ============ ФИЛЬТРЫ СООБЩЕНИЙ ============
    
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & 
        ~filters.COMMAND & 
        ~filters.User(user_id=ADMIN_IDS),
        block_private
    ))
    
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & (
            (filters.ChatType.PRIVATE & filters.User(user_id=ADMIN_IDS)) |
            (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP)
        ),
        handle_message
    ))
    
    app.add_handler(MessageHandler(
        filters.CAPTION & ~filters.COMMAND & (
            (filters.ChatType.PRIVATE & filters.User(user_id=ADMIN_IDS)) |
            (filters.ChatType.GROUPS | filters.ChatType.SUPERGROUP)
        ),
        handle_message
    ))
    
    # ============ КОМАНДЫ АДМИНИСТРАТОРА ============
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("reload", reload_kb))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("health", health_cmd))
    app.add_handler(CommandHandler("metrics", metrics_cmd))
    app.add_handler(CommandHandler("clearcache", clear_cache))
    app.add_handler(CommandHandler("optimize", optimize_memory))
    app.add_handler(CommandHandler("addadmin", add_admin_cmd))
    app.add_handler(CommandHandler("removeadmin", remove_admin_cmd))
    app.add_handler(CommandHandler("adminlist", adminlist_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("threshold", set_threshold_cmd))
    app.add_handler(CommandHandler("testquery", testquery_cmd))
    app.add_handler(CommandHandler("addalarm", addalarm_cmd))
    app.add_handler(CommandHandler("delalarm", delalarm_cmd))
    
    # ============ ОБРАБОТЧИК КНОПОК ============
    app.add_handler(CallbackQueryHandler(handle_quick_access_callback))
    
    # ============ ОБРАБОТЧИК ОШИБОК ============
    app.add_error_handler(error_handler)
    
    # ============ ОТЛОЖЕННЫЕ ЗАДАЧИ ============
    async def update_and_test(context: ContextTypes.DEFAULT_TYPE):
        await update_vector_db(context)
        await run_startup_test(context)

    app.job_queue.run_once(update_and_test, when=15)
    
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
        import asyncio
        asyncio.run(shutdown(app))
