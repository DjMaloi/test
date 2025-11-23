# Этап 1: устанавливаем все зависимости
FROM python:3.12-slim AS builder
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Обновляем pip
RUN pip install --upgrade pip

# Ставим остальные зависимости (torch установится с CPU индексом из requirements.txt)
RUN pip install --no-cache-dir torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Этап 2: финальный образ
FROM python:3.12-slim
WORKDIR /app

# Копируем установленные пакеты и исполнимые файлы из builder'а
COPY --from=builder /usr/local /usr/local

# Копируем код бота
COPY bot.py .
COPY requirements.txt .

# Установить переменные окружения для работы Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False

# Создаём директории для persistent volumes
# RUN mkdir -p /app/chroma/general_kb /app/chroma/technical_kb /app/models_cache 
RUN mkdir -p /app/chroma/general_kb /app/chroma/technical_kb /app/models_cache /app/data

# Persistent volumes для Coolify:
# /app/chroma/general_kb - база общих вопросов
# /app/chroma/technical_kb - база технических вопросов
# /app/chroma - общая директория ChromaDB
# /app/models_cache - кэш моделей sentence-transformers
# /app/data - adminlist.json, stats.json, paused.flag
VOLUME ["/app/chroma", "/app/models_cache", "/app/data"]

# Health check для Coolify
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/chroma') else 1)" || exit 1

# Запуск бота
CMD ["python", "bot.py"]

