# Этап 1: устанавливаем все зависимости
FROM python:3.12-slim AS builder
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Обновляем pip
RUN pip install --upgrade pip

# Ставим torch CPU отдельно, чтобы не тянулся CUDA (ставится из requirements файла)
# RUN pip install --no-cache-dir torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Ставим остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Этап 2: финальный образ
FROM python:3.12-slim
WORKDIR /app

# Копируем установленные пакеты и исполнимые файлы из builder'а
COPY --from=builder /usr/local /usr/local

# Копируем код бота
#COPY . .
# Копируем код бота и файлы конфигурации
COPY bot.py .
COPY requirements.txt .
COPY runtime.txt .
COPY README.md .
COPY service_account.json .

# Указываем переменные окружения для хранилищ
ENV GENERAL_KB_PATH="/app/chroma/general_kb"
ENV TECHNICAL_KB_PATH="/app/chroma/technical_kb"

# Отдать Chroma от пересоздания, создаём том для хранилищ
# VOLUME /app/chroma

# Создаём директории для persistent volumes
RUN mkdir -p /app/chroma/general_kb /app/chroma/technical_kb /app/models_cache

# Persistent volumes для Coolify:
# /app/chroma/general_kb - база общих вопросов
# /app/chroma/technical_kb - база технических вопросов
# /app/chroma - общая директория ChromaDB
# /app/models_cache - кэш моделей sentence-transformers
VOLUME ["/app/chroma", "/app/models_cache"]

# Health check для Coolify
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/chroma') else 1)" || exit 1


# Установить переменные окружения для работы Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False

# Запуск бота
CMD ["python", "bot.py"]

