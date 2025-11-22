# Этап 1: устанавливаем все зависимости (тяжёлый, но кэшируется автоматически в Docker на сервере)
FROM python:3.12-slim AS builder
WORKDIR /app

# Копируем только requirements.txt, чтобы использовать кэширование зависимостей
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости с кэшированием
RUN pip install --upgrade pip && \
    pip install --cache-dir=/tmp/pip-cache -r requirements.txt

# Этап 2: финальный лёгкий образ (~500–700 МБ вместо 1.5+ ГБ)
FROM python:3.12-slim
WORKDIR /app

# Копируем установленные пакеты и исполнимые файлы из builder'а
COPY --from=builder /usr/local /usr/local  # Копируем системные пакеты и исполнимые файлы

# Копируем код бота
COPY . .

# Указываем переменные окружения для хранилищ
ENV GENERAL_KB_PATH="/app/chroma/general_kb"
ENV TECHNICAL_KB_PATH="/app/chroma/technical_kb"

# Отать Chroma от пересоздания, создаём том для хранилищ
VOLUME /app/chroma

# Установить переменные окружения для работы Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False

# Запуск бота
CMD ["python", "bot.py"]
