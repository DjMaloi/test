# Этап 1: устанавливаем все зависимости
FROM python:3.12-slim AS builder
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip
RUN pip install --cache-dir=/tmp/pip-cache -r requirements.txt

# Добавляем отладочную команду, чтобы увидеть структуру директорий
RUN echo "Проверка содержимого /usr/local:" && ls -la /usr/local

# Этап 2: финальный образ
FROM python:3.12-slim
WORKDIR /app

# Копируем установленные пакеты и исполнимые файлы из builder'а
COPY --from=builder /usr/local /usr/local

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
