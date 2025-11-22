# Этап 1: устанавливаем все зависимости
FROM python:3.12-slim AS builder
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Обновляем pip
RUN pip install --upgrade pip

# Ставим torch CPU отдельно, чтобы не тянулся CUDA
RUN pip install --no-cache-dir torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Ставим остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

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

# Отдать Chroma от пересоздания, создаём том для хранилищ
VOLUME /app/chroma

# Установить переменные окружения для работы Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False

# Запуск бота
CMD ["python", "bot.py"]
