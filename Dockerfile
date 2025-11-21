# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# Отключаем telemetry ChromaDB (чтобы не ставить posthog вообще)
ENV ANONYMIZED_TELEMETRY=False
ENV POSTHOG_API_KEY=
ENV POSTHOG_HOST=

WORKDIR /app

# Копируем только зависимости
COPY requirements.txt .

# Максимальный кэш pip + torch
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/torch \
    pip install --no-cache-dir -r requirements.txt

# Копируем код (только после установки зависимостей!)
COPY . .

# Папка для ChromaDB (чтобы не терялась база при перезапуске контейнера)
# В Coolify сделай volume на /app/chroma
VOLUME /app/chroma

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "bot.py"]
