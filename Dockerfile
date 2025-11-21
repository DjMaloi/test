FROM python:3.12-slim AS base

WORKDIR /app
COPY requirements.txt .

# Используем кэш pip для ещё большей скорости
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "bot.py"]
