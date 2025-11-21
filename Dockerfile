# Stage 1: builder (тяжёлый, но кэшируется навсегда)
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y build-essential gcc g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: лёгкий финальный образ (~300–400 МБ вместо 1.5+ ГБ)
FROM python:3.12-slim
WORKDIR /app
# Копируем только установленные пакеты из builder'а
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
CMD ["python", "bot.py"]
