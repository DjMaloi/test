# Этап 1: устанавливаем все зависимости (тяжёлый, но кэшируется автоматически в Docker на сервере)
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Этап 2: финальный лёгкий образ (~500–700 МБ вместо 1.5+ ГБ)
FROM python:3.12-slim
WORKDIR /app

# Копируем только установленные пакеты из builder'а
COPY --from=builder /root/.local /root/.local
ENV PATH="/root/.local/bin:${PATH}"

# Копируем код бота
COPY . .

# Отать Chroma от пересоздания
VOLUME /app/chroma

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False

CMD ["python", "bot.py"]
