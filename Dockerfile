FROM python:3.12-slim
ARG CACHEBUST=2025-11-20-19-00
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONUNBUFFERED=1
USER root

CMD ["python", "bot.py"]
