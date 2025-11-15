import telebot
import pandas as pd
import requests
import json
import re
from datetime import datetime
import threading
import os
from dotenv import load_dotenv

# Загрузка переменных из .env
load_dotenv()

# === НАСТРОЙКИ ===
TOKEN = os.getenv('TOKEN')  # Токен от @BotFather
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Ключ с console.groq.com
SHEET_URL = "https://docs.google.com/spreadsheets/d/1HBdZBWjlplVdZ4a7A5hdXxPyb2vyQ68ntIJ-oPfRwhA/export?format=csv"  # Твоя ссылка!

if not TOKEN or not GROQ_API_KEY:
    raise ValueError("❌ Укажите TOKEN и GROQ_API_KEY в .env файле!")

bot = telebot.TeleBot(TOKEN)

# Загрузка базы знаний
def load_knowledge_base():
    try:
        df = pd.read_csv(SHEET_URL)
        print(f"✅ Загружено {len(df)} записей из базы.")
        return df
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return pd.DataFrame(columns=['keywords', 'answer', 'tags', 'priority', 'last_updated'])

df = load_knowledge_base()

# === ПОИСК В БАЗЕ ЗНАНИЙ (с приоритетом) ===
def search_in_kb(question):
    q = question.lower().strip()
    best_match = None
    best_priority = float('inf')  # Ищем самый высокий приоритет (меньшее число)
    
    for _, row in df.iterrows():
        keywords_str = str(row['keywords']).lower()
        keywords = [k.strip() for k in keywords_str.split(',')]
        if any(kw in q for kw in keywords):
            priority = int(row['priority']) if pd.notna(row['priority']) else 999
            if priority < best_priority:
                best_priority = priority
                best_match = row['answer']
    
    return best_match

# === ИИ-ОТВЕТ (Groq + Llama 3.1) ===
def ask_ai(question, kb_context=""):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
Ты — техподдержка. Отвечай кратко, по делу, на русском языке.
Если есть контекст из базы — используй его как основу.

Контекст из базы знаний: {kb_context}

Вопрос: {question}

Ответ (с HTML-тегами для форматирования, если нужно):
    """.strip()

    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 300
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content'].strip()
        else:
            return "⚠️ Ошибка ИИ. Попробуйте позже или напишите @admin."
    except Exception as e:
        print(f"❌ ИИ-ошибка: {e}")
        return "⚠️ Не могу связаться с ИИ. Обратитесь к администратору."

# === ОБРАБОТЧИК СООБЩЕНИЙ ===
@bot.message_handler(func=lambda m: True)
def handle_message(message):
    if message.chat.type not in ['group', 'supergroup']:
        return  # Только в группах

    text = message.text or ""
    bot_username = bot.get_me().username.lower()
    # Активация: упоминание @bot или прямое сообщение
    if not (text.startswith(f'@{bot_username}') or bot_username in text.lower()):
        return

    question = re.sub(r'@[A-Za-z0-9_]+', '', text).strip()  # Убираем @mentions
    if not question or len(question) < 3:
        bot.reply_to(message, "❓ Уточните вопрос, пожалуйста.")
        return

    # 1. Поиск в базе (с контекстом для ИИ)
    kb_answer = search_in_kb(question)
    kb_context = ""  # Для ИИ, если не нашли
    if kb_answer:
        bot.reply_to(message, kb_answer, parse_mode='HTML')
        print(f"✅ KB-ответ на: {question[:50]}...")
        return
    else:
        # Ищем похожие для контекста ИИ
        q_lower = question.lower()
        similar = df[df['keywords'].str.lower().str.contains('|'.join([w for w in q_lower.split() if len(w)>2]), na=False)]
        if not similar.empty:
            kb_context = similar.iloc[0]['answer']  # Берем первый похожий

    # 2. ИИ-ответ
    msg = bot.reply_to(message, "🔍 Ищу решение...", quote=True)
    ai_answer = ask_ai(question, kb_context)
    bot.edit_message_text(ai_answer, message.chat.id, msg.message_id, parse_mode='HTML')
    print(f"🤖 ИИ-ответ на: {question[:50]}...")

# === АВТО-ОБНОВЛЕНИЕ БАЗЫ ===
def update_kb():
    global df
    df = load_knowledge_base()
    print("🔄 База обновлена.")

# Каждые 5 минут
def scheduler():
    update_kb()
    threading.Timer(300.0, scheduler).start()
scheduler()

# === ЗАПУСК ===
if __name__ == "__main__":
    print("🚀 Бот запущен! База: OK.")
    bot.infinity_polling()