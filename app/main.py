from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import requests
import json
from typing import List

from database import get_db, init_db
from models import Methodic
from search import search_methodics, format_methodics_for_prompt
from config import settings

app = FastAPI(title="Methodics Chat Bot", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Инициализация базы данных при запуске
@app.on_event("startup")
def on_startup():
    init_db()


# Модели запросов и ответов
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    max_results: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    found_methodics: int


class MethodicResponse(BaseModel):
    id: int
    title: str
    subject: str
    author: str
    content: str


#TODO: Подобрать модель, изменить обработку(опционально)
def call_groq_api(question: str, context: str) -> str:
    """Groq API - правильная реализация"""
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Ты - помощник по методическим материалам. Ответь на вопрос используя ТОЛЬКО предоставленную информацию.

    КОНТЕКСТ ИЗ МЕТОДИЧЕК:
    {context}

    ВОПРОС: {question}

    Если в контексте нет ответа - вежливо сообщи об этом.
    """

    data = {
        "messages": [
            {
                "role": "system",
                "content": "Ты полезный помощник по методическим материалам. Отвечай точно и по делу."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "llama-3.1-8b-instant",  # ← Используем эту модель
        "temperature": 0.3,
        "max_tokens": 1000,
        "top_p": 0.9
    }

    try:
        print("🔄 Отправляем запрос к Groq API...")
        response = requests.post(settings.GROQ_API_URL, headers=headers, json=data, timeout=30)
        print(f"📡 Статус: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print("✅ Успешный ответ от Groq!")
            return answer
        else:
            print(f"⚠️ Ошибка Groq: {response.status_code} - {response.text[:200]}")
            return create_fallback_response(question, context)

    except Exception as e:
        print(f"💥 Ошибка: {e}")
        return create_fallback_response(question, context)

#TODO: Реализовать ответа faq
def create_fallback_response(question: str, context: str) -> str:
    """Создает умный ответ когда AI недоступен"""
    if not context or "Релевантные методички не найдены" in context:
        return "К сожалению, в базе методичек нет информации по вашему вопросу."

    keywords = {
        "python": "Python - это язык программирования с простым синтаксисом.",
        "программирование": "Программирование - создание компьютерных программ.",
        "база данных": "База данных - организованное хранилище информации.",
        "sql": "SQL - язык для работы с базами данных.",
        "математика": "Математика - наука о числах и пространственных формах."
    }

    question_lower = question.lower()
    for keyword, answer in keywords.items():
        if keyword in question_lower:
            return f"""**Вопрос:** {question}

**Ответ:** {answer}

*Это базовый ответ по ключевому слову. Для более точного ответа дождитесь загрузки AI модели.*"""

    # Если ключевых слов нет, возвращаем контекст
    return f"""**Вопрос:** {question}

**Информация из методичек:**
{context[:800]}...

*AI модель временно недоступна. Это сырая информация из методичек.*"""

@app.post("/chat", response_model=ChatResponse)
async def chat_with_methodics(
        request: ChatRequest,
        db: Session = Depends(get_db)
):
    """
    Основной endpoint для чата с ботом
    """
    methodics = search_methodics(db, request.question, request.max_results)

    context = format_methodics_for_prompt(methodics)

    answer = call_groq_api(request.question, context)

    sources = [
        {
            "id": methodic.id,
            "title": methodic.title,
            "subject": methodic.subject,
            "author": methodic.author
        }
        for methodic in methodics
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        found_methodics=len(methodics)
    )


@app.get("/search", response_model=List[MethodicResponse])
async def search_methodics_endpoint(
        query: str,
        limit: int = 10,
        db: Session = Depends(get_db)
):
    """
    Поиск методичек по запросу (прямой поиск без AI)
    """
    methodics = search_methodics(db, query, limit)
    return [
        MethodicResponse(
            id=methodic.id,
            title=methodic.title,
            subject=methodic.subject,
            author=methodic.author,
            content=methodic.content
        )
        for methodic in methodics
    ]


@app.get("/methodics/{methodic_id}", response_model=MethodicResponse)
async def get_methodic(
        methodic_id: int,
        db: Session = Depends(get_db)
):
    """
    Получение конкретной методички по ID
    """
    methodic = db.query(Methodic).filter(Methodic.id == methodic_id).first()
    if not methodic:
        raise HTTPException(status_code=404, detail="Методичка не найдена")
    return methodic


@app.get("/")
async def root():
    return {"message": "Methodics Chat Bot API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)