# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import requests
import re

from database import get_db, init_db
from models import Methodic
from search import search_methodics, format_methodics_for_prompt
from config import settings
from pydantic import BaseModel

app = FastAPI(title="Methodics Chat Bot (Strict Mode)", version="2.6.0")

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------ DB INIT ------------------
@app.on_event("startup")
def on_startup():
    init_db()


# ------------------ MODELS ------------------
class ChatRequest(BaseModel):
    question: str
    max_results: int = 5
    full: bool = False


class MethodicSnippet(BaseModel):
    id: int
    title: str
    subject: Optional[str] = None
    author: Optional[str] = None
    content_snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[MethodicSnippet]
    found_methodics: int


class MethodicCreate(BaseModel):
    title: str
    content: str
    subject: Optional[str] = None
    author: Optional[str] = None


# ------------------ HELPERS ------------------
def format_sources(methodics: List[Methodic], full: bool = False) -> List[MethodicSnippet]:
    """Формирует список источников (отрезки или полные тексты)."""
    sources = []
    for m in methodics:
        text = m.content if full else (m.content[:400] + "..." if len(m.content) > 400 else m.content)
        sources.append(
            MethodicSnippet(
                id=m.id,
                title=m.title,
                subject=m.subject,
                author=m.author,
                content_snippet=text
            )
        )
    return sources


def extract_from_methodics(question: str, methodics: List[Methodic]) -> Optional[str]:
    """
    Ищет ключевые слова из вопроса в тексте методичек.
    Если найдено предложение с совпадением — возвращает его.
    """
    if not methodics:
        return None

    keywords = re.findall(r'\w+', question.lower())
    keywords = [k for k in keywords if len(k) > 2]
    if not keywords:
        return None

    found_sentences = []
    for m in methodics:
        text = m.content or ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            for kw in keywords:
                if kw in s.lower():
                    found_sentences.append(f"Из методички «{m.title}» ({m.author}): {s.strip()}")
                    break

    if found_sentences:
        return "\n\n".join(found_sentences[:3])
    return None

def clean_text(text: str) -> str:
    """Очищаем текст: лишние пробелы и переносы строк"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ------------------ GEMINI API ------------------
def call_gemini_api(question: str, context: str) -> str:
    """
    Gemini используется только как инструмент для обобщения контекста.
    Модель не имеет права добавлять собственные знания.
    Если данных нет — должна вежливо сообщить об этом.
    """
    instruction = (
        "Ты — помощник по методическим материалам. "
        "Отвечай строго по приведённому контексту. "
        "Если в контексте нет ответа — вежливо сообщи, "
        "что данных по запросу в базе методичек нет, "
        "и предложи пользователю переформулировать вопрос "
        "или обратиться к администратору."
    )

    user_prompt = f"""
КОНТЕКСТ:
{context[:6000]}

ВОПРОС:
{question}
"""

    url = f"{settings.GEMINI_API_URL}?key={settings.GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": instruction}, {"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1000,
            "topP": 0.9
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        if resp.status_code != 200:
            print(f"⚠️ Ошибка Gemini: {resp.status_code} - {resp.text[:200]}")
            return ""
        data = resp.json()
        answer = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        return answer
    except Exception as e:
        print(f"💥 Ошибка обращения к Gemini: {e}")
        return ""


# ------------------ MAIN LOGIC ------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_with_methodics(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Алгоритм:
    1. Поиск методичек по вопросу.
    2. Если методички найдены — попытка извлечь ответ локально.
    3. Если локально не найдено — Gemini формулирует ответ только на основе контекста.
    4. Если ничего не найдено — возвращается понятное сообщение с рекомендациями.
    """
    methodics = search_methodics(db, request.question, request.max_results)

    # --- Сценарий 1: нет методичек ---
    if not methodics:
        print("❌ Методички не найдены.")
        answer = (
            "По вашему запросу в базе методических материалов ничего не найдено. "
            "Проверьте формулировку вопроса или уточните тему. "
            "Если проблема повторяется, обратитесь к администратору."
        )
        return ChatResponse(answer=answer, sources=[], found_methodics=0)

    sources = format_sources(methodics, request.full)

    # --- Сценарий 2: есть методички, пробуем найти локально ---
    local_answer = extract_from_methodics(request.question, methodics)
    if local_answer:
        print("✅ Найден ответ внутри методичек.")
        return ChatResponse(answer=local_answer, sources=sources, found_methodics=len(methodics))

    # --- Сценарий 3: методички есть, но локально не найдено — Gemini обобщает контекст ---
    print("⚙️ Не найдено точного ответа, формируем контекст для Gemini.")
    context = format_methodics_for_prompt(methodics)
    gemini_answer = call_gemini_api(request.question, context)

    # --- Проверка результата Gemini ---
    if not gemini_answer or len(gemini_answer.strip()) < 15:
        print("⚠️ Gemini не дал содержательного ответа.")
        gemini_answer = (
            "Ответ на данный вопрос отсутствует в базе методических материалов. "
            "Рекомендуется переформулировать запрос или уточнить тему. "
            "При необходимости обратитесь к администратору для добавления новых данных."
        )

    return ChatResponse(answer=gemini_answer, sources=sources, found_methodics=len(methodics))


# ------------------ /search ------------------
@app.get("/search", response_model=List[MethodicSnippet])
async def search_methodics_endpoint(
    query: str = Query(..., description="Поисковый запрос"),
    limit: int = Query(10, description="Максимальное количество результатов"),
    db: Session = Depends(get_db)
):
    methodics = search_methodics(db, query, limit)
    return format_sources(methodics, full=False)


# ------------------ /methodics/{id} ------------------
@app.get("/methodics/{methodic_id}", response_model=MethodicSnippet)
async def get_methodic(methodic_id: int, db: Session = Depends(get_db)):
    methodic = db.query(Methodic).filter(Methodic.id == methodic_id).first()
    if not methodic:
        raise HTTPException(status_code=404, detail="Методичка не найдена")
    return MethodicSnippet(
        id=methodic.id,
        title=methodic.title,
        subject=methodic.subject,
        author=methodic.author,
        content_snippet=methodic.content
    )


# ------------------ /upload_methodic ------------------
@app.post(
    "/upload_methodic",
    response_model=MethodicSnippet,
    summary="Upload new methodic",
    description="""
Добавляет новую методичку в базу данных.  

**Доступные варианты:**  
1. **JSON** – передаем title, content, subject, author  
2. **Форма + файл .txt** – передаем title, subject, author и файл с текстом методички  
"""
)
async def upload_methodic(
    methodic: Optional[MethodicCreate] = Body(
        None,
        description="Используйте JSON для передачи полей"
    ),
    title: Optional[str] = Form(None, description="Название методички"),
    content: Optional[str] = Form(None, description="Содержимое методички"),
    subject: Optional[str] = Form(None, description="Предмет методички"),
    author: Optional[str] = Form(None, description="Автор методички"),
    file: Optional[UploadFile] = File(None, description="Файл .txt с содержимым методички"),
    db: Session = Depends(get_db)
):
#JSON-данные
    if methodic:
        title = methodic.title
        content = methodic.content
        subject = methodic.subject
        author = methodic.author

#Данные из файла
    if file:
        if not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Только файлы .txt поддерживаются")
        if file.spool_max_size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Файл слишком большой (>5MB)")
        file_content = await file.read()
        if not file_content.strip():
            raise HTTPException(status_code=400, detail="Файл пустой")
        content = file_content.decode("utf-8") if not content else content

#Проверка обязательных полей
    if not title or not content:
        raise HTTPException(status_code=400, detail="Нужно указать title и content")

#Очистка текста
    content = clean_text(content)

#Создание записи
    new_methodic = Methodic(
        title=title,
        content=content,
        subject=subject,
        author=author
    )
    db.add(new_methodic)
    db.commit()
    db.refresh(new_methodic)

    return MethodicSnippet(
        id=new_methodic.id,
        title=new_methodic.title,
        subject=new_methodic.subject,
        author=new_methodic.author,
        content_snippet=new_methodic.content
    )

# ------------------ ROOT ------------------
@app.get("/")
async def root():
    return {"message": "Methodics Chat Bot API (Strict Mode)", "version": "2.6.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
