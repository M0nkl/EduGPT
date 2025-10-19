from sqlalchemy.orm import Session
from sqlalchemy import or_
from models import Methodic
import re


def search_methodics(db: Session, query: str, limit: int = 5):
    """
    Поиск методичек по ключевым словам в заголовке и содержании
    """
    keywords = re.findall(r'\w+', query.lower())

    if not keywords:
        return []

    conditions = []
    for keyword in keywords:
        if len(keyword) > 2:
            conditions.append(Methodic.title.ilike(f"%{keyword}%"))
            conditions.append(Methodic.content.ilike(f"%{keyword}%"))
            conditions.append(Methodic.subject.ilike(f"%{keyword}%"))

    if not conditions:
        return []

    results = db.query(Methodic).filter(or_(*conditions)).limit(limit).all()
    return results


def format_methodics_for_prompt(methodics):
    """
    Форматируем найденные методички для промпта
    """
    if not methodics:
        return "Релевантные методички не найдены."

    formatted = []
    for i, methodic in enumerate(methodics, 1):
        formatted.append(f"""
            Методичка #{i}:
            Заголовок: {methodic.title}
            Предмет: {methodic.subject}
            Автор: {methodic.author}
            Содержание: {methodic.content[:500]}...  # Ограничиваем длину для промпта
        """)

    return "\n".join(formatted)