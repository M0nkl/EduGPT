from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from models import MethodicEntry, QAEntry
import re
from difflib import SequenceMatcher


def calculate_similarity(text1: str, text2: str) -> float:
    """Вычисляет схожесть двух текстов от 0 до 1"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def search_qa_entries(db: Session, question: str, threshold: float = 0.6, limit: int = 3):
    """
    Ищет наиболее похожие вопросы в таблице qa_entries
    Возвращает готовые ответы, если найдены похожие вопросы
    """
    # Очищаем и токенизируем вопрос пользователя
    question_clean = re.sub(r'\s+', ' ', question.lower()).strip()

    # Получаем все вопросы из базы
    all_qa = db.query(QAEntry).all()

    # Вычисляем схожесть для каждого вопроса
    qa_with_similarity = []
    for qa in all_qa:
        qa_question_clean = re.sub(r'\s+', ' ', qa.question.lower()).strip()
        similarity = calculate_similarity(question_clean, qa_question_clean)

        if similarity >= threshold:
            qa_with_similarity.append({
                'qa': qa,
                'similarity': similarity
            })

    # Сортируем по схожести и возвращаем лучшие
    qa_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)

    return [item['qa'] for item in qa_with_similarity[:limit]]


def search_methodic_texts(db: Session, query: str, limit: int = 5):
    """
    Поиск в полных текстах методичек (methodic_text)
    """
    keywords = re.findall(r'\w+', query.lower())

    if not keywords:
        return []

    conditions = []
    for keyword in keywords:
        if len(keyword) > 2:
            conditions.append(MethodicEntry.methodic_text.ilike(f"%{keyword}%"))
            conditions.append(MethodicEntry.source_title.ilike(f"%{keyword}%"))
            conditions.append(MethodicEntry.author.ilike(f"%{keyword}%"))

    if not conditions:
        return []

    # Используем DISTINCT или группировку чтобы избежать дубликатов
    results = db.query(MethodicEntry).filter(or_(*conditions)).limit(limit).all()
    return results


def find_relevant_sentences(text: str, question: str, max_sentences: int = 3) -> list:
    """
    Улучшенный поиск релевантных предложений с учетом контекста вопроса
    """
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z0-9])', text)

    keywords = re.findall(r'\w+', question.lower())
    keywords = [k for k in keywords if len(k) > 3]  # Более длинные слова

    relevant_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        score = 0

        # 1. Ключевые слова
        for keyword in keywords:
            if keyword in sentence_lower:
                score += 3

        # 2. Тематические слова (можно расширить)
        thematic_words = ['студент', 'обучен', 'преподава', 'образован', 'метод', 'технолог']
        for word in thematic_words:
            if word in sentence_lower:
                score += 1

        # 3. Длина предложения (предпочитаем средние)
        word_count = len(sentence.split())
        if 10 <= word_count <= 30:
            score += 1

        if score > 0:
            relevant_sentences.append({
                'sentence': sentence.strip(),
                'score': score,
                'word_count': word_count
            })

    relevant_sentences.sort(key=lambda x: (x['score'], -x['word_count']), reverse=True)

    return [s['sentence'] for s in relevant_sentences[:max_sentences]]


def search_methodics_with_context(db: Session, question: str, limit: int = 5):
    """
    Основная функция поиска
    """
    qa_results = search_qa_entries(db, question, limit=limit)

    methodic_results = search_methodic_texts(db, question, limit)

    methodic_contexts = []
    for methodic in methodic_results:
        if methodic.methodic_text:
            relevant_sentences = find_relevant_sentences(
                methodic.methodic_text,
                question,
                max_sentences=3
            )

            if relevant_sentences:
                methodic_contexts.append({
                    'methodic': methodic,
                    'relevant_sentences': relevant_sentences,
                    'score': len(relevant_sentences)
                })

    methodic_contexts.sort(key=lambda x: x['score'], reverse=True)

    return {
        'qa_results': qa_results,
        'methodic_contexts': methodic_contexts
    }


def format_context_for_prompt(search_results: dict, question: str) -> str:
    """
    Улучшенное форматирование контекста для промпта Gemini
    """
    formatted_parts = []

    # 1. Форматируем Q&A результаты
    if search_results['qa_results']:
        formatted_parts.append("=== НАЙДЕННЫЕ ГОТОВЫЕ ОТВЕТЫ ===")
        for i, qa in enumerate(search_results['qa_results'], 1):
            formatted_parts.append(f"\nВОПРОС {i}: {qa.question}")
            formatted_parts.append(f"ОТВЕТ {i}: {qa.answer}")
            if qa.methodic:
                formatted_parts.append(f"(Из методички: {qa.methodic.source_title})")

    # 2. Форматируем контекст из методичек с указанием релевантности
    if search_results['methodic_contexts']:
        formatted_parts.append("\n=== РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ ИЗ МЕТОДИЧЕК ===")

        for i, ctx in enumerate(search_results['methodic_contexts'], 1):
            methodic = ctx['methodic']

            formatted_parts.append(f"\n──── МЕТОДИЧКА {i} ────")
            formatted_parts.append(f"Заголовок: {methodic.source_title or 'Без названия'}")
            if methodic.author:
                formatted_parts.append(f"Автор: {methodic.author}")

            formatted_parts.append(f"Релевантные фрагменты:")
            for j, sentence in enumerate(ctx['relevant_sentences'], 1):
                clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                if len(clean_sentence) > 300:
                    clean_sentence = clean_sentence[:300] + "..."
                formatted_parts.append(f"  {j}. {clean_sentence}")

    if not formatted_parts:
        return "По вашему запросу ничего не найдено."

    instruction = f"""
ВАЖНАЯ ИНСТРУКЦИЯ:
На основе приведенных выше материалов ответьте на вопрос: "{question}"

ТРЕБОВАНИЯ К ОТВЕТУ:
1. Отвечайте ТОЛЬКО на основе предоставленных материалов
2. Не добавляйте информацию "от себя"
3. Если информации недостаточно, скажите об этом
4. Структурируйте ответ по пунктам
5. Цитируйте источники (Методичка 1, Методичка 2 и т.д.)
"""

    formatted_parts.append(instruction)

    return "\n".join(formatted_parts)